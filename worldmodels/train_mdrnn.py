from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
import yaml
import numpy as np
from functools import partial
from tqdm import tqdm
from torch.nn import functional as f
from torchvision import transforms
from dataload.action_dataset import RolloutSequenceDataset
from eval_utils.visdom_plotter import VisdomLinePlotter
from torch.utils.data import DataLoader
from models.VAE import VAE
from models.MD_RNN import MDRNN, gmm_loss
from train_utils.helper_functions import save_checkpoint
import matplotlib.pyplot as plt
import json


parser = argparse.ArgumentParser(description='MD-RNN')
parser.add_argument('--params', default="params/mdrnn_train_params.yaml", metavar='params',
                    help="Path to file containing parameters for training")
args = parser.parse_args()

with open(args.params, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        print(params)
    except yaml.YAMLError as exc:
        print(exc)

cuda = params["cuda"] and torch.cuda.is_available()
torch.manual_seed(params["seed"])
device = torch.device("cuda" if cuda else "cpu")

torch.cuda.empty_cache()
#transform=transforms.ToTensor()


transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset(params["path_data"], params["seq_len"], transform, buffer_size=params["train_buffer_size"]),
    batch_size=params['batch_size'], num_workers=1, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset(params["path_data"],  params["seq_len"], transform, train=False, buffer_size=params["test_buffer_size"]),
    batch_size=params['batch_size'], num_workers=1)



vae_file = os.path.join(params['logdir'], 'vae', 'best.tar')
assert os.path.exists(vae_file), "VAE Checkpoint does not exist."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae_model = VAE(nc=3, ngf=params["img_size"], ndf=params["img_size"], latent_variable_size=params["latent_size"], cuda=cuda).to(device)
vae_model.load_state_dict(state['state_dict'])

rnn_dir = os.path.join(params['logdir'], 'mdrnn')
rnn_file = os.path.join(rnn_dir, 'best.tar')
if os.path.exists(rnn_file):
    state_rnn = torch.load(rnn_file)
    print("Loading MD-RNN at epoch {} "
          "with test error {}".format(
              state_rnn['epoch'], state_rnn['precision']))

    mdrnn = MDRNN(params['latent_size'], params['action_size'], params['hidden_size'], params['num_gmm']).to(device)
    rnn_state_dict = {k: v for k, v in state_rnn['state_dict'].items()}
    mdrnn.load_state_dict(rnn_state_dict)
else:
    mdrnn = MDRNN(params['latent_size'], params['action_size'], params['hidden_size'], params['num_gmm'])
mdrnn.to(device)
if not os.path.exists(rnn_dir):
    os.mkdir(rnn_dir)

optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)


def plot_curve(train, test, epochs):
    plt.plot(epochs, train, label="train loss")
    plt.plot(epochs, test, label="test loss")
    plt.legend()
    plt.grid()
    plt.savefig(params["logdir"] + "/mdrnn_training_curve.png")
    plt.close()


def to_latent(obs, next_obs):
    """ Transform observations to latent space.
    :args obs: 5D torch tensor (batch_size, sequence_len, action_size, img_size, img_size)
    :args next_obs: 5D torch tensor (batch_size, sequence_len, action_size, img_size, img_size)
    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (batch_size, sequence_len, latent_size)
        - next_latent_obs: 4D torch tensor (batch_size, sequence_len, latent_size)
    """
    with torch.no_grad():

        obs, next_obs = [
            f.upsample(x.view(-1, 3, params['img_size'], params['img_size']), size=params['res_size'],
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae_model(x)[1:] for x in (obs, next_obs)]
        dim = int(obs_mu.shape[0]/params['seq_len'])

        latent_obs, latent_next_obs = [(x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(dim, params['seq_len'], params['latent_size']) for x_mu, x_logsigma in [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

    return latent_obs, latent_next_obs



def get_loss(latent_obs, action, latent_next_obs):
    """ Compute losses.
    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted)  / (LSIZE )
    The LSIZE factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).
    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :returns: dictionary of losses, containing the gmm, the mse and
        the averaged loss.
    """
    latent_obs, action,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       latent_next_obs]]
    mus, sigmas, logpi = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    scale = params['latent_size']
    loss = gmm / scale
    return dict(gmm=gmm, loss=loss)


def data_pass(epoch, train):
    if train:
        mdrnn.train()
        loader = train_loader
        mode="train"
    else:
        mdrnn.eval()
        loader = test_loader
        mode = "test"



    num_of_files = len(loader.dataset._files)
    buffer_size = loader.dataset._buffer_size
    iteration = 0
    final_loss = 0
    all_files = 0
    loader.dataset._buffer_index = 0
    break_cond = False
    while True:

        loader.dataset.load_next_buffer()
        cum_loss = 0
        cum_gmm = 0
        pbar = tqdm(total=len(loader.dataset), desc="Epoch {} - {}".format(epoch, mode))

        for i, data in enumerate(loader):
            obs, action, next_obs = [arr.to(device) for arr in data]
            latent_obs, latent_next_obs = to_latent(obs, next_obs)
            if train:
                losses = get_loss(latent_obs, action, latent_next_obs)
                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    losses = get_loss(latent_obs, action, latent_next_obs)

            cum_loss += losses['loss'].item()
            cum_gmm += losses['gmm'].item()

            pbar.set_postfix_str("loss={loss:10.6f} "
                                 "gmm={gmm:10.6f} ".format(
                loss=cum_loss / (i + 1), gmm=cum_gmm / params["latent_size"] / (i + 1)))
            pbar.update(params['batch_size'])
        pbar.close()
        final_loss += cum_loss
        all_files += len(loader.dataset)
        iteration +=1
        print("Iteration: " +str(iteration))
        print("Buffer index: "+str(loader.dataset._buffer_index))
        if buffer_size < num_of_files:

            if loader.dataset._buffer_index == 0 or break_cond:
                final_loss = final_loss * params['batch_size'] / all_files
                break
            if num_of_files - loader.dataset._buffer_index < buffer_size:
                break_cond = True
        else:
            final_loss = final_loss * params['batch_size'] / all_files
            break

    print("Average loss {}".format(final_loss))
    if train:
        mdn_plotter.plot('loss', 'train', 'MDRNN Train Loss', epoch, final_loss)
    else:
        mdn_plotter.plot('loss', 'test', 'MDRNN Test Loss', epoch, final_loss)
    return final_loss

train = partial(data_pass, train=True)
test = partial(data_pass, train=False)

cur_best = None
global mdn_plotter
mdn_plotter = VisdomLinePlotter(env_name=params['env'])
cum_train_loss = []
cum_test_loss = []
epochs_list = []
for e in range(params['epochs']):
    train_loss=train(e)
    test_loss = test(e)
    cum_test_loss.append(test_loss)
    cum_train_loss.append(train_loss)
    epochs_list.append(e)
    plot_curve(cum_train_loss, cum_test_loss, epochs_list)

    scheduler.step(test_loss)
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    if e % 10:
        name = "/mdrnn_train_report" + str(e)+".json"
        out_path = params["logdir"] + name
        params["epochs"] = epochs_list
        params["train_loss"] = cum_train_loss
        params["test_loss"] = cum_test_loss
        print("Saving status.")
        with open(out_path, 'w') as outfile:
            json.dump(params, outfile)
    checkpoint_fname = os.path.join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)



