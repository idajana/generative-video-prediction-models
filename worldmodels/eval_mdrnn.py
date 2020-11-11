import torch
import argparse
import os
from torch.autograd import Variable
import progressbar
import numpy as np
import yaml
from torchvision import transforms
from dataload.action_dataset import RolloutSequenceDataset
from torch.utils.data import DataLoader
from train_utils import svp_utils
import json
from torch.distributions.categorical import Categorical
from models.VAE import VAE
from models.MD_RNN import MDRNNCell

#-------------load world model
parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--params', default="params/eval_wm.yaml", metavar='yaml',
                    help="Path to file containing parameters for evaluation")
args = parser.parse_args()

with open(args.params, 'r') as stream:
    try:
        params_yaml= yaml.safe_load(stream)
        print(params_yaml)
    except yaml.YAMLError as exc:
        print(exc)


n_eval = params_yaml["n_eval"]
n_past = params_yaml["n_past"]
n_future =params_yaml["seq_len"] - params_yaml["n_past"]
nsample = params_yaml["nsample"]

batch_size = params_yaml["batch_size"]
log_dir = params_yaml["eval_dir"]
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

class WorldModel:
    def __init__(self, params):

        cuda = params["cuda"] and torch.cuda.is_available()
        torch.manual_seed(params["seed"])
        device = torch.device("cuda" if cuda else "cpu")
        vae_file = os.path.join(params['logdir'], 'vae', 'best.tar')

        assert os.path.exists(vae_file), "VAE Checkpoint does not exist."
        state = torch.load(vae_file, map_location=torch.device('cpu'))
        print("Loading VAE at epoch {} "
              "with test error {}".format(
                  state['epoch'], state['precision']))

        self.vae_model = VAE(nc=3, ngf=params["img_size"], ndf=params["img_size"], latent_variable_size=params["latent_size"], cuda=cuda).to(device)
        self.vae_model.load_state_dict(state['state_dict'])
        rnn_dir = os.path.join(params['logdir'], 'mdrnn')
        rnn_file = os.path.join(rnn_dir, 'best.tar')
        assert os.path.exists(rnn_file), "MD-RNN Checkpoint does not exist."
        state_rnn = torch.load(rnn_file)
        print("Loading MD-RNN at epoch {} "
              "with test error {}".format(
                  state_rnn['epoch'], state_rnn['precision']))

        self.mdrnn = MDRNNCell(params['latent_size'], params['action_size'], params['hidden_size'], params['num_gmm']).to(device)
        rnn_state_dict = {k.strip('_l0'): v for k, v in state_rnn['state_dict'].items()}
        self.mdrnn.load_state_dict(rnn_state_dict)

        self.latent = torch.randn(1, params['latent_size'])
        self.hidden = 2 * [torch.zeros(1, params['hidden_size'])]

        self.monitor = None
        self.figure = None
        # obs
        self.start_obs = None
        self.start_obs_recon = None
        self._obs = None
        self._visual_obs = None

        self.red_size = params["red_size"]
        self.params = params

    def forw(self):
        ovo=self.vae_model.forward(self.start_obs)
        return ovo[0]
    def compute_z(self, obs):
        self.latent = self.vae_model.get_latent_var(obs)
        return self.latent

    def decode(self):
        with torch.no_grad():
            self.start_obs_recon = self.vae_model.decode(self.latent)
            np_obs = self.start_obs_recon.numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self.start_obs_recon_np = np_obs

    def reset(self):
        """ Resetting """
        self.latent = torch.randn(1, self.params['latent_size'])
        self.hidden = 2 * [torch.zeros(1, self.params['hidden_size'])]


    def step(self, action):
        """ One step forward """
        with torch.no_grad():

            #action = torch.Tensor(action).unsqueeze(0)
            mu, sigma, pi, n_h = self.mdrnn(action, self.latent, self.hidden)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()

            self.latent = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self.hidden = n_h

            self._obs = self.vae_model.decode(self.latent)
            return self._obs
            #np_obs = self._obs.numpy()
            #np_obs = np.clip(np_obs, 0, 1) * 255
            #np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            #np_obs = np_obs.squeeze()
            #np_obs = np_obs.astype(np.uint8)
            #self._visual_obs = np_obs

            #return np_obs

world = WorldModel(params_yaml)

def make_gifs(x, idx, name, action, plot):
    # get approx posterior sample

    gen_seq = []
    gt_seq = []

    ssim = np.zeros((batch_size, nsample, n_future))
    psnr = np.zeros((batch_size, nsample, n_future))
    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []

    for s in range(nsample):
        progress.update(s + 1)
        gen_seq = []
        gt_seq = []
        world.reset()
        x_in = x[:,0]
        all_gen.append([])
        all_gen[s].append(x_in)

        for i in range(1, n_eval):

            z_t = world.compute_z(x_in)
            if i < n_past:
                x_in = x[:,i]
                all_gen[s].append(x_in)

            else:
                world.decode()
                x_in = world.step(action[:, i-1])
                all_gen[s].append(x_in)
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[:,i].data.cpu().numpy())
                x_in = x[:,i]

        _, ssim[:, s, :], psnr[:, s, :] =svp_utils.eval_seq(gt_seq, gen_seq)

    progress.finish()
    svp_utils.clear_progressbar()

    ssim_per_t = []
    psnr_per_t = []
    ###### ssim ######
    for i in range(batch_size):
        gifs = [[] for t in range(n_eval)]
        text = [[] for t in range(n_eval)]
        mean_ssim = np.mean(ssim[i], 1)
        mean_psnr = np.mean(psnr[i], 1)

        ordered = np.argsort(mean_ssim)
        best_psnr = np.argsort(mean_psnr)
        ssim_per_t.append(ssim[i][ordered[-1]])
        psnr_per_t.append(psnr[i][best_psnr[-1]])

        if plot:
            rand_sidx = [np.random.randint(nsample) for s in range(3)]
            for t in range(n_eval):
                # gt
                gifs[t].append(add_border(x[:, t][i], 'green'))
                text[t].append('Ground\ntruth')

                # best
                if t < n_past:
                    color = 'green'
                else:
                    color = 'red'
                sidx = ordered[-1]
                gifs[t].append(add_border(all_gen[sidx][t][i], color))
                text[t].append('Best SSIM')
                # random 3
                for s in range(len(rand_sidx)):
                    gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                    text[t].append('Random\nsample %d' % (s + 1))

            fname = '%s/%s_%d.gif' % (log_dir, name, idx + i)
            svp_utils.save_gif_with_text(fname, gifs, text)

    return np.mean(np.array(ssim_per_t), 0), np.mean(np.array(psnr_per_t), 0)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w + 2 * pad + 30, w + 2 * pad))
    if color == 'red':
        px[2] = 0.7
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w + pad, pad:w + pad] = x
    else:
        px[:, pad:w + pad, pad:w + pad] = x
    return px




# --------- load a dataset ------------------------------------
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
test_loader = DataLoader(
    RolloutSequenceDataset(params_yaml["path_data"], params_yaml["seq_len"], transform, train=False,
                           buffer_size=params_yaml["test_buffer_size"]),
    batch_size=params_yaml['batch_size'], num_workers=4, shuffle=False, drop_last=True)



num_of_files = len(test_loader.dataset._files)
buffer_size = test_loader.dataset._buffer_size
counter = 0
mean_ssim = []
mean_psnr = []
plot = True
while True:
    test_loader.dataset.load_next_buffer()
    for i, data in enumerate(test_loader):
        obs, action, next_obs = [arr.to("cpu") for arr in data]
        obs = torch.cat([obs, next_obs[:, -1].unsqueeze(1)], 1)
        ssim, psnr = make_gifs(obs, i, '{}_eval'.format(counter), action, plot)
        mean_psnr.append(psnr)
        mean_ssim.append(ssim)

        print("Current mean psnr t: ", np.mean(np.array(mean_psnr), 0))
        print("Current mean ssim t: ", np.mean(np.array(mean_ssim), 0))

        counter+=batch_size
        if counter > 200:
            plot = False
    print(str(test_loader.dataset._buffer_index)+"/"+str(num_of_files))
    if num_of_files - test_loader.dataset._buffer_index < buffer_size/2:
        break

report = dict()
report["psnr_t"] = str(np.mean(np.array(mean_psnr), 0))
report["ssim_t"] = str(np.mean(np.array(mean_ssim), 0))
out_path = os.path.join(params_yaml["logdir"], "report.json")
print("Evaluation finished.")
with open(out_path, 'w') as outfile:
    json.dump(params_yaml, outfile)
