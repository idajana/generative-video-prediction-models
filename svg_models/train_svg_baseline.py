import argparse
import os
from os.path import join
import yaml
import json
import torch
import torch.utils.data
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from train_utils.helper_functions import save_checkpoint, mse_loss, ms_ssim_loss, mix_loss, check_dir, kl_criterion
from eval_utils.visdom_plotter import VisdomLinePlotter, VisdomImagePlotter
import matplotlib.pyplot as plt
import models.lstm as lstm_models
from train_utils import svp_utils
import models.vgg_64 as model
#import svg_models.dcgan_64 as model
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataload.action_dataset import RolloutSequenceDataset
from torch.utils.data import DataLoader
from functools import partial



class SVG_LP_TRAINER():

    def __init__(self, params):

        self.params = params
        self.loss_function = nn.MSELoss().cuda()
        # choose device
        self.cuda = params["cuda"] and torch.cuda.is_available()
        torch.manual_seed(params["seed"])
        # Fix numeric divergence due to bug in Cudnn
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # Load or init models
        if params["noreload"]:
            self.frame_predictor = lstm_models.lstm(params["g_dim"] +params["action_size"], params["g_dim"], params["rnn_size"], params["predictor_rnn_layers"],
                                           params["batch_size"]).cuda()
            self.encoder = model.encoder(params["g_dim"], params["n_channels"]).cuda()
            self.decoder = model.decoder(params["g_dim"], params["n_channels"]).cuda()
        else:
            self.load_checkpoint()
        self.frame_predictor.apply(svp_utils.init_weights)
        self.encoder.apply(svp_utils.init_weights)
        self.decoder.apply(svp_utils.init_weights)

        # Init optimizers
        self.frame_predictor_optimizer = optim.Adam(self.frame_predictor.parameters(), lr=params["learning_rate"], betas=(params["beta1"], 0.999))
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=params["learning_rate"], betas=(params["beta1"], 0.999))
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=params["learning_rate"], betas=(params["beta1"], 0.999))
        if params["plot_visdom"]:
            self.plotter = VisdomLinePlotter(env_name=params['env'])
            self.img_plotter = VisdomImagePlotter(env_name=params['env'])


        # Select transformations
        transform = transforms.Lambda(
            lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
        self.train_loader = DataLoader(
            RolloutSequenceDataset(params["path_data"], params["seq_len"], transform, buffer_size=params["train_buffer_size"]),
            batch_size=params['batch_size'], num_workers=4, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(
            RolloutSequenceDataset(params["path_data"],  params["seq_len"], transform, train=False, buffer_size=params["test_buffer_size"]),
            batch_size=params['batch_size'], num_workers=4, shuffle=False, drop_last=True)


    def load_checkpoint(self):
        tmp = torch.load(self.params["model_path"])
        print("LOADING CHECKPOINT.............")
        self.frame_predictor = tmp['frame_predictor']
        self.encoder = tmp['encoder']
        self.decoder = tmp['decoder']
        self.frame_predictor.batch_size = self.params["batch_size"]



    def plot_samples(self, x, actions, epoch):
        nsample = 5
        gen_seq = [[] for i in range(nsample)]

        gt_seq = [x[:,i] for i in range(x.shape[1])]

        #h_seq = [self.encoder(x[:,i]) for i in range(params["n_past"])]
        for s in range(nsample):
            self.frame_predictor.hidden = self.frame_predictor.init_hidden()

            gen_seq[s].append(x[:,0])
            x_in = x[:,0]
            for i in range(1, self.params["n_eval"]):
                h = self.encoder(x_in)
                if self.params["last_frame_skip"] or i < self.params["n_past"]:
                    h, skip = h
                    h = h.detach()
                else:
                    h, _ = h
                    h = h.detach()
                if i < self.params["n_past"]:
                    h_target = self.encoder(x[:, i])[0].detach()
                    self.frame_predictor(torch.cat([h, actions[:,i-1]], 1))
                    x_in = x[:,i]

                    gen_seq[s].append(x_in)
                else:
                    h = self.frame_predictor(torch.cat([h, actions[:,i-1]], 1)).detach()
                    x_in = self.decoder([h, skip]).detach()
                    gen_seq[s].append(x_in)

        to_plot = []
        gifs = [[] for t in range(self.params["n_eval"])]
        nrow = min(self.params["batch_size"], 10)
        for i in range(nrow):
            # ground truth sequence
            row = []
            for t in range(self.params["n_eval"]):
                row.append(gt_seq[t][i])
            to_plot.append(row)

            for s in range(nsample):
                row = []
                for t in range(self.params["n_eval"]):
                    row.append(gen_seq[s][t][i])
                to_plot.append(row)
            for t in range(self.params["n_eval"]):
                row = []
                row.append(gt_seq[t][i])
                for s in range(nsample):
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)

        fname = '%s/gen/sample_%d.png' % (self.params["logdir"], epoch)
        svp_utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (self.params["logdir"], epoch)
        svp_utils.save_gif(fname, gifs)


    def plot_rec(self, x, actions, epoch):
        # Generate a frame sequence visualization
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        gen_seq = []
        gen_seq.append(x[:,0])
        x_in = x[:,0]
        h_seq = [self.encoder(x[:,i]) for i in range(params["seq_len"])]
        for i in range(1, self.params["seq_len"]):
            h_target = h_seq[i][0].detach()
            if self.params["last_frame_skip"] or i < self.params["n_past"]:
                h, skip = h_seq[i - 1]
            else:
                h, _ = h_seq[i - 1]
            h = h.detach()
            if i < self.params["n_past"]:
                self.frame_predictor(torch.cat([h, actions[:,i-1]], 1))
                gen_seq.append(x[:,i])
            else:
                h = self.frame_predictor(torch.cat([h, actions[:,i-1]], 1)).detach()
                x_pred = self.decoder([h, skip]).detach()
                gen_seq.append(x_pred)

        to_plot = []
        nrow = min(self.params["batch_size"], 10)
        for i in range(nrow):
            row = []
            for t in range(self.params["seq_len"]):
                row.append(gen_seq[t][i])
            to_plot.append(row)
        check_dir(params["logdir"], "gen")
        fname = '%s/gen/rec_%d.png' % (self.params["logdir"], epoch)
        svp_utils.save_tensors_image(fname, to_plot)

    def data_pass(self, epoch, train):
        if train:
            self.frame_predictor.train()
            self.encoder.train()
            self.decoder.train()
            loader = self.train_loader
            mode = "train"
        else:
            self.frame_predictor.eval()
            self.encoder.eval()
            self.decoder.eval()
            loader = self.test_loader
            mode = "test"

        num_of_files = len(loader.dataset._files)
        buffer_size = loader.dataset._buffer_size
        iteration = 0
        final_loss = 0
        all_files = 0
        loader.dataset._buffer_index = 0
        break_cond = False
        plot_epoch = True
        while True:
            loader.dataset.load_next_buffer()
            cum_loss = 0
            cum_mse = 0.0
            cum_dl = 0.0
            pbar = tqdm(total=len(loader.dataset), desc="Epoch {} - {}".format(epoch, mode))

            for i, data in enumerate(loader):
                obs, action, next_obs = [arr.to(self.device) for arr in data]
                obs = torch.cat([obs, next_obs[:,-1].unsqueeze(1)], 1)

                # initialize the hidden state.
                self.frame_predictor.hidden = self.frame_predictor.init_hidden()
                seq_len = obs.shape[1]


                if not train:
                    if plot_epoch:
                        print(">>>>>>>>>>>PLOT<<<<<<<<<<<<<")
                        self.plot_rec(obs, action, epoch)
                        self.plot_samples(obs, action, epoch)
                        plot_epoch = False
                    h_seq = [self.encoder(obs[:, j]) for j in range(params["n_past"])]
                    mse = 0
                    kld = 0
                    x_in = obs[:, 0]

                    for t in range(1, seq_len):
                        h = self.encoder(x_in)
                        if t < params["n_past"] and params["last_frame_skip"]:
                            h, skip = h
                        else:
                            h, _ = h

                        if t < self.params["n_past"]:
                            h_target = self.encoder(obs[:, t])[0]


                        if t < self.params["n_past"]:
                            self.frame_predictor(torch.cat([h, action[:,t-1]], 1))
                            x_in = obs[:, t]
                        else:
                            h = self.frame_predictor(torch.cat([h, action[:,t-1]], 1))
                            x_in = self.decoder([h, skip])
                        x_pred = x_in
                        with torch.no_grad():
                            mse += self.loss_function(x_pred, obs[:, t])
                            kld = 0
                    mse /= params["seq_len"]
                    kld /= params["seq_len"]
                    loss = mse + kld*params["beta"]
                else:
                    self.frame_predictor.zero_grad()
                    self.encoder.zero_grad()
                    self.decoder.zero_grad()

                    h_seq = [self.encoder(obs[:,j]) for j in range(seq_len)]
                    mse = 0
                    kld = 0

                    for t in range(1, seq_len):
                        h_target = h_seq[t][0]
                        if t < params["n_past"] or params["last_frame_skip"]:
                            h, skip = h_seq[t-1]
                        else:
                            h = h_seq[t-1][0]
                        h_pred = self.frame_predictor(torch.cat([h, action[:,t-1]], 1))
                        x_pred = self.decoder([h_pred, skip])

                        mse += self.loss_function(x_pred, obs[:, t])

                    mse /= params["seq_len"]
                    loss = mse + kld*params["beta"]
                    loss.backward()
                    self.frame_predictor_optimizer.step()
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                cum_loss += loss
                cum_mse += mse
                cum_dl += 0
                pbar.set_postfix_str("loss={loss:5.4f} , MSE={mse_loss:5.4f}, DL_loss={dl_loss:5.4f} ".format(
                    loss=cum_loss / (i + 1), mse_loss=cum_mse/ (i + 1), dl_loss=cum_dl/ (i + 1)))
                pbar.update(params['batch_size'])
            pbar.close()
            final_loss += cum_loss
            all_files += len(loader.dataset)
            iteration += 1
            print("Iteration: " + str(iteration))
            print("Buffer index: " + str(loader.dataset._buffer_index))
            if buffer_size < num_of_files:
                if params["shorten_epoch"]==iteration:
                    break_cond = True
                if loader.dataset._buffer_index == 0 or break_cond:
                    final_loss = final_loss * params['batch_size'] / all_files
                    break
                if num_of_files - loader.dataset._buffer_index < buffer_size:
                    break_cond = True
            else:
                final_loss = final_loss * params['batch_size'] / all_files
                break

        print("Average loss {}".format(final_loss))
        if self.params["plot_visdom"]:
            if train:
                self.plotter.plot('loss', 'train', 'SVG_FP Train Loss', epoch, final_loss.item())
            else:
                self.plotter.plot('loss', 'test', 'SVG_FP Test Loss', epoch, final_loss.item())
        return final_loss.item()

    def init_svg_model(self):
        self.svg_dir = os.path.join(self.params["logdir"], 'svg')
        check_dir(self.svg_dir, 'samples')


    def checkpoint(self, cur_best, test_loss):
        best_filename = os.path.join(self.svg_dir, 'best.tar')
        filename = os.path.join(self.svg_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        save_checkpoint({
        'encoder': self.encoder,
        'decoder': self.decoder,
        'frame_predictor': self.frame_predictor,
        'test_loss': test_loss,
        'params': self.params
        }, is_best, filename, best_filename)
        return cur_best


    def plot(self, train, test, epochs):
        plt.plot(epochs, train, label="train loss")
        plt.plot(epochs, test, label="test loss")
        plt.legend()
        plt.grid()
        plt.savefig(self.params["logdir"] + "/svg_fp_training_curve.png")
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SVP_BASELINE')
    parser.add_argument('--params', default="params/svg_fp_train_params.yaml", metavar='yaml',
                        help="Path to file containing parameters for training")
    args = parser.parse_args(

    # Open config file
    with open(args.params, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print(params)
        except yaml.YAMLError as exc:
            print(exc
    # Check if directories exists, if not, create them
    check_dir(params["logdir"])
    out_path = params["logdir"] + "/train_params.json"
    with open(out_path, 'w') as outfile:
        json.dump(params, outfile)

    if params["sample"]:
        check_dir(params["logdir"], 'results')
    #CHANGE
    #device = torch.device('cuda:1')
    #torch.cuda.set_device(device)

    trainer = SVG_LP_TRAINER(params)
    trainer.init_svg_model()
    train = partial(trainer.data_pass, train=True)
    test = partial(trainer.data_pass, train=False)
    cur_best = None
    epochs = params["epochs"]
    cum_train_loss = []
    cum_test_loss = []
    epochs_list = []

    # Start training
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        cum_test_loss.append(test_loss)
        cum_train_loss.append(train_loss)
        epochs_list.append(epoch)
        trainer.plot(cum_train_loss, cum_test_loss, epochs_list)
        cur_best = trainer.checkpoint(cur_best, test_loss)

    out_path = params["logdir"] + "/svg_train_report.json"
    params["epochs"] = epochs_list
    params["train_loss"] = cum_train_loss
    params["test_loss"] = cum_test_loss
    print("Training finished.")
    with open(out_path, 'w') as outfile:
        json.dump(params, outfile)