import torch
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from train_utils.helper_functions import check_dir
import yaml
from torchvision import transforms
from dataload.action_dataset import RolloutSequenceDataset
from torch.utils.data import DataLoader
from train_utils import svp_utils
import json



parser = argparse.ArgumentParser(description='SVP_LP_eval')
parser.add_argument('--params', default="params/svg_lp_eval.yaml", metavar='params',
                    help="Path to file containing parameters for training")
args = parser.parse_args()

with open(args.params, 'r') as stream:
    try:
        eval_params = yaml.safe_load(stream)
        print(eval_params)
    except yaml.YAMLError as exc:
        print(exc)
check_dir(eval_params["logdir"])

torch.manual_seed(eval_params["seed"])


# ---------------- load the models  ----------------

tmp = torch.load(eval_params["model_path"], map_location="cuda:0")
params = tmp["params"]
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
prior.eval()
frame_predictor.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.eval()
decoder.eval()
if eval_params["batch_size"]:
    params["batch_size"] = eval_params["batch_size"]
frame_predictor.batch_size = params["batch_size"]
posterior.batch_size = params["batch_size"]
prior.batch_size = params["batch_size"]

batch_size = params["batch_size"]
g_dim = params["g_dim"]
z_dim = params['z_dim']
n_past = params["n_past"]
n_eval = params["n_eval"]
nsample = eval_params["nsample"]
n_future = params["seq_len"]-n_past
log_dir = eval_params["logdir"]

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
last_frame_skip = params['last_frame_skip']
channels = params['n_channels']
image_width = params['img_size']

# --------- load a dataset ------------------------------------
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
test_loader = DataLoader(
    RolloutSequenceDataset(eval_params["path_data"], params["seq_len"], transform, train=False,
                           buffer_size=eval_params["test_buffer_size"]),
    batch_size=params['batch_size'], num_workers=4, shuffle=True, drop_last=True)


# --------- eval funtions ------------------------------------

def make_gifs(x, idx, name, action, plot):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[:,0])
    x_in = x[:,0]
    for i in range(1, n_eval):
        h = encoder(x_in)
        h_target = encoder(x[:,i])[0].detach()
        if last_frame_skip or i < n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _ = posterior(h_target)  # take the mean
        if i < n_past:
            frame_predictor(torch.cat([h, z_t, action[:,i-1]], 1))
            posterior_gen.append(x[:,i])
            x_in = x[:,i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t, action[:,i-1]], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    ssim = np.zeros((batch_size, nsample, n_future))
    psnr = np.zeros((batch_size, nsample, n_future))
    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []

    for s in range(nsample):
        progress.update(s + 1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[:,0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, n_eval):
            h = encoder(x_in)
            if last_frame_skip or i < n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < n_past:
                h_target = encoder(x[:,i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t, action[:,i-1]], 1))
                x_in = x[:,i]
                all_gen[s].append(x_in)

            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t, action[:,i-1]], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[:,i].data.cpu().numpy())
                all_gen[s].append(x_in)

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
                gifs[t].append(add_border(x[:,t][i], 'green'))
                text[t].append('Ground\ntruth')
                # posterior
                #if t < n_past:
                #    color = 'green'
                #else:
                #    color = 'red'
                #gifs[t].append(add_border(posterior_gen[t][i], color))
                #text[t].append('Approx.\nposterior')
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


num_of_files = len(test_loader.dataset._files)
buffer_size = test_loader.dataset._buffer_size
counter = 0
mean_ssim = []
mean_psnr = []
plot = True
while True:
    test_loader.dataset.load_next_buffer()
    #print(batch_size)
    for i, data in enumerate(test_loader):
        obs, action, next_obs = [arr.to("cuda") for arr in data]
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
out_path = os.path.join(eval_params["logdir"], "report.json")
print("Evaluation finished.")
with open(out_path, 'w') as outfile:
    json.dump(params, outfile)
