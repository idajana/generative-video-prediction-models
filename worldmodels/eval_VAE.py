from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
from pytorch_msssim import msssim
from torch import nn, optim
from math import log10
from eval_utils.eval_func import plot_table
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import yaml
from models.VAE import VAE
import numpy as np
import json
from train_utils.helper_functions import check_dir
from dataload.action_dataset import RolloutObservationDataset


def validate(vae_name, params, val_loader, vae_param):
    cuda = params["cuda"] and torch.cuda.is_available()
    torch.manual_seed(params["seed"])
    device = torch.device("cuda" if cuda else "cpu")
    model_path = os.path.join(params["vae_dir"], vae_name)
    vae_file = os.path.join(model_path, 'vae', 'best.tar')
    assert os.path.exists(vae_file), "VAE Checkpoint does not exist."
    state = torch.load(vae_file)
    print("Loading VAE at epoch {} "
          "with test error {}".format(
        state['epoch'], state['precision']))
    print(str(vae_name))
    model = VAE(nc=3, ngf=params["img_size"], ndf=params["img_size"], latent_variable_size=vae_param["latent_size"],
                cuda=cuda).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    avg_psnr = 0
    avg_ms_ssim = 0
    index=0
    with torch.no_grad():
        val_loader.dataset.load_next_buffer()

        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            mse = F.mse_loss(recon_batch, data)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            #print("PSNR:", str(psnr))
            ms_ssim = msssim(recon_batch, data).item()
            avg_ms_ssim += ms_ssim
            #print("MS-SSIM:", str(ms_ssim))

            if index < 10:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n], recon_batch.view(params["batch_size"], 3, params["img_size"], params["img_size"])[:n]])
                save_image(comparison.cpu(),
                         os.path.join(params["report_dir"], str(vae_name) +"_"+str(index)+ '.png'), nrow=n)
            index+=1

    step = len(val_loader.dataset) / params["batch_size"]
    avg_ms_ssim /= step
    avg_psnr /= step

    print("AVG PSNR", str(avg_psnr))
    print("AVG MS-SSIM", str(avg_ms_ssim))
    print("index", str(index))
    print("step", str(step))

    return [avg_psnr, avg_ms_ssim]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eval_VAE')
    parser.add_argument('--params', default="params/eval_vae.yaml", metavar='params',
                        help="Path to file containing parameters for training")
    args = parser.parse_args()
    with open(args.params, 'r') as stream:
        try:
            param = yaml.safe_load(stream)
            print(param)
        except yaml.YAMLError as exc:
            print(exc)
    if not os.path.exists(param["report_dir"]):
        os.mkdir(param["report_dir"])
    img_out = os.path.join(param["report_dir"], "vae_eval.png")

    if param["load_report"]:
        with open(param["load_report"], 'r') as stream:
            plot_data = json.load(stream)
            plot_table(np.array(plot_data["scores"]), img_out, None, None, ["PSNR", "MS-SSIM"], plot_data["names"])
    else:
        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((param["img_size"], param["img_size"])),
            transforms.ToTensor(),
        ])

        val_dataset = RolloutObservationDataset(param["path_data"], transform_val, train=False)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=param["batch_size"],
                                shuffle=False, num_workers=1)
        vae_files = os.listdir(param["vae_dir"])
        scores = []
        names = []
        for vae_file in vae_files:
            vae_path = os.path.join(param["vae_dir"], vae_file, "train_params.json")
            print(vae_path)
            #Ability to eval multiple trained VAEs
            if os.path.exists(vae_path):
                with open(os.path.join(param["vae_dir"], vae_file, "train_params.json"), 'r') as stream:
                    vae_param = json.load(stream)

                names.append(vae_file)
                score = validate(vae_file, param, val_loader, vae_param)
                scores.append(score)

        table_data = np.array(scores)
        plot_table(table_data, img_out, None, None, ["PSNR", "MS-SSIM"], names)
        report_dict = dict()
        report_dict["scores"] = scores
        report_dict["names"] = names
        #Save report for later
        out_path = param["report_dir"] + "/vae_report.json"
        with open(out_path, 'w') as outfile:
            json.dump(report_dict, outfile)