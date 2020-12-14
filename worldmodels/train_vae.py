import argparse
import os
import yaml
import json
import torch
import torch.utils.data
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from dataload.action_dataset import RolloutObservationDataset

from train_utils.helper_functions import save_checkpoint, mse_loss, ms_ssim_loss, mix_loss, check_dir
from models.VAE import VAE
from eval_utils.visdom_plotter import VisdomLinePlotter, VisdomImagePlotter
import matplotlib.pyplot as plt



class VAE_TRAINER():

    def __init__(self, params):

        self.params = params
        self.loss_function = {
            'ms-ssim': ms_ssim_loss,
            'mse': mse_loss,
            'mix': mix_loss
        }[params["loss"]]

        # Choose device
        self.cuda = params["cuda"] and torch.cuda.is_available()
        torch.manual_seed(params["seed"])
        # Fix numeric divergence due to bug in Cudnn
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # Prepare data transformations
        red_size = params["img_size"]
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((red_size, red_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((red_size, red_size)),
            transforms.ToTensor(),
        ])

        # Initialize Data loaders
        op_dataset = RolloutObservationDataset(params["path_data"], transform_train, train=True)
        val_dataset = RolloutObservationDataset(params["path_data"], transform_val, train=False)

        self.train_loader = torch.utils.data.DataLoader(op_dataset, batch_size=params["batch_size"],
                                                   shuffle=True, num_workers=0)
        self.eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params["batch_size"],
                                                  shuffle=False, num_workers=0)

        # Initialize model and hyperparams
        self.model = VAE(nc=3, ngf=64, ndf=64, latent_variable_size=params["latent_size"], cuda=self.cuda).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.init_vae_model()
        self.visualize = params["visualize"]
        if self.visualize:
            self.plotter = VisdomLinePlotter(env_name=params['env'])
            self.img_plotter = VisdomImagePlotter(env_name=params['env'])
        self.alpha = params["alpha"] if params["alpha"] else 1.0

    def train(self, epoch):
        self.model.train()
        # dataset_train.load_next_buffer()
        mse_loss = 0
        ssim_loss = 0
        train_loss = 0
        # Train step
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss, mse, ssim = self.loss_function(recon_batch, data, mu, logvar, self.alpha)
            loss.backward()

            train_loss += loss.item()
            ssim_loss+=ssim
            mse_loss+=mse
            self.optimizer.step()

            if batch_idx % params["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item()))
                print('MSE: {} , SSIM: {:.4f}'.format(
                    mse, ssim))

        step = len(self.train_loader.dataset) / float(self.params["batch_size"])
        mean_train_loss = train_loss / step
        mean_ssim_loss = ssim_loss / step
        mean_mse_loss = mse_loss / step
        print('-- Epoch: {} Average loss: {:.4f}'.format(
            epoch, mean_train_loss))
        print('-- Average MSE: {:.5f} Average SSIM: {:.4f}'.format(
            mean_mse_loss, mean_ssim_loss))
        if self.visualize:
            self.plotter.plot('loss', 'train', 'VAE Train Loss', epoch, mean_train_loss)
        return

    def eval(self):
        self.model.eval()
        # dataset_test.load_next_buffer()
        eval_loss = 0
        mse_loss = 0
        ssim_loss = 0
        vis = True
        with torch.no_grad():
            # Eval step
            for data in self.eval_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)

                loss, mse, ssim = self.loss_function(recon_batch, data, mu, logvar, self.alpha)
                eval_loss += loss.item()
                ssim_loss += ssim
                mse_loss += mse
                if vis:
                    org_title = "Epoch: " + str(epoch)
                    comparison1 = torch.cat(
                        [data[:4], recon_batch.view(params["batch_size"], 3, params["img_size"], params["img_size"])[:4]])
                    if self.visualize:
                        self.img_plotter.plot(comparison1, org_title)
                    vis = False

        step = len(self.eval_loader.dataset) / float(params["batch_size"])
        mean_eval_loss = eval_loss / step
        mean_ssim_loss = ssim_loss / step
        mean_mse_loss = mse_loss / step
        print('-- Eval set loss: {:.4f}'.format(mean_eval_loss))
        print('-- Eval MSE: {:.5f} Eval SSIM: {:.4f}'.format(
            mean_mse_loss, mean_ssim_loss))
        if self.visualize:
            self.plotter.plot('loss', 'eval', 'VAE Eval Loss', epoch, mean_eval_loss)
            self.plotter.plot('loss', 'mse train', 'VAE MSE Loss', epoch, mean_mse_loss)
            self.plotter.plot('loss', 'ssim train', 'VAE MSE Loss', epoch, mean_ssim_loss)

        return mean_eval_loss

    def init_vae_model(self):
        self.vae_dir = os.path.join(self.params["logdir"], 'vae')
        check_dir(self.vae_dir, 'samples')
        if not self.params["noreload"]:# and os.path.exists(reload_file):
            reload_file = os.path.join(self.params["vae_location"], 'best.tar')
            state = torch.load(reload_file)
            print("Reloading model at epoch {}"
                  ", with eval error {}".format(
                state['epoch'],
                state['precision']))
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])

    def checkpoint(self, cur_best, eval_loss):
        # Save the best and last checkpoint
        best_filename = os.path.join(self.vae_dir, 'best.tar')
        filename = os.path.join(self.vae_dir, 'checkpoint.tar')
        is_best = not cur_best or eval_loss < cur_best
        if is_best:
            cur_best = eval_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'precision': eval_loss,
            'optimizer': self.optimizer.state_dict()
        }, is_best, filename, best_filename)
        return cur_best

    def plot(self, train, eval, epochs):
        plt.plot(epochs, train, label="train loss")
        plt.plot(epochs, eval, label="eval loss")
        plt.legend()
        plt.grid()
        plt.savefig(self.params["logdir"]+"/vae_training_curve.png")
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--params', default="params/vae_train_params.yaml", metavar='yaml',
                        help="Path to file containing parameters for training")
    args = parser.parse_args()

    with open(args.params, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            print(params)
        except yaml.YAMLError as exc:
            print(exc)

    # Create relevant directories
    check_dir(params["logdir"])
    out_path = os.path.join(params["logdir"], "train_params.json")
    with open(out_path, 'w') as outfile:
        json.dump(params, outfile)

    if params["sample"]:
        check_dir(params["logdir"], 'results')
    trainer = VAE_TRAINER(params)
    cur_best = None
    epochs = params["epochs"]
    cum_train_loss = []
    cum_eval_loss = []
    epochs_list = []

    # Start training
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train(epoch)
        eval_loss = trainer.eval()
        cum_eval_loss.append(eval_loss)
        cum_train_loss.append(train_loss)
        epochs_list.append(epoch)
        trainer.plot(cum_train_loss, cum_eval_loss, epochs_list)
        cur_best = trainer.checkpoint(cur_best, eval_loss)

        if params["sample"]:
            with torch.no_grad():
                sample = torch.randn(16, params["latent_size"]).to(trainer.device)
                sample = trainer.model.decode(sample).cpu()
                save_image(sample.view(16, 3, params["img_size"], params["img_size"]),
                           os.path.join(params["logdir"], 'results/sample_' + str(epoch) + '.png'))

    out_path = params["logdir"] + "/vae_train_report.json"
    params["epochs"] = epochs_list
    params["train_loss"] = cum_train_loss
    params["eval_loss"] = cum_eval_loss
    print("Training finished.")
    with open(out_path, 'w') as outfile:
        json.dump(params, outfile)