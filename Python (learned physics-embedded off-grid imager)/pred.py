# physics-embedded off-grid imager
# author: Yixuan Huang, public date: 20250829
# citation: Y. Huang, J. Yang, S. Xia, C.-K. Wen, and S. Jin, 
# “Learned Off-Grid Imager for Low-Altitude Economy with Cooperative ISAC Network,” 
# IEEE Transactions on Wireless Communications, 2025.


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import logging
from tqdm import tqdm
import scipy.io as io

from model import resCNN, BasicBlock
from utils import AverageMeter, seed_everything, ohem_loss1, ohem_loss2, dataloader, load_config, cal_detect_rate
from pytorch_msssim import ssim

os.environ["WANDB_SILENT"] = "true"


def train_model(
        model,
        device,
        config,
):
    dataset = dataloader(config)
    n_val = len(dataset)

    loader_args = dict(batch_size=config.batch_size, num_workers=2, pin_memory=False)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    criterion = nn.MSELoss(reduction='mean')
    if config.ohem_loss != 0: criterion.reduction = 'none' # if use ohem loss, do not reduction for MSELoss
    threshold_mse_criterion = nn.MSELoss(reduction='mean')

    estimate_all = []
    estimate_threshold_all = []
    true_all = []
    model.eval()
    epoch_loss = AverageMeter()
    epoch_mse = AverageMeter()
    epoch_threshold_mse = AverageMeter()
    epoch_ssim = AverageMeter()
    epoch_threshold_ssim = AverageMeter()
    epoch_detect_rate = AverageMeter()
    epoch_error_rate = AverageMeter()
    epoch_threshold_detect_rate = AverageMeter()
    epoch_threshold_error_rate = AverageMeter()
    real_mse = torch.tensor([0]).to(device=device)
    global_step = 0
    with tqdm(total=n_val, unit=' measurement', disable=config.tqdm_disable_flag) as pbar:
        for batch in val_loader:

            input, label = batch
            input = input.to(device=device)
            label = label.to(device=device, dtype=torch.float32).squeeze()

            preds = model(input)
            preds = preds.squeeze()

            if config.ohem_loss == 0: # use traditional mse loss
                loss = criterion(preds, label).mean()
                real_mse = loss
            elif config.ohem_loss == 1: # use ohem loss 1
                pre_loss = criterion(preds, label)
                loss = ohem_loss1(pre_loss, label, config.neg_scale, config.neg_min_case)
                real_mse = pre_loss.mean() # record real mse, rather than ohem loss
            elif config.ohem_loss == 2: # use ohem loss 2
                pre_loss = criterion(preds, label)
                loss = ohem_loss2(pre_loss, label, config.neg_scale, config.neg_min_case)
                real_mse = pre_loss.mean() # record real mse, rather than ohem loss

            loss = loss + config.sparse_loss_scale * preds.sum() / label.shape[0] / label.shape[1] / label.shape[2] # add sparsity loss

            detect_rate, error_rate = cal_detect_rate(preds.cpu().numpy(), label.cpu().numpy())
            epoch_detect_rate.update(detect_rate, input.shape[0])
            epoch_error_rate.update(error_rate, input.shape[0])

            # threshold and calculate various metrics
            img_max_value = preds.max(dim=1)[0].max(dim=1)[0]
            img_max_value = img_max_value.repeat(40, 40, 1).permute(2, 0, 1)
            threshold_flag = preds < img_max_value / config.img_threshold
            threshold_img = preds.detach().clone()
            threshold_img[threshold_flag] = 0.0
            threshold_mse = threshold_mse_criterion(threshold_img, label)
            epoch_threshold_mse.update(threshold_mse.data.item(), input.shape[0])

            detect_rate, error_rate = cal_detect_rate(threshold_img.cpu().numpy(), label.cpu().numpy())
            epoch_threshold_detect_rate.update(detect_rate, input.shape[0])
            epoch_threshold_error_rate.update(error_rate, input.shape[0])

            epoch_loss.update(loss.data.item(), input.shape[0])
            epoch_mse.update(real_mse.data.item(), input.shape[0])
            epoch_ssim.update(ssim(preds.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])
            epoch_threshold_ssim.update(ssim(threshold_img.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])

            pbar.update(input.shape[0])
            global_step += 1
            pbar.set_postfix(**{'loss_epoch_avg': epoch_loss.avg})

            for pred, thre_pred, true in zip(preds, threshold_img, label):
                estimate_all.append(pred)
                estimate_threshold_all.append(thre_pred)
                true_all.append(true)

    estimate_all = torch.stack(estimate_all).cpu().detach().numpy()
    estimate_threshold_all = torch.stack(estimate_threshold_all).cpu().detach().numpy()
    true_all = torch.stack(true_all).cpu().detach().numpy()

    # save all results in mat format
    save_str = './valid_data_mat/' + config.time_str_now + '.mat'
    io.savemat(save_str, {'estimate_all': estimate_all, 'estimate_threshold_all': estimate_threshold_all, 'true_all': true_all, 'epoch_loss': epoch_loss.avg,
                          'epoch_detect_rate': epoch_detect_rate.avg, 'epoch_error_rate': epoch_error_rate.avg,
                          'epoch_threshold_detect_rate': epoch_threshold_detect_rate.avg, 'epoch_threshold_error_rate': epoch_threshold_error_rate.avg,
                          'epoch_mse': epoch_mse.avg, 'epoch_threshold_mse': epoch_threshold_mse.avg,
                          'epoch_ssim': epoch_ssim.avg, 'epoch_threshold_ssim': epoch_threshold_ssim.avg})


def str2bool(v):
    return v.lower() in ('true', '1')

def get_args():
    parser = argparse.ArgumentParser(description='off-grid imager')
    parser.add_argument('--config_dir', type=str, default='./checkpoints/20241030_094027/config.txt', help='load saved config in training')
    parser.add_argument('--dataset_dir_test', type=str, default='./data/test_data_1w_uav6.mat')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_args()
    config = load_config(vars(config), config.config_dir) # use saved config values
    seed_everything(config.random_seed)
    config = argparse.Namespace(**config)
    config.dataset_dir = config.dataset_dir_test # real test dataset dir

    input_channel = 2
    output_channel = 1
    model = resCNN(BasicBlock, input_channel=input_channel, output_channel=output_channel,
                   inner_channels=config.CNN_inner_channel, kernel_size=config.CNN_kernel_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info({model})

    para_dict = torch.load(config.config_dir[:-10] + 'checkpoint_epoch200.pth', map_location=device)
    model_state_dict = para_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    logging.info(f'Model loaded from {config.config_dir[:-10]} checkpoint_epoch200.pth')
    model.to(device=device)

    for param in model.parameters():
        param.requires_grad = False

    train_model(
        model=model,
        device=device,
        config=config,
    )
