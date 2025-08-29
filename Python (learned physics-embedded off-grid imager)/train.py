# physics-embedded off-grid imager
# author: Yixuan Huang, public date: 20250829
# citation: Y. Huang, J. Yang, S. Xia, C.-K. Wen, and S. Jin, 
# “Learned Off-Grid Imager for Low-Altitude Economy with Cooperative ISAC Network,” 
# IEEE Transactions on Wireless Communications, 2025.

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split

import wandb
import datetime
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

from model import resCNN, BasicBlock
from utils import AverageMeter, ohem_loss1, ohem_loss2, save_config, dataloader, seed_everything
from pytorch_msssim import ssim

os.environ["WANDB_SILENT"] = "true"


def train_model(
        model,
        optimizer,
        device,
        config,
):
    # 1. Create dataset
    dataset = dataloader(config)
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * config.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = Subset(train_set, range(int(len(train_set) * config.train_data_scale)))

    # 3. Create data loaders
    loader_args = dict(batch_size=config.batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # 4. scheduler is used for adaptively changing learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config.lr_patience, min_lr=2e-10)
    criterion = nn.MSELoss(reduction='mean')
    if config.ohem_loss != 0: criterion.reduction = 'none' # if use ohem loss, do not reduction for MSELoss
    threshold_mse_criterion = nn.MSELoss(reduction='mean')

    # 5. Begin training
    if config.wandbflag:
        experiment = wandb.init(project=config.wandb_project, resume='allow', anonymous='must', config=config)
    global_step = 0

    for epoch in range(1, config.epochs + 1):

        model.train()
        epoch_loss = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_threshold_mse = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_threshold_ssim = AverageMeter()
        real_mse = torch.tensor([0]).to(device=device)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{config.epochs}', unit=' measurement', disable=config.tqdm_disable_flag) as pbar:
            for batch in train_loader:

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

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                # threshold and calculate various metrics
                img_max_value = preds.max(dim=1)[0].max(dim=1)[0] # max pixel values for each instance
                img_max_value = img_max_value.repeat(40, 40, 1).permute(2, 0, 1)
                threshold_flag = preds < img_max_value / config.img_threshold # threshold for each instance
                threshold_img = preds.detach().clone()
                threshold_img[threshold_flag] = 0.0
                threshold_mse = threshold_mse_criterion(threshold_img, label)
                epoch_threshold_mse.update(threshold_mse.data.item(), input.shape[0])
                epoch_loss.update(loss.data.item(), input.shape[0])
                epoch_mse.update(real_mse.data.item(), input.shape[0])
                epoch_ssim.update(ssim(preds.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])
                epoch_threshold_ssim.update(ssim(threshold_img.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])

                pbar.update(input.shape[0])
                global_step += 1
                pbar.set_postfix(**{'loss_epoch_avg': epoch_loss.avg})
                if config.wandbflag: experiment.log({'train loss': epoch_loss.avg, 'train mse': epoch_mse.avg, 'train threshold mse': epoch_threshold_mse.avg,
                                                     'train ssim': epoch_ssim.avg, 'train threshold ssim': epoch_threshold_ssim.avg, 'step': global_step, 'epoch': epoch})

                division_step = (n_train // (2 * config.batch_size))
                if global_step % division_step == 0:
                    # valid after each epoch
                    val_loss, val_mse, val_threshold_mse, val_ssim, val_threshold_ssim = evaluate(model, val_loader, device, criterion, config, threshold_mse_criterion)
                    scheduler.step(val_loss)
                    logging.info('Validation loss: {}'.format(val_loss))

                    # upload images to wandb
                    if config.wandbflag:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any(): histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any(): histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        experiment.log({'valid loss': val_loss, 'valid mse': val_mse, 'valid threshold mse': val_threshold_mse, 
                                        'valid ssim': val_ssim, 'valid threshold ssim': val_threshold_ssim, 'step': global_step, 
                                        'learning rate': optimizer.param_groups[0]['lr'], 'epoch': epoch, **histograms})

                        input_image = torch.complex(input[0][0].squeeze(), input[0][1].squeeze())
                        input_image = input_image.abs()
                        input_image = input_image / input_image.max().float().cpu()
                        experiment.log({'input': wandb.Image(input_image)})
                        
                        true_image = label[0].float().cpu()
                        pred_image = preds[0].argmax(dim=0).float().cpu() if model.output_channel == 2 else preds[0].float().cpu()
                        pred_threshold_image = threshold_img[0].argmax(dim=0).float().cpu() if model.output_channel == 2 else threshold_img[0].float().cpu()
                        experiment.log({'output': {'true': wandb.Image(true_image), 'pred': wandb.Image(pred_image), 'pred_threshold': wandb.Image(pred_threshold_image)}})

            # save checkpoint
            if config.save_checkpoint:
                para_dict = {}
                para_dict['model_state_dict'] = model.state_dict()
                para_dict['optimizer_state_dict'] = optimizer.state_dict()
                torch.save(para_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def evaluate(net, dataloader, device, criterion, config, threshold_mse_criterion):
    net.eval()
    epoch_loss = AverageMeter()
    epoch_real_mse = AverageMeter()
    epoch_threshold_mse = AverageMeter()
    epoch_ssim = AverageMeter()
    epoch_threshold_ssim = AverageMeter()
    real_mse = torch.tensor([0]).to(device=device)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False, disable=config.tqdm_disable_flag):
        input, label = batch
        input = input.to(device=device)
        label = label.to(device=device, dtype=torch.float32).squeeze()

        preds = net(input)
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

        epoch_loss.update(loss.data.item(), input.shape[0])
        epoch_real_mse.update(real_mse.data.item(), input.shape[0])

        # threshold and calculate various metrics
        img_max_value = preds.max(dim=1)[0].max(dim=1)[0]
        img_max_value = img_max_value.repeat(40, 40, 1).permute(2, 0, 1)
        threshold_flag = preds < img_max_value / config.img_threshold
        threshold_img = preds
        threshold_img[threshold_flag] = 0.0
        threshold_mse = threshold_mse_criterion(threshold_img, label)
        epoch_threshold_mse.update(threshold_mse.data.item(), input.shape[0])

        epoch_ssim.update(ssim(preds.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])
        epoch_threshold_ssim.update(ssim(threshold_img.unsqueeze(dim=1), label.unsqueeze(dim=1), data_range=1).data.item(), input.shape[0])

    net.train()
    return epoch_loss.avg, epoch_real_mse.avg, epoch_threshold_mse.avg, epoch_ssim.avg, epoch_threshold_ssim.avg


def str2bool(v):
    return v.lower() in ('true', '1')

def get_args():
    parser = argparse.ArgumentParser(description='off-grid imager')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_patience', type=int, default=5, help='decay the learning rate when the loss is not degraded for * validation')
    parser.add_argument('--random_seed', type=int, default=2, help='')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    
    parser.add_argument('--dataset_dir', type=str, default='./data/train_data_1w_uav6.mat')
    parser.add_argument('--train_data_scale', type=float, default=1.0, help='')
    parser.add_argument('--val_percent', type=float, default=0.1)
    parser.add_argument('--wandbflag', type=str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='UAV_test_20240827')
    parser.add_argument('--tqdm_disable_flag', type=str2bool, default=False)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)

    parser.add_argument('--CNN_inner_channel', type=str, default='32,64,32', help='')
    parser.add_argument('--CNN_kernel_size', type=int, default=5, help='3 / 5')
    parser.add_argument('--ohem_loss', type=int, default=1, help='0 for false, 1 for ohem-1, 2 for ohem-2')
    parser.add_argument('--neg_scale', type=float, default=10.0, help='negative case number for loss')
    parser.add_argument('--neg_min_case', type=int, default=1, help='minimum negative case number')
    parser.add_argument('--sparse_loss_scale', type=float, default=1.0, help='regularization for sparsity')
    parser.add_argument('--img_threshold', type=float, default=3.0, help='pixel lower than max/img_threshold should be set to 0')
    
    return parser.parse_args()


if __name__ == '__main__':
    config = get_args()
    seed_everything(config.random_seed)

    config.time_str_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_checkpoint = Path('./checkpoints/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '/')
    config.running_code = os.path.abspath(__file__)
    config.CNN_inner_channel = list(map(lambda x: int(x), config.CNN_inner_channel.split(',')))

    input_channel = 2
    output_channel = 1
    model = resCNN(BasicBlock, input_channel=input_channel, output_channel=output_channel,
                   inner_channels=config.CNN_inner_channel, kernel_size=config.CNN_kernel_size)
    
    if config.wandbflag:
        config.save_checkpoint = True
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=str(dir_checkpoint)+'/logging.txt')
        save_config(config, str(dir_checkpoint)+'/config.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info({model})

    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=1e-8, momentum=0.999, foreach=True)

    if config.load:
        para_dict = torch.load(config.load, map_location=device)
        model_state_dict = para_dict['model_state_dict']
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(para_dict['optimizer_state_dict'])
        logging.info(f'Model loaded from {config.load}')

    model.to(device=device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    train_model(
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
    )
