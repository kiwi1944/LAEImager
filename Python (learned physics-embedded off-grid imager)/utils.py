import torch
import json
import hdf5storage
from collections import deque
import numpy as np
import random


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader(config):
    data = hdf5storage.loadmat(config.dataset_dir)
    input = data['data_herm_image']
    output = data['data_true_image']

    data_num = len(input)
    dataset_in = [0] * data_num
    for i in range(data_num):
        tmp = torch.tensor(input[i], dtype=torch.complex64).unsqueeze(dim=0)
        tmp = torch.cat((tmp.real, tmp.imag), 0)
        dataset_in[i] = tmp

    dataset_out = [0] * data_num
    for i in range(data_num):
        dataset_out[i] = torch.tensor(output[i], dtype=torch.long)

    dataset = [0] * data_num
    for i in range(data_num):
        dataset[i] = (dataset_in[i], dataset_out[i])
    
    return dataset


def ohem_loss1(pre_loss, loss_label, neg_rto, n_min_neg):

    positive_loss, negative_loss = 0, 0
    instance_num = 0
    for single_loss, single_label in zip(pre_loss, loss_label):

        # positive_loss
        pos_pixel = (single_label >= 0.1).float()
        n_pos_pixel = torch.sum(pos_pixel)
        pos_loss_region = single_loss * pos_pixel
        positive_loss += torch.sum(pos_loss_region)
        instance_num += max(n_pos_pixel, 1e-12)

        # negative_loss
        neg_pixel = (single_label < 0.1).float()
        n_neg_pixel = torch.sum(neg_pixel)
        neg_loss_region = single_loss * neg_pixel

        if n_pos_pixel != 0:
            if n_neg_pixel < torch.round(neg_rto * n_pos_pixel):
                negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), int(n_neg_pixel), sorted=False)[0])
                instance_num += n_neg_pixel
            else:
                n_hard_neg = max(n_min_neg, torch.round(neg_rto * n_pos_pixel))
                negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), int(n_hard_neg))[0])
                instance_num += n_hard_neg
        else:
            raise RuntimeError('why no positive pixel')

    total_loss = (positive_loss + negative_loss) / instance_num
    return total_loss


def ohem_loss2(pre_loss, loss_label, neg_rto, n_min_neg):

    batch_size = pre_loss.shape[0]

    positive_loss, negative_loss = 0, 0
    for single_loss, single_label in zip(pre_loss, loss_label):

        # positive_loss
        pos_pixel = (single_label >= 0.1).float()
        n_pos_pixel = torch.sum(pos_pixel)
        pos_loss_region = single_loss * pos_pixel
        positive_loss += torch.sum(pos_loss_region) / max(n_pos_pixel, 1e-12)

        # negative_loss
        neg_pixel = (single_label < 0.1).float()
        n_neg_pixel = torch.sum(neg_pixel)
        neg_loss_region = single_loss * neg_pixel

        if n_pos_pixel != 0:
            if n_neg_pixel < torch.round(neg_rto * n_pos_pixel):
                negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), int(n_neg_pixel), sorted=False)[0]) / n_neg_pixel
            else:
                n_hard_neg = max(n_min_neg, torch.round(neg_rto * n_pos_pixel))
                negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), int(n_hard_neg))[0]) / n_hard_neg
        else:
            raise RuntimeError('why no positive pixel')

    total_loss = (positive_loss + negative_loss) / batch_size
    return total_loss



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_config(config, filename):
    if config.wandbflag:
        with open(filename, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def load_config(args_dict, config_path):
    import json
    summary_filename = config_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
 
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
 
    return args_dict


def cal_detect_rate(preds, labels):

    label_target_count = 0.0
    detect_target_count = 0.0
    error_target_count = 0.0
    for pred, label in zip(preds, labels):
        pred_detect_flag = np.zeros([pred.shape[0], pred.shape[1]])
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i, j] >= 0.1:
                    label_target_count += 1 # add target num in label
                    flag = False # flag for if this target is detected
                    for ii in range(max(0, i - 1), min(label.shape[0], i + 2)):
                        for jj in range(max(0, j - 1), min(label.shape[1], j + 2)):
                            # if the target is detected in pred at the 3*3 region centered at its true position
                            if pred[ii, jj] > 0.1 and pred_detect_flag[ii, jj] == 0:
                                # it is detected only when pixel is nonzero and previously not used
                                # note that if two point targets in label are detected as one large target in pred, 
                                # only one target is believed to be detected in final results
                                flag = True
                                detect_target_count += 1
                                # renew flag matrix through DFS/BFS
                                pred_detect_flag = renew_flag(pred_detect_flag, pred, ii, jj)
                                break
                        if flag: break
        
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] > 0.1 and pred_detect_flag[i, j] == 0:
                    # re-examine pred, if nonzero targets in pred are not labeled, they are falsely detected
                    error_target_count += 1
                    pred_detect_flag = renew_flag(pred_detect_flag, pred, i, j)
    
    return detect_target_count / label_target_count, error_target_count / (error_target_count + detect_target_count + 1e-8)


def renew_flag(flag_mtx, pred, i, j):
    flag_mtx[i, j] = 1.0
    queue = deque([(i, j)])
    exist_mtx = np.zeros([flag_mtx.shape[0], flag_mtx.shape[1]])
    while queue:
        (x, y) = queue.popleft()
        exist_mtx[x, y] = 1.0 # all points from the queue are lebeled 1
        for ii in range(max(0, x - 1), min(flag_mtx.shape[0], x + 2)):
            for jj in range(max(0, y - 1), min(flag_mtx.shape[1], y + 2)):
                if pred[ii, jj] > 0.1 and exist_mtx[ii, jj] == 0:
                    flag_mtx[ii, jj] = 1
                    exist_mtx[ii, jj] = 1
                    queue.append((ii, jj))
    return flag_mtx
