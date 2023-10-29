import torch
from torch.autograd import Variable
import numpy as np


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def get_mask(img, model):
    # img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        img = to_variable(img)
        vis = model(img)
        vis = to_data(vis)

    # # DDT
    # vis = torch.squeeze(vis)
    vis = vis.numpy()

    # vis = np.transpose(vis, (1, 2, 0))
    vis = np.transpose(vis, (0, 2, 3, 1))
    # he, we, ce = vis.shape

    vis_sum = np.sum(vis, axis=3)

    return vis_sum


def convert_index_to_list(index):
    index = index[0]
    return list(zip(index[:,0], index[:,1]))


def get_FG_idx(mask, index):
    """
    Index as foreground if patch mask value >= mask.mean()
    Args:
        mask: sorted mask
        index: sorted index based on mask

    Returns: foreground index (i,j) list

    """
    mask_mean = mask.mean()
    mask_std = np.std(mask)
    FG_list = list()
    for(i,j) in index:
        if mask[i,j]>=mask_mean:
            FG_list.append((i, j))
        elif mask[i,j]<mask_mean and np.abs(mask[i,j]-mask_mean)<=0.6*mask_std:
            FG_list.append((i, j))
    return FG_list


def ax2idx(ax):
    idx=np.zeros(49)
    for [i,j] in ax:
        idx[i*7+j]=1
    return idx


def localize(imgs, model):
    two_crop_idxs = []
    two_crop_FG_ratios = []
    for i in range(2):
        # get the mask
        masks = get_mask(imgs[i], model)
        feat_size = masks.shape[1]

        idxs = []
        FG_ratios = []
        
        for mask in masks:
            index = np.dstack(np.unravel_index(np.argsort(-mask.ravel()), (feat_size, feat_size)))
            index = convert_index_to_list(index)
            sorted_mask = np.sort(mask)

            FG_index = get_FG_idx(sorted_mask,index)
            
            FG_index = np.array(FG_index).tolist()
            idx = ax2idx(FG_index)
            FG = np.count_nonzero(idx)
            FG_ratio = np.float64(FG) / 49
            idxs.append(idx)
            FG_ratios.append(FG_ratio)
        
        two_crop_idxs.append(idxs)
        two_crop_FG_ratios.append(FG_ratios)
        
    return np.array(two_crop_idxs), np.array(two_crop_FG_ratios)
