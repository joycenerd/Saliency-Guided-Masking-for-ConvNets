import torch

class RandomGridMask(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, sample, Mean, FG_ratio, FG_idx, BG_idx, mask_ratio):
        FG_mask_num = round(mask_ratio.item() * FG_ratio * 49)
        BG_mask_num = int(49 * mask_ratio) - FG_mask_num

        if len(FG_idx.shape) != 0:
            FG_rows, FG_cols, _ = self._get_mask_idx(FG_idx, FG_mask_num)
        else:
            FG_rows = torch.IntTensor([])
            FG_cols = torch.IntTensor([])
            
        if self.mode=="pos" and len(BG_idx.shape) != 0:
            BG_rows, BG_cols, _ = self._get_mask_idx(BG_idx, BG_mask_num)
        else:
            BG_rows = torch.IntTensor([])
            BG_cols = torch.IntTensor([])

        rows = torch.cat((FG_rows, BG_rows))
        cols = torch.cat((FG_cols, BG_cols))


        for (r,c) in zip(rows, cols):
            sample[0, r * 32:(r+1)*32, c*32:(c+1)*32] = Mean[0]
            sample[1, r * 32:(r+1)*32, c*32:(c+1)*32] = Mean[1]
            sample[2, r * 32:(r+1)*32, c*32:(c+1)*32] = Mean[2]

        return sample 
    
    def _get_mask_idx(self, idxs, mask_num):
        shuff_idx = torch.randperm(idxs.shape[0])
        idxs = idxs[shuff_idx].view(-1)
        mask_idx = idxs[:mask_num]
        rows=(mask_idx/7).int()
        cols=(mask_idx%7).int()
        return rows, cols, mask_idx


class Masking(object):
    def __init__(self, mask_ratio, mode):
        self.grid_mask = RandomGridMask(mode)
        self.mode = mode
        self.mask_ratio = mask_ratio
        
    def __call__(self, imgs, patches, FG_ratios):
        B = imgs.shape[0]
        
        mask_ratio = torch.FloatTensor(B).uniform_(float(self.mask_ratio[0]), float(self.mask_ratio[1]))
        Mean = torch.mean(imgs, dim = [2,3])

        for b in range(B):
            FG_idx = patches[b].nonzero().squeeze()
            BG_idx = (patches[b]==0).nonzero().squeeze()
            imgs[b] = self.grid_mask(imgs[b], Mean[b], FG_ratios[b], FG_idx, BG_idx, mask_ratio[b])

        return imgs
