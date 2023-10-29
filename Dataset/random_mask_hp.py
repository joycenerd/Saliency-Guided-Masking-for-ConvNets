import torch
import torchvision
import numbers
import random



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class FocalMask(object):
    def __init__(self, size, noise_std, mode):
        self.t = torchvision.transforms.RandomCrop(size)
        self.noiseT = torchvision.transforms.Compose([AddGaussianNoise(0., noise_std)])
        self.size = size
        self.mode = mode
        
    def __call__(self, sample, noise_patch):
        crop_size = self.t.get_params(sample, (self.size, self.size))
        [i, j, h, w] = crop_size
        r1, r2, c1, c2 = i, i+h, j, j+w

        if self.mode == "pos": 
            sample[:, :r1, :] = self.noiseT(torch.zeros(3, r1, 224)) # top
            sample[:, r2:, :] = self.noiseT(torch.zeros(3, 224-r2, 224)) # bottom
            sample[:, r1:r2, :c1] = self.noiseT(torch.zeros(3, h, c1)) # left
            sample[:, r1:r2, c2:] = self.noiseT(torch.zeros(3, h, 224-c2)) # right
        elif self.mode == "neg":
            sample[:, r1:r2, c1:c2] = noise_patch
        
        return sample


class RandomGridMask(object):
    def __init__(self, mask_ratio, noise_std, mode):
        self.mask_ratio = mask_ratio
        self.noiseT = torchvision.transforms.Compose([AddGaussianNoise(0., noise_std)])
        self.mode = mode

    def __call__(self, sample, FG_ratio, FG_idx, BG_idx, strat, noise_patch):
        if isinstance(self.mask_ratio, numbers.Number):
            mask_ratio = self.mask_ratio
        elif isinstance(self.mask_ratio, list):
            mask_ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        
        if mask_ratio*FG_ratio>=0.56:
            ratio = 0.55
        else:
            ratio = mask_ratio*FG_ratio
        FG_mask_num = round(ratio * 49)
        BG_mask_num = int(49 * mask_ratio) - FG_mask_num
        
        if strat == "spatial":
            if len(FG_idx.shape) != 0:
                FG_rows, FG_cols = self._get_mask_idx(FG_idx, FG_mask_num)
            else:
                FG_rows = torch.IntTensor([])
                FG_cols = torch.IntTensor([])
            
            if self.mode == "pos" and len(BG_idx.shape) != 0:
                BG_rows, BG_cols = self._get_mask_idx(BG_idx, BG_mask_num)
            else:
                BG_rows = torch.IntTensor([])
                BG_cols = torch.IntTensor([])
            rows = torch.cat((FG_rows, BG_rows))
            cols = torch.cat((FG_cols, BG_cols))

            for (r,c) in zip(rows, cols):
                sample[:, r * 32:(r+1)*32, c*32:(c+1)*32] = noise_patch
        elif strat == "channel":
            for i in range(3):
                if len(FG_idx.shape) != 0:
                    FG_rows, FG_cols = self._get_mask_idx(FG_idx, FG_mask_num)
                else:
                    FG_rows = torch.IntTensor([])
                    FG_cols = torch.IntTensor([])
                
                if self.mode == "pos" and len(BG_idx.shape) != 0:
                    BG_rows, BG_cols = self._get_mask_idx(BG_idx, BG_mask_num)
                else:
                    BG_rows = torch.IntTensor([])
                    BG_cols = torch.IntTensor([])
                rows = torch.cat((FG_rows, BG_rows))
                cols = torch.cat((FG_cols, BG_cols))

                for (r,c) in zip(rows, cols):
                    sample[i, r * 32:(r+1)*32, c*32:(c+1)*32] = noise_patch[i]

        return sample
    
    def _get_mask_idx(self, idxs, mask_num):
        shuff_idx = torch.randperm(idxs.shape[0])
        idxs = idxs[shuff_idx].view(-1)
        mask_idx = idxs[:mask_num]
        rows=(mask_idx/7).int()
        cols=(mask_idx%7).int()
        return rows, cols


class Masking(object):
    def __init__(self, mask_ratio, noise_std, crop_size, mode):
        self.grid_mask = RandomGridMask(mask_ratio, noise_std, mode)
        self.focal_mask = FocalMask(crop_size, noise_std, mode)
        self.noiseT = torchvision.transforms.Compose([AddGaussianNoise(0., noise_std)])
        self.noise_std = noise_std
        self.mode = mode
        self.crop_size = crop_size
        
    def __call__(self, imgs, patches, FG_ratios):
        B = imgs.shape[0]    
        prob1 = torch.rand(B)
        
        spatial_noise, channel_noise, focal_noise = self._gen_noise_patch()
        
        for b in range(B):
            FG_idx = patches[b].nonzero().squeeze()
            BG_idx = (patches[b]==0).nonzero().squeeze()

            if prob1[b] < 0.2:
                imgs[b] = self.focal_mask(imgs[b], focal_noise)
        
            else:
                prob2 = torch.rand(1)
                if prob2 < 0.3:
                    imgs[b] = self.grid_mask(imgs[b], FG_ratios[b], FG_idx, BG_idx, "spatial", spatial_noise)
                else:
                    imgs[b] = self.grid_mask(imgs[b], FG_ratios[b], FG_idx, BG_idx, "channel", channel_noise)

        return imgs

    def _gen_noise_patch(self):
        # spatical-wise noise
        spatial_noise = self.noiseT(torch.zeros((3,32,32)))
        
        # channel-wise noise
        c1 = self.noiseT(torch.zeros((1,32,32)))
        c2 = self.noiseT(torch.zeros((1,32,32)))
        c3 = self.noiseT(torch.zeros((1,32,32)))
        channel_noise = torch.vstack((c1,c2,c3))

        # focal noise for negative sample
        focal_noise = None
        if self.mode == "neg":
            focal_noise = self.noiseT(torch.zeros(3,self.crop_size,self.crop_size))
        
        return spatial_noise, channel_noise, focal_noise
         