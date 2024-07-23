from dataclasses import dataclass
import h5py
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
# from abc import ABC, abstractmethod
# import torch.nn.functional as F
import math
# from PIL import Image
import os
# from torch.utils.tensorboard import SummaryWriter
# import copy
# from tqdm.auto import tqdm
# from torchvision import transforms
# from diffusers import UNet2DModel#, UNet3DConditionModel
# from diffusers import DDPMScheduler
# from diffusers.utils import make_image_grid
from time import time
import datetime
# from pathlib import Path
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from accelerate import notebook_launcher, Accelerator
# from huggingface_hub import create_repo, upload_folder

class Dataset4h5(Dataset):
    def __init__(
        self,
        dir_name, 
        num_image=10, 
        field='brightness_temp', 
        idx='random', 
        num_redshift=512, 
        HII_DIM=64, 
        rescale=True, 
        drop_prob = 0, 
        dim=2, 
        transform=True, 
        ranges_dict=None, 
        # shuffle=False,
        ):
        super().__init__()
        
        self.dir_name = dir_name
        self.num_image = num_image
        self.idx = idx
        self.field = field
        # self.shuffle = shuffle
        self.num_redshift = num_redshift
        self.HII_DIM = HII_DIM
        self.drop_prob = drop_prob
        self.dim = dim
        self.transform = transform
        
        # if ranges_dict == None:
        #     ranges_dict = dict(
        #         images = {
        #             0: [4, 6], # ION_Tvir_MIN
        #             1: [10, 250], # HII_EFF_FACTOR
        #             },
        #         params = {
        #             0: [0, 80], # brightness_temp
        #             }
        #         ),

        self.load_h5()
        if rescale:
            rescale_start = time()
            self.images = self.rescale(self.images, ranges=ranges_dict['images'], to=[-1,1])
            self.params = self.rescale(self.params, ranges=ranges_dict['params'], to=[0,1])
            rescale_end = time()
            print(f"rescaling costs {rescale_end-rescale_start:.3f} sec")
            print(f"images rescaled to [{self.images.min()}, {self.images.max()}]")
            print(f"params rescaled to [{self.params.min()}, {self.params.max()}]")

        # from_numpy_start = time()
        self.len = len(self.params)
        self.images = torch.from_numpy(self.images)
        # from_numpy_end = time()
        # print(f"torch.from_numpy costs {from_numpy_end-from_numpy_start:.3f} sec")

        # rescale_start = time()
        cond_filter = torch.bernoulli(torch.ones(len(self.params),1)-self.drop_prob).repeat(1,self.params.shape[1]).numpy()
        self.params = torch.from_numpy(self.params*cond_filter)
        # rescale_end = time()

    def load_h5(self):
        with h5py.File(self.dir_name, 'r') as f:
            print(f"dataset content: {f.keys()}")
            max_num_image = len(f['brightness_temp'])#.shape[0]
            print(f"{max_num_image} images can be loaded")
            field_shape = f['brightness_temp'].shape[1:]
            print(f"field.shape = {field_shape}")
            self.params_keys = list(f['params']['keys'])
            print(f"params keys = {self.params_keys}")

            # if self.idx is None:
            #     if self.shuffle:
            #         self.idx = np.sort(random.sample(range(max_num_image), self.num_image))
            #         print(f"loading {self.num_image} images randomly")
            #         # print(self.idx)
            #     else:
            #         self.idx = range(self.num_image)
            #         print(f"loading {len(self.idx)} images with idx = {self.idx}")
            if self.idx == "random":
                self.idx = np.sort(random.sample(range(max_num_image), self.num_image))
                print(f"loading {self.num_image} images randomly")
                # print(self.idx)
            elif self.idx == "range":
                rank = torch.cuda.current_device()
                self.idx = range(
                    rank*self.num_image, (rank+1)*self.num_image
                    )
                print(f"loading {len(self.idx)} images with idx = {self.idx}")
            else:
                print(f"loading {len(self.idx)} images with idx = {self.idx}")

            load_start = time()
            if self.dim == 2:
                self.images = f[self.field][self.idx,0,:self.HII_DIM,-self.num_redshift:][:,None]
                # self.images = f[self.field][self.idx,:self.HII_DIM,:self.HII_DIM,-3][:,None]
                # self.images = self.images[:,:,::x_step,:]
            elif self.dim == 3:
                self.images = f[self.field][self.idx,:self.HII_DIM,:self.HII_DIM,-self.num_redshift:][:,None]
            load_end = time()
            print(f"images of shape {self.images.shape} loaded after {load_end-load_start:.3f} sec")

            transform_start = time()
            if self.transform:
                self.images = self.flip_rotate(self.images)
            # print(f"images transformed:", self.images.shape)
            transform_end = time()
            print(f"images transformed after {transform_end-transform_start:.3f} sec")

            param_start = time()
            self.params = f['params']['values'][self.idx]
            # print("params loaded:", self.params.shape)
            param_end = time()
            print(f"params of shape {self.params.shape} loaded after {param_end-param_start:.3f} sec")

            
            # plt.imshow(self.images[0,0,0])
            # plt.show()

    def flip_rotate(self, img):
        # print(f"flip_rotate, img.shape = {img.shape}")
        # num_transform = np.random.randint(img.shape[0])
        x_flip_idx = random.sample(range(len(img)), len(img)//2)
        img[x_flip_idx] = img[x_flip_idx, :, ::-1, :]
        # print(f"device{torch.cuda.current_device()}, x_flip_idx = {x_flip_idx}")
        # if img.ndim-2 == 2:
        #     img[x_flip_idx] = img[x_flip_idx, :, ::-1, :]
        if img.ndim-2 == 3:
            y_flip_idx = random.sample(range(len(img)), len(img)//2)
            xy_flip_idx = random.sample(range(len(img)), len(img)//2)
            #print(f"device{torch.cuda.current_device()}, y_flip_idx = {np.sort(y_flip_idx)}")
            #print(f"device{torch.cuda.current_device()}, xy_flip_idx = {np.sort(xy_flip_idx)}")
            # img[x_flip_idx] = img[x_flip_idx, :, ::-1, :, :]
            img[y_flip_idx] = img[y_flip_idx, :, :, ::-1, :]
            img[xy_flip_idx] = img[xy_flip_idx, :, :, :, :].transpose(0,1,3,2,4)
        return img

    def rescale(self, value, ranges, to: list):
        # print("value.ndim =", np.ndim(value))
        # print('value.shape =', value.shape)
        # ranges = self.ranges_dict[np.ndim(value)]
        # if np.ndim(value)==2:
        #     # print(f"rescale params of shape {value.shape}")
        #     ranges = \
        #         {
        #             0: [4, 6], # ION_Tvir_MIN
        #             1: [10, 250], # HII_EFF_FACTOR
        #             # 1: [np.log10(10), np.log10(250)], # HII_EFF_FACTOR
        #         }
        #     # value[:,1] = np.log10(value[:,1])
        # # elif np.ndim(value)==5:  
        # else:  
        #     # value = np.array(value)
        #     # print(f"rescale images of shape {np.shape(value)}")
        #     ranges = \
        #         {
        #             0: [0, 80], # brightness_temp
        #         }
        # print(f"value.min = {value.min()}, value.max = {value.max()}")
        for i in range(np.shape(value)[1]):
            value[:,i] = (value[:,i] - ranges[i][0]) / (ranges[i][1]-ranges[i][0])
            # print(f"i = {i}, value.min = {value[:,i].min()}, value.max = {value[:,i].max()}")
        value = value * (to[1]-to[0]) + to[0]
        return value 

    def __getitem__(self, index):
        return self.images[index], self.params[index]

    def __len__(self):
        return self.len
