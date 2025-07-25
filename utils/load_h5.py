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
from datetime import datetime
import concurrent.futures
import psutil
# from pathlib import Path
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from accelerate import notebook_launcher, Accelerator
# from huggingface_hub import create_repo, upload_folder
import socket
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import joblib
# import json
# from scipy import stats


ranges_dict = dict(
    params = {
        0: [4, 6], # ION_Tvir_MIN
        1: [10, 250], # HII_EFF_FACTOR
        },
    # images = {
    #     ## 0: [-338, 54],#[0, 80], # brightness_temp
    #     0: [-36.840145, 50.21427 * 8],
    #     #0: [-387, 86],
    #     }
    )

class Dataset4h5(Dataset):
    def __init__(
        self,
        dir_name, 
        num_image=10, 
        field='brightness_temp', 
        idx='range', 
        num_redshift=512, 
        HII_DIM=64, 
        scale_path=True, 
        drop_prob = 0, 
        dim=2, 
        transform=False, 
        #ranges_dict=None,
        num_workers=1,#len(os.sched_getaffinity(0))//torch.cuda.device_count(),
        startat=0,
        # shuffle=False,
        str_len = 120,
        squish = [1,1],
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
        self.num_workers = num_workers
        self.startat = startat 
        self.str_len = str_len
        self.ranges_dict = ranges_dict

        self.load_h5()
        if scale_path:
            scale_start = time()
            self.params = self.MinMaxScaler(self.params, ranges=ranges_dict['params'], to=[0,1])
            self.images = self.ImagesScaler(self.images, scale_path=scale_path, squish=squish)
            # self.images = self.MinMaxScaler(self.images, ranges=ranges_dict['images'], to=[-1,1])
            #scale_end = time()
            print(f"images & params scaled to [{self.images.min():.4f}, {self.images.max():.4f}] (mean={self.images.mean():.4f}, median={torch.median(self.images):.4f}, std={self.images.std():.4f}) & [{self.params.min():.4e}, {self.params.max():.6f}] after {time()-scale_start:.2f}s")

        # from_numpy_start = time()
        self.len = len(self.params)
        #self.images = torch.from_numpy(self.images)
        # from_numpy_end = time()
        # print(f"torch.from_numpy costs {from_numpy_end-from_numpy_start:.3f} s")

        cond_filter = torch.bernoulli(torch.ones(len(self.params),1)-self.drop_prob).repeat(1,self.params.shape[1]).numpy()
        self.params = torch.from_numpy(self.params*cond_filter)

    def load_h5(self):
        with h5py.File(self.dir_name, 'r') as f:
            print(f"dataset content: {f.keys()}")
            max_num_image = len(f['brightness_temp'])#.shape[0]
            field_shape = f['brightness_temp'].shape[1:]
            #print(f"field.shape = {field_shape}")
            self.params_keys = list(f['params']['keys'])
            print(f"{max_num_image} {f['brightness_temp'].dtype} images of shape {field_shape} can be loaded with params.keys {self.params_keys}")
            #print(f"params keys = {self.params_keys}")

        if self.idx == "random":
            self.idx = np.sort(random.sample(range(max_num_image), self.num_image))
            print(f"loading {self.num_image} images randomly with idx = {self.idx[:5]}...{self.idx[-5:]}")
            # print(self.idx)
        elif self.idx == "range":
            rank = torch.cuda.current_device()
            local_world_size = torch.cuda.device_count()
            self.global_rank = rank + local_world_size * int(os.environ["SLURM_NODEID"])
            self.idx = range(
                self.global_rank*self.num_image, (self.global_rank+1)*self.num_image
                )
            print(f"loading {len(self.idx)} images with idx = {self.idx}")
        else:
            print(f"loading {len(self.idx)} images with idx = {self.idx}")

        self.params = np.empty((self.num_image, len(self.params_keys)), dtype=np.float32) 
        if self.dim == 2:
            self.images = np.empty((self.num_image, 1, self.HII_DIM, self.num_redshift), dtype=np.float32)
        elif self.dim == 3:
            self.images = np.empty((self.num_image, 1, self.HII_DIM, self.HII_DIM, self.num_redshift), dtype=np.float32)
        # self.num_workers = len(os.sched_getaffinity(0))//torch.cuda.device_count()

        concurrent_init_start = time()
        if self.num_workers == 1:
            print(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.global_rank}, loading by {self.num_workers} workers, {datetime.now().strftime('%d-%H:%M:%S.%f')}".center(self.str_len, '-'))
            self.images, self.params = self.read_data_chunk(self.dir_name, self.idx, torch.cuda.current_device(), concurrent_init_start, concurrent_init_start)
            self.params = self.params.astype(self.images.dtype)
            concurrent_start = time()
            print(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.global_rank}, images {self.images.shape} & params {self.params.shape} loaded after {concurrent_start-concurrent_init_start:.3f}s, {datetime.now().strftime('%d-%H:%M:%S.%f')}".center(self.str_len, '-'))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                concurrent_init_end = time()
                print(f" {socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.global_rank}, concurrently loading by {self.num_workers}/{len(os.sched_getaffinity(0))} workers, initialized after {concurrent_init_end-concurrent_init_start:.3f}s ".center(self.str_len, '-'))
                futures = [None] * self.num_workers
                for i, idx in enumerate(np.array_split(self.idx, self.num_workers)):
                    executor_start = time()
                    futures[i] = executor.submit(self.read_data_chunk, self.dir_name, idx, torch.cuda.current_device(), concurrent_init_end, executor_start)
    
                concurrent_start = time()
                start_idx = 0
                for future in concurrent.futures.as_completed(futures):
                    images, params = future.result()
                    batch_size = params.shape[0]        
                    self.images[start_idx:start_idx+batch_size] = images
                    self.params[start_idx:start_idx+batch_size] = params
                    start_idx += batch_size
                concurrent_end = time()
                print(f" {socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.global_rank}, {start_idx} images {self.images.shape} & params {self.params.shape} loaded after {concurrent_start-concurrent_init_start:.3f}/{concurrent_end-concurrent_start:.3f}s ".center(self.str_len, '-'))

        if self.transform:
            transform_start = time()
            self.images = self.flip_rotate(self.images)
            transform_end = time()
            print(f"images transformed after {transform_end-transform_start:.3f}s")

    def read_data_chunk(self, f, idx, device, concurrent_init_end, executor_start):
        # process = psutil.Process(pid)
        # cpu_affinity = process.cpu_affinity()
        # cpu_num = psutil.Process().cpu_num()
        # print(f"cpu_num = {cpu_num}")#, cpu_affinity = {cpu_affinity}")
        set_device = time()
        #torch.cuda.set_device(device)
        open_h5py = time()
        with h5py.File(self.dir_name, 'r') as f:
            images_start = time()
            if self.dim == 2:
                #images = f[self.field][idx, :self.HII_DIM, :self.HII_DIM, self.startat][:,None]
                images = f[self.field][idx, 0, :self.HII_DIM, self.startat:self.startat+self.num_redshift][:,None]
                # images = f[self.field][idx,:self.HII_DIM,:self.HII_DIM,-3][:,None]
            elif self.dim == 3:
                images = f[self.field][idx, :self.HII_DIM, :self.HII_DIM, self.startat:self.startat+self.num_redshift][:,None]
            images_end = time()
            pid = os.getpid()
            cpu_num = psutil.Process(pid).cpu_num()

            param_start = time()
            params = f['params']['values'][idx]
            param_end = time()
            print(f"cuda:{torch.cuda.current_device()}/{self.global_rank}, CPU:{cpu_num}, images {images.shape} & params {params.shape} loaded after {executor_start-concurrent_init_end:.3f}/{set_device-executor_start:.3f}/{open_h5py-set_device:.3f}/{images_start-open_h5py:.3f}s + {images_end-images_start:.3f}s & {param_end-param_start:.3f}s")

        return images, params

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

    def MinMaxScaler(self, value, ranges, to: list):
        # print(f"value.min = {value.min()}, value.max = {value.max()}")
        for i in range(np.shape(value)[1]):
            value[:,i] = (value[:,i] - ranges[i][0]) / (ranges[i][1]-ranges[i][0])
            # if to == [0,1]:
            #     value[:,i] = (value[:,i] - ranges[i][0]) / (ranges[i][1]-ranges[i][0])
            # elif to == [-1,1]:
            #     value[:,i] = (value[:,i] - ranges[i][0]) / ranges[i][1]
        #value = value * (to[1]-to[0]) + to[0]
        return value 
    
    #def squish(self, x, Ak=[1,1]):
    #    #print(f"squish = {Ak}")
    #    A, k = Ak
    #    if k == 0:
    #        return A * x
    #    else:
    #        return A * torch.tanh(x/k)


    def ImagesScaler(self, images, scale_path, squish):
        original_shape = images.shape
        images = images.reshape(-1, original_shape[-1])
        start_time = time()
    
        # 根据 scale_path 中的关键词决定使用哪种 transformer
        if "PowerTransformer" in scale_path:
            transformer_cls = PowerTransformer
            transformer_args = dict(method='yeo-johnson', standardize=True)
        elif "QuantileTransformer" in scale_path:
            transformer_cls = QuantileTransformer
            transformer_args = dict(output_distribution='normal', random_state=0, subsample=int(2e6))
        else:
            raise ValueError("scale_path 必须包含 'PowerTransformer' 或 'QuantileTransformer' 以决定使用哪种归一化方法。")
    
        if os.path.exists(scale_path):
            preprocessor = joblib.load(scale_path)
            images[:] = preprocessor.transform(images)
            print(f"🍀 cuda:{torch.cuda.current_device()}/{self.global_rank} scaled by {scale_path} after {time()-start_time:.3f} sec 🍀")
        else:
            preprocessor = transformer_cls(**transformer_args)
            images[:] = preprocessor.fit_transform(images)
            print(f"🌱 cuda:{torch.cuda.current_device()}/{self.global_rank} fitted {scale_path} after {time()-start_time:.3f} sec 🌱")
            joblib.dump(preprocessor, scale_path)
    
        images = torch.from_numpy(images.reshape(*original_shape))
        return images
    
    # def ImagesScaler(self, images, scale_path, squish):
    #     original_shape = images.shape
    #     images = images.reshape(-1, original_shape[-1])
    #     start_time = time()
    #     if os.path.exists(scale_path):
    #         preprocessor = joblib.load(scale_path)
    #         images[:] = preprocessor.transform(images)
    #         print(f"🍀 cuda:{torch.cuda.current_device()}/{self.global_rank} scaled by power_transformer loaded from {scale_path} after {time()-start_time:.3f} sec 🍀")
    #     else:
    #         preprocessor = PowerTransformer(method='yeo-johnson', standardize=True)
    #         images[:] = preprocessor.fit_transform(images)
    #         print(f"🌱 cuda:{torch.cuda.current_device()}/{self.global_rank} fitted power_transformer after {time()-start_time:.3f} sec 🌱")
    #         joblib.dump(preprocessor, scale_path)

    #     images = torch.from_numpy(images.reshape(*original_shape))
    #     return images

    def __getitem__(self, index):
        return self.images[index], self.params[index]

    def __len__(self):
        return self.len
