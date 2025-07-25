# %%
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
#import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
# from abc import ABC, abstractmethod
import torch.nn.functional as F
import math
# from PIL import Image
import os
#from torch.utils.tensorboard import SummaryWriter
import copy
from tqdm.auto import tqdm
# from diffusers import UNet2DModel#, UNet3DConditionModel
# from diffusers import DDPMScheduler
from datetime import datetime
from pathlib import Path
#from diffusers.optimization import get_cosine_schedule_with_warmup
#from accelerate import notebook_launcher, Accelerator
#import accelerate
#print("accelerate:", accelerate.__version__, accelerate.__path__)#, accelerate.__file__)
from huggingface_hub import create_repo, upload_folder

from utils.load_h5 import Dataset4h5, ranges_dict
from models.context_unet import ContextUnet

from huggingface_hub import notebook_login

#from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.distributed as dist

import argparse
import socket
import sys
from datetime import timedelta
from time import time, sleep

from torch.cuda.amp import autocast, GradScaler
from random import getrandbits

import subprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import wandb
#def ddp_setup_backup(rank: int, world_size: int, master_addr, master_port):
#    """
#    Args:
#       rank: Unique identifier of each process
#       world_size: Total number of processes
#    """
#    os.environ["MASTER_ADDR"] = master_addr
#    os.environ["MASTER_PORT"] = master_port
#
#    init_process_group(
#            backend="nccl", 
#            init_method=f"tcp://{master_addr}:{master_port}", 
#            rank=rank, 
#            world_size=world_size,
#            timeout=timedelta(minutes=20)
#            )

def ddp_setup():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=20))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    #print(f"🆘 global_rank = {global_rank}, torch.cuda.current_device() = {torch.cuda.current_device()}, world_size = {world_size} , local_rank = {local_rank} 🆘", flush=True)
    torch.cuda.set_device(local_rank)
    #return local_rank, global_rank, world_size 

def cosine_beta_schedule(beta_1, beta_T, timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_1, beta_T)
    #return torch.clip(betas, 0.0001, 0.9999)

class DDPMScheduler(nn.Module):
    def __init__(self, betas: tuple, num_timesteps: int, img_shape: list, device='cpu', config=None):#, dtype=torch.float16,
        super().__init__()
        
        beta_1, beta_T = betas
        assert 0 < beta_1 <= beta_T <= 1, "ensure 0 < beta_1 <= beta_T <= 1"
        self.device = device
        self.num_timesteps = num_timesteps
        self.img_shape = img_shape

        if config.beta_schedule == 'cosine':
            self.beta_t = cosine_beta_schedule(beta_1, beta_T, self.num_timesteps).to(self.device)
        elif config.beta_schedule == 'linear':
            self.beta_t = torch.linspace(beta_1, beta_T, self.num_timesteps).to(self.device)
        else:
            raise ValueError(f"Unknown beta_schedule: {config.beta_schedule}. Choose 'cosine' or 'linear'.")
        print(f"⚠️ {config.beta_schedule=}: beta_t.shape = {self.beta_t.shape}, beta_t.min() = {self.beta_t.min()}, beta_t.max() = {self.beta_t.max()}")

        self.alpha_t = 1 - self.beta_t
        self.bar_alpha_t = torch.cumprod(self.alpha_t, dim=0)
        self.config = config

    def add_noise(self, clean_images):
        shape = clean_images.shape
        expand = torch.ones(len(shape)-1, dtype=int)

        noise = torch.randn_like(clean_images).to(self.device)
        ts = torch.randint(0, self.num_timesteps, (shape[0],)).to(self.device)
                
        noisy_images = (
            clean_images * torch.sqrt(self.bar_alpha_t[ts]).view(shape[0], *expand.tolist())
            + noise * torch.sqrt(1-self.bar_alpha_t[ts]).view(shape[0], *expand.tolist())
            )

        return noisy_images, noise, ts

    def sample(self, nn_model, params, device, guide_w = 0):
        n_sample = len(params) #params.shape[0]
        x_i = torch.randn(n_sample, *self.img_shape)#.to(self.dtype)
        x_i = x_i.to(device)
        if guide_w != -1:
            c_i = params

        x_i_entire = [] # keep track of generated steps in case want to plot something

        pbar_sample = tqdm(total=self.num_timesteps, file=sys.stderr, disable=True)
        pbar_sample.set_description(f"cuda:{torch.cuda.current_device()}|{self.config.global_rank} sampling")
        for i in reversed(range(0, self.num_timesteps)):
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample)

            z = torch.randn(n_sample, *self.img_shape).to(device) if i > 0 else torch.tensor(0.)

            if guide_w == -1:
                eps = nn_model(x_i, t_is)#.to(self.dtype)
            else:
                eps = nn_model(x_i, t_is, c_i)#.to(self.dtype)
            x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z

            pbar_sample.update(1)
            
        x_i_entire = np.array(x_i_entire)
        x_i = x_i.detach().cpu().numpy()
        return x_i, x_i_entire

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted x_0 (clean data) from noisy sample x_t and predicted noise ε_θ(x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at timestep t, shape (B, C, X, Y, Z)
            t (torch.Tensor): Timesteps, shape (B,)
            noise_pred (torch.Tensor): Predicted noise ε_θ, shape same as x_t

        Returns:
            x0_hat (torch.Tensor): Predicted x_0, shape same as x_t
        """
        shape = x_t.shape
        expand = torch.ones(len(shape)-1, dtype=int)

        bar_alpha_t = self.bar_alpha_t[t].view(shape[0], *expand.tolist()).to(x_t.device)  # shape (B, 1, 1, 1, 1)
        sqrt_bar_alpha_t = bar_alpha_t.sqrt()
        sqrt_one_minus_bar_alpha_t = (1.0 - bar_alpha_t).sqrt()

        x0_hat = (x_t - sqrt_one_minus_bar_alpha_t * noise_pred) / sqrt_bar_alpha_t
        return x0_hat


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        

# %%
@dataclass
class TrainConfig:
    ###########################
    ## hardcoding these here ##
    ###########################
    push_to_hub = False #True
    hub_model_id = "Xsmos/ml21cm"
    hub_private_repo = False
    dataset_name = "/storage/home/hcoda1/3/bxia34/scratch/LEN128-DIM64-CUB8.h5"
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'
    world_size = 1#torch.cuda.device_count()
    # repeat = 2

    #dim = 2
    dim = 3#2
    stride = (2,4) if dim == 2 else (2,2,4)
    num_image = 32#0#0#640#320#6400#3000#480#1200#120#3000#300#3000#6000#30#60#6000#1000#2000#20000#15000#7000#25600#3000#10000#1000#10000#5000#2560#800#2560
    batch_size = 1#1#10#50#10#50#20#50#1#2#50#20#2#100 # 10
    n_epoch = 100#30#50#20#1#50#10#1#50#1#50#5#50#5#50#100#50#100#30#120#5#4# 10#50#20#20#2#5#25 # 120
    HII_DIM = 64
    num_redshift = HII_DIM #1024
    startat = 0 #512 #-num_redshift

    channel = 1
    img_shape = (channel, HII_DIM, num_redshift) if dim == 2 else (channel, HII_DIM, HII_DIM, num_redshift)

    #ranges_dict = dict(
    #    params = {
    #        0: [4, 6], # ION_Tvir_MIN
    #        1: [10, 250], # HII_EFF_FACTOR
    #        },
    #    images = {
    #        # 0: [-338, 54],#[0, 80], # brightness_temp
    #        0: [-387, 86],
    #        }
    #    )

    num_timesteps = 1000#1000 # 1000, 500; DDPM time steps
    # n_sample = 24 # 64, the number of samples in sampling process
    n_param = 2
    guide_w = 0#-1#0#-1#0#-1#0.1#[0,0.1] #[0,0.5,2] strength of generative guidance
    dropout = 0
    #drop_prob = 0.1 #0.28 # only takes effect when guide_w != -1
    ema=0 # whether to use ema
    ema_rate=0.995

    # seed = 0
    # save_dir = './outputs/'

    save_period = 10 #np.infty #n_epoch // 2 #np.infty#.1 # the period of sampling
    # general parameters for the name and logger    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    lrate = 1e-4
    lr_warmup_steps = 0#5#00
    output_dir = "./training/outputs/"
    save_name = os.path.join(output_dir, 'model')
    # save_period = 1 #10 # the period of saving model
    # cond = True # if training using the conditional information
    # lr_decay = False #True# if using the learning rate decay
    resume = False # if resume from the trained checkpoints
    # params_single = torch.tensor([0.2,0.80000023])
    # params = torch.tile(params_single,(n_sample,1)).to(device)
    # params =  params
    # data_dir = './data' # data directory

    #use_fp16 = True 
    #dtype = torch.float32 #if use_fp16 else torch.float32
    #mixed_precision = "no" #"fp16"
    gradient_accumulation_steps = 1

    #pbar_update_step = 20 

    channel_mult = (1,2,2,2,4)
    num_res_blocks = 2
    model_channels = 128
    # date = datetime.datetime.now().strftime("%m%d-%H%M")
    # run_name = f'{date}' # the unique name of each experiment
    str_len = 128
# config = TrainConfig()
# print("device =", config.device)

# %%
# import os
# print(os.cpu_count())
# print(len(os.sched_getaffinity(0)))
# import torch
# data = torch.randn((64,64))
# print(data.dtype)

# %%
# @dataclass

# def check_params_consistency(model, rank, world_size):
#     all_params_consistent = True
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             param_tensor = param.detach().clone()
#             dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)
#             param_tensor /= world_size

#             if not torch.allclose(param_tensor, param.detach()):
#                 all_params_consistent = False
#                 if rank == 0:
#                     print(f"Parameter {name} is not consistent across GPUs.")
#     if rank == 0 and all_params_consistent:
#         print("All model parameters are consistent across GPUs.")
#     return all_params_consistent

# def check_gradients_consistency(model, rank, world_size):
#     all_gradients_consistent = True
#     for name, param in model.named_parameters():
#         if param.requires_grad and param.grad is not None:
#             grad_tensor = param.grad.detach().clone()
#             dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
#             grad_tensor /= world_size

#             if not torch.allclose(grad_tensor, param.grad.detach()):
#                 all_gradients_consistent = False
#                 if rank == 0:
#                     print(f"Gradient {name} is not consistent across GPUs.")
#     if rank == 0 and all_gradients_consistent:
#         print("All model gradients are consistent across GPUs.")
#     return all_gradients_consistent
def get_gpu_info(device):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = reserved_memory - allocated_memory
    return {
        'total': int(total_memory / 1024**2),
        'used': int(allocated_memory / 1024**2),
        'free': int(free_memory / 1024**2),
    }

class DDPM21CM:
    def __init__(self, config):
        self.config = config
        #self.ddpm = DDPMScheduler(betas=(0.0001, 0.9999), num_timesteps=config.num_timesteps, img_shape=config.img_shape, device=config.device, config=config,)#, dtype=config.dtype
        self.ddpm = DDPMScheduler(betas=(1e-4, 0.02), num_timesteps=config.num_timesteps, img_shape=config.img_shape, device=config.device, config=config,)#, dtype=config.dtype

        # initialize the unet
        self.nn_model = ContextUnet(
            n_param=config.n_param, 
            image_size=config.HII_DIM, 
            dim=config.dim, 
            stride=config.stride, 
            channel_mult=config.channel_mult, 
            use_checkpoint=config.use_checkpoint, 
            dropout=config.dropout,
            num_res_blocks = config.num_res_blocks,
            model_channels = config.model_channels,
        )#, dtype=config.dtype)

        self.nn_model.train()
        self.nn_model.to(self.ddpm.device)

        self.nn_model = DDP(self.nn_model, device_ids=[self.ddpm.device], find_unused_parameters=False)

        self.optimizer = torch.optim.AdamW(self.nn_model.module.parameters(), lr=config.lrate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer = self.optimizer,
                T_max = int(config.num_image / config.batch_size * config.n_epoch / config.gradient_accumulation_steps),
                )

        # whether to use ema
        if config.ema:
            self.ema = EMA(config.ema_rate)
            self.ema_model = ContextUnet(
                    n_param=config.n_param, 
                    image_size=config.HII_DIM, 
                    dim=config.dim, 
                    stride=config.stride, 
                    channel_mult=config.channel_mult, 
                    use_checkpoint=config.use_checkpoint, 
                    dropout=config.dropout,
                    num_res_blocks = config.num_res_blocks,
                    model_channels = config.model_channels,
                )
            #self.ema_model.train()
            self.ema_model.eval().requires_grad_(False)
            self.ema_model.to(self.ddpm.device)

        self.global_step = 0
        self.start_epoch = 0
        if config.resume and os.path.exists(config.resume):
            if dist.is_initialized():
                map_loc = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            else:
                map_loc = f"cuda:{torch.cuda.current_device()}"
            checkpoint = torch.load(config.resume, map_location=map_loc)

            self.nn_model.module.load_state_dict(checkpoint['unet_state_dict'])
            if not config.reset_epoch:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step']
            print(f"🍀 {config.run_name} cuda:{torch.cuda.current_device()}|{self.config.global_rank} resumed nn_model from {config.resume} with {sum(x.numel() for x in self.nn_model.module.parameters())*1e-6:.2f}M parameters 🍀".center(self.config.str_len,'+'))#, flush=True)
            if config.ema:
                self.ema_model.load_state_dict(checkpoint['ema_unet_state_dict'])
                print(f"🍀 {config.run_name} cuda:{torch.cuda.current_device()}|{self.config.global_rank} resumed ema_model from {config.resume} with {sum(x.numel() for x in self.ema_model.parameters())*1e-6:.2f}M parameters 🍀".center(self.config.str_len,'+'))

        else:
            print(f"🌱 {config.run_name} cuda:{torch.cuda.current_device()}|{self.config.global_rank} initialized nn_model randomly with {sum(x.numel() for x in self.nn_model.module.parameters())*1e-6:.2f}M parameters 🌱".center(self.config.str_len,'+'))#, flush=True)
            if config.ema:
                self.ema_model = copy.deepcopy(self.nn_model.module).eval().requires_grad_(False).to(self.ddpm.device)
                print(f"🌱 {config.run_name} cuda:{torch.cuda.current_device()}|{self.config.global_rank} initialized ema_model randomly with {sum(x.numel() for x in self.ema_model.parameters())*1e-6:.2f}M parameters 🌱".center(self.config.str_len,'+'))

        self.ranges_dict = ranges_dict
        self.scaler = GradScaler()

        self.transform_vmap = torch.vmap(self.transform, in_dims=0)

    def load(self):
        num_workers = len(os.sched_getaffinity(0))//self.config.world_size
        min_num_workers = min(1,num_workers)
        dataset = Dataset4h5(
            self.config.dataset_name, 
            num_image=self.config.num_image,
            idx = 'range',#"random",#
            HII_DIM=self.config.HII_DIM, 
            num_redshift=self.config.num_redshift,
            startat=self.config.startat,
            scale_path=self.config.scale_path,
            #drop_prob=self.config.drop_prob, 
            dim=self.config.dim,
            #ranges_dict=self.ranges_dict,
            num_workers=min_num_workers,
            str_len = self.config.str_len,
            squish = self.config.squish,
            )

        dataloader_start = time()
        self.dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,#False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # sampler=DistributedSampler(dataset),
            )

        if len(self.dataloader) % self.config.gradient_accumulation_steps != 0:
            raise ValueError(f"len(self.dataloader) % self.config.gradient_accumulation_steps = {len(self.dataloader) % self.config.gradient_accumulation_steps} instead of 0. Make sure len(dataloader)={len(self.dataloader)} is dividable by gradient_accumulation_steps={self.config.gradient_accumulation_steps}.")

        dataloader_end = time()
        #print(f"cuda:{torch.cuda.current_device()}|{self.config.global_rank} dataloader costs {dataloader_end-dataloader_start:.3f}s")

        del dataset

    def transform(self, img):
        if getrandbits(1):
            img = torch.flip(img, dims=[-2])
        if img.ndim == 4:
            if getrandbits(1):
                img = torch.flip(img, dims=[-3])
            if getrandbits(1):
                img = img.transpose(-2,-3)

        ##flip along x or y or both
        #flip_xy = [dim + 1 for dim in range(2) if getrandbits(1)]
        #img = torch.flip(img, dims=flip_xy) 
        ## flip diagonally 
        #if getrandbits(1):
        #    img = img.transpose(-2,-3) #.contiguous()
        #torch.distributed.breakpoint()
        return img

    def squish(self, x, Ak):
        start_time = time()
        A, k = Ak
        if k == 0:
            y = A * x
        else:
            y = A * torch.sign(x) * torch.log1p(torch.abs(x) / k) #A * torch.tanh(x/k)
        #print(f"squish = {Ak}; cuda:{torch.cuda.current_device()}|{self.config.global_rank}; {time()-start_time:.3f} sec")
        return y

    def inverse_squish(self, y, Ak):
        start_time = time()
        A, k = Ak
        if k == 0:
            x = 1/A * y
        else:
            x = k * np.sign(y) * np.expm1(np.abs(y) / A) #k * np.arctanh(y/A)
        print(f"inverse_squish = {Ak}: {time()-start_time:.3f} sec, {y.min()=}, {y.max()=}, {x.min()=}, {x.max()=}")
        return x

    def train(self):
        ###################      
        ## training loop ##
        ###################

        if self.config.global_rank == 0: # or torch.cuda.current_device() == 0:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            if self.config.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.config.hub_model_id or Path(self.config.output_dir).name, exist_ok=True
                ).repo_id
            #self.accelerator.init_trackers(f"{self.config.run_name}")
            #self.config.logger = SummaryWriter(f"logs/{self.config.run_name}") 
            wandb.init(
                    project = "ml21cm",
                    name = self.config.run_name,
                    config = self.config,
            )

        self.load()
        print(f"cuda:{torch.cuda.current_device()}|{self.config.global_rank} training 🚀 ")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for ep in range(self.start_epoch, self.config.n_epoch):
            self.ddpm.train()
            pbar_train = tqdm(total=len(self.dataloader), file=sys.stderr, disable=True)#, mininterval=self.config.pbar_update_step)#, disable=True)#not self.accelerator.is_local_main_process)
            pbar_train.set_description(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}|{self.config.global_rank} Epoch {ep}")
            epoch_start = time()

            for i, (x, c) in enumerate(self.dataloader):
                x = self.transform_vmap(x)
                x = x.to(self.config.device)#.to(self.config.dtype)
                x = self.squish(x, Ak=self.config.squish)
                # autocast forward propogation
                with autocast(enabled=self.config.autocast):
                    xt, noise, ts = self.ddpm.add_noise(x)

                    if self.config.guide_w == -1:
                        noise_pred = self.nn_model(xt, ts)#.to(x.dtype)
                    else:
                        c = c.to(self.config.device)
                        noise_pred = self.nn_model(xt, ts, c)#.to(x.dtype)
                    
                    #print(f"ep = {ep}, noise_pred.shape = {noise_pred.shape}")
                    loss = F.mse_loss(noise, noise_pred)

                    #if self.config.amp_loss_weight > 0:
                    #    x0_hat = self.ddpm.predict_x0(xt, ts, noise_pred)
                    #    amp_pred = x0_hat.mean(axis=(1,2,3)) if self.config.dim == 3 else x0_hat.mean(axis=(1,2))
                    #    amp_real = x.mean(axis=(1,2,3)) if self.config.dim == 3 else x.mean(axis=(1,2))
                    #    loss += F.mse_loss(amp_pred, amp_real) * self.config.amp_loss_weight
                    #    loss /= 1+self.config.amp_loss_weight # normalize loss by amp_loss_weight
                    #    # print(f"⚠️ {x0_hat.shape=}; {amp_pred.shape=}, {amp_real.shape=}, {loss.item()=}, {self.config.amp_loss_weight=}")

                    loss /= self.config.gradient_accumulation_steps

                    #print(f"ep = {ep}, loss = {loss}")
                    if torch.isnan(loss).any():
                        raise ValueError(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}|{self.config.global_rank} Epoch {ep}, loss: {loss}")

                # scaler backward propogation
                self.scaler.scale(loss).backward()

                if (i+1) % self.config.gradient_accumulation_steps == 0:
                    #Jprint(f"ep = {ep}, before unscale")
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.nn_model.module.parameters(), max_norm=1.0)

                    #print(f"ep = {ep}, before scaler")
                    self.scaler.step(self.optimizer)
                    #print(f"ep = {ep}, before step")
                    self.lr_scheduler.step()

                    #print(f"ep = {ep}, before scaler.update")
                    self.scaler.update()
                    #print(f"ep = {ep}, after scaler.update")
                    self.optimizer.zero_grad()

                    # ema update
                    if self.config.ema:
                        self.ema.step_ema(self.ema_model, self.nn_model.module)

                #if (i+1) % self.config.pbar_update_step == 0:
                pbar_train.update(1)#self.config.pbar_update_step)

                logs = dict(
                    loss=loss.detach().item(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    step=self.global_step
                )
                pbar_train.set_postfix(**logs)

                if self.config.global_rank == 0:
                    wandb.log({
                        'MSE': logs['loss'],
                        'learning_rate': logs['lr'],
                        'global_step': self.global_step,
                        'epoch': ep,
                        })
                self.global_step += 1

            self.save(ep)
            print(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}|{self.config.global_rank} Epoch{ep}:{i+1}/{len(self.dataloader)} costs {(time()-epoch_start)/60:.2f} min")#, flush=True)

    def save(self, ep):
        if ep == self.config.n_epoch-1 or (ep+1) % self.config.save_period == 0:
            if self.config.global_rank == 0:# or torch.cuda.current_device() == 0:
                self.nn_model.eval()
                with torch.no_grad():
                    if self.config.push_to_hub:
                        upload_folder(
                            repo_id = self.repo_id,
                            folder_path = ".",#config.output_dir,
                            commit_message = f"{self.config.run_name}",
                            ignore_patterns = ["step_*", "epoch_*", "*.npy", "__pycache__"],
                            )
                    if self.config.save_name:
                        model_state = {
                            'epoch': ep,
                            'global_step': self.global_step,
                            'unet_state_dict': self.nn_model.module.state_dict(),
                            'ema_unet_state_dict': self.ema_model.state_dict() if self.config.ema else None,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.lr_scheduler.state_dict(),
                            }
                        save_name = self.config.save_name + f"-epoch{ep+1}.pt"
                        torch.save(model_state, save_name)
                        print(f'🌟 cuda:{torch.cuda.current_device()}|{self.config.global_rank} saved model at ' + save_name)

    def rescale(self, params, ranges, to: list):
        # value = np.array(params).copy()
        value = params.clone()

        if value.ndim == 1:
            value = value.view(-1,len(value))
            
        for i in range(np.shape(value)[1]):
            value[:,i] = (value[:,i] - ranges[i][0]) / (ranges[i][1]-ranges[i][0])
            # print(f"i = {i}, value.min = {value[:,i].min()}, value.max = {value[:,i].max()}")
        value = value * (to[1]-to[0]) + to[0]
        return value 

    def sample(self, params:torch.tensor=None, num_new_img_per_gpu=192, entire=False, save=True):
        # n_sample = params.shape[0]
        # file = self.config.resume

        # print(f"cuda:{torch.cuda.current_device()}, sample, params = {params}")
        if params is None:
            params = torch.tensor([4.4, 131.341])

        elif type(params) is not torch.Tensor:
            params = torch.tensor(params)
            # params_backup = params.numpy().copy()
        # else:
        params_backup = params.numpy().copy()
        params_normalized = self.rescale(params, self.ranges_dict['params'], to=[0,1])

        if self.config.global_rank == 0:
            print(f"🚀 sampling {num_new_img_per_gpu} images with params = {params_backup}, {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚀")#, flush=True)
            #print(f"🚀 sampling {num_new_img_per_gpu} images with normalized params = {params_normalized}, {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚀")#, flush=True)

        params_normalized = params_normalized.repeat(num_new_img_per_gpu,1)
        assert params_normalized.dim() == 2, "params_normalized must be a 2D torch.tensor"
        # print("params =", params)

        self.nn_model.module.eval()
        if self.config.ema:
            self.ema_model.eval()

        sample_start = time()
        with torch.no_grad():
            with autocast(enabled=self.config.autocast):
                x_last, x_entire = self.ddpm.sample(
                    nn_model=self.nn_model.module, 
                    params=params_normalized.to(self.config.device), 
                    device=self.config.device, 
                    guide_w=self.config.guide_w,
                    )
                x_last = self.inverse_squish(x_last, self.config.squish)
                #x_entire = self.inverse_squish(x_entire, self.config.squish)

                if self.config.ema:
                    x_last_ema, x_entire_ema = self.ddpm.sample(
                        nn_model=self.ema_model, 
                        params=params_normalized.to(self.config.device), 
                        device=self.config.device, 
                        guide_w=self.config.guide_w,
                        )
                    x_last_ema = self.inverse_squish(x_last_ema, self.config.squish)
                    #x_entire_ema = self.inverse_squish(x_entire_ema, self.config.squish)

        if save:    
            # np.save(os.path.join(self.config.output_dir, f"{self.config.run_name}{'ema' if ema else ''}.npy"), x_last)
            savetime = datetime.now().strftime("%d%H%M%S")
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

            for ema in range(self.config.ema + 1):
                elapsed_time = (time()-sample_start)/(self.config.ema + 1)
                savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]:.3f}-zeta{params_backup[1]:.3f}-device{self.config.global_rank}-{Path(self.config.resume).stem}-{savetime}-ema{ema}")
                np.save(savename, x_last if ema==0 else x_last_ema)
                print(f"cuda:{torch.cuda.current_device()}|{self.config.global_rank} saved {x_last.shape} to {os.path.basename(savename)} with {elapsed_time/60:.2f} min")#, flush=True)

                if entire:
                    #savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]:.3f}-zeta{params_backup[1]:.3f}-device{self.config.global_rank}-{os.path.basename(self.config.resume)}-{savetime}-ema{ema}_entire")
                    savename += '_entire'
                    np.save(savename, x_entire)
                    print(f"cuda:{torch.cuda.current_device()}|{self.config.global_rank} saved images of shape {x_entire.shape} to {savename}")

        #if dist.is_initialized():
        #    print(f"🗿 global_rank = {self.config.global_rank}, barrier, {datetime.now().strftime('%d-%H:%M:%S.%f')} 🗿", flush=True)
        #    dist.barrier()
        #    torch.cuda.empty_cache()
        #    torch.cuda.synchronize()

        #sleep(sleep_time)
        #print(f"🆘 cuda:{torch.cuda.current_device()}|{self.config.global_rank} end of DDPM21CM.sample at {datetime.now().strftime('%d-%H:%M:%S.%f')} 🆘", flush=True)
        # else:
        #return x_last
# %%

#num_train_image_list = [6000]#[60]#[8000]#[1000]#[100]#
#def train_backup(rank, world_size, local_world_size, master_addr, master_port, config):
#    global_rank = rank + local_world_size * int(os.environ["SLURM_NODEID"])
#    ddp_setup(global_rank, world_size, master_addr, master_port)
#    torch.cuda.set_device(rank)
#
#    config.device = f"cuda:{rank}"
#    config.world_size = local_world_size
#    config.global_rank = global_rank 
#
#    ddpm21cm = DDPM21CM(config)
#    ddpm21cm.train()
#
#    if dist.is_initialized():
#        print(f"🚥 cuda:{local_rank}|{global_rank} dist.destroy_process_group started at {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚥")#, flush=True)
#        dist.barrier()
#        torch.cuda.empty_cache()
#        torch.cuda.synchronize()
#        dist.destroy_process_group()
#        print(f"✅ cuda:{local_rank}|{global_rank} dist.destroy_process_group completed at {datetime.now().strftime('%d-%H:%M:%S.%f')} ✅")#, flush=True)

def train(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    config.device = f"cuda:{local_rank}"
    config.world_size = world_size
    config.global_rank = global_rank

    print(f"⛳️ training cuda:{local_rank}|{global_rank}/{world_size} {datetime.now().strftime('%d-%H:%M:%S.%f')} ⛳️".center(config.str_len,'#'))
    ddpm21cm = DDPM21CM(config)
    ddpm21cm.train()

    if dist.is_initialized():
        dist.barrier()

    #if dist.is_initialized():
    #    print(f"🚥 cuda:{local_rank}|{global_rank} dist.destroy_process_group started at {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚥")#, flush=True)
    #    dist.barrier()
    #    torch.cuda.empty_cache()
    #    torch.cuda.synchronize()
    #    dist.destroy_process_group()
    #    print(f"✅ cuda:{local_rank}|{global_rank} dist.destroy_process_group completed at {datetime.now().strftime('%d-%H:%M:%S.%f')} ✅")#, flush=True)

def generate_samples(config, num_new_img_per_gpu, max_num_img_per_gpu, params_pairs):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    config.device = f"cuda:{local_rank}"
    config.world_size = world_size
    config.global_rank = global_rank

    print(f"⛳️ sampling cuda:{local_rank}|{global_rank}/{world_size} {datetime.now().strftime('%d-%H:%M:%S.%f')} ⛳️".center(config.str_len,'#'))
    if dist.is_initialized():
        dist.barrier()

    ddpm21cm = DDPM21CM(config)

    for params in params_pairs:
        for _ in range(num_new_img_per_gpu // max_num_img_per_gpu):    
            ddpm21cm.sample(
                params=params, 
                num_new_img_per_gpu=max_num_img_per_gpu,
                )
        if num_new_img_per_gpu % max_num_img_per_gpu:
            ddpm21cm.sample(
                params=params, 
                num_new_img_per_gpu=num_new_img_per_gpu % max_num_img_per_gpu,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=False, help="whether to train the model", default=False)
    parser.add_argument("--sample", type=int, required=False, help="whether to sample", default=1)
    parser.add_argument("--resume", type=str, required=False, help="filename of the model to resume", default=False)
    parser.add_argument("--reset-epoch", "--restart-epoch-zero", action="store_true", help="If set, resumes training from a checkpoint but resets the epoch counter to 0.")
    parser.add_argument("--num_new_img_per_gpu", type=int, required=False, default=4)
    parser.add_argument("--max_num_img_per_gpu", type=int, required=False, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=1) # as tested, higher value leads to slower training and higher loss in the end
    parser.add_argument("--num_image", type=int, required=False, default=32)
    parser.add_argument("--n_epoch", type=int, required=False, default=50)
    parser.add_argument("--batch_size", type=int, required=False, default=2)
    parser.add_argument("--channel_mult", type=float, nargs="+", required=False, default=(1,2,2,2,4))
    parser.add_argument("--autocast", type=int, required=False, default=1)
    parser.add_argument("--use_checkpoint", type=int, required=False, default=1)
    parser.add_argument("--dropout", type=float, required=False, default=0)
    parser.add_argument("--lrate", type=float, required=False, default=1e-5)
    parser.add_argument("--dim", type=int, required=False, default=3)
    parser.add_argument("--num_redshift", type=int, required=False, default=64)
    parser.add_argument("--num_res_blocks", type=int, required=False, default=1)
    parser.add_argument("--model_channels", type=int, required=False, default=128)
    parser.add_argument("--stride", type=int, nargs="+", required=False, default=(2,2,1))
    parser.add_argument("--squish", type=float, nargs="+", required=False, default=(1,1))
    parser.add_argument("--guide_w", type=int, required=False, default=0)
    parser.add_argument("--ema", type=int, required=False, default=0)
    parser.add_argument("--scale_path", required=True, type=str, help="scale for the model")
    parser.add_argument("--beta_schedule", required=True, type=str)
    #parser.add_argument("--amp_loss_weight", type=float, required=False, default=0.0, help="weight for the amp loss")

    args = parser.parse_args()

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    local_world_size = torch.cuda.device_count()
    total_nodes = int(os.environ["SLURM_NNODES"])
    world_size = local_world_size * total_nodes #6#int(os.environ["SLURM_NTASKS"])

    config = TrainConfig()
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.num_image = args.num_image
    config.n_epoch = args.n_epoch
    config.batch_size = args.batch_size
    config.channel_mult = args.channel_mult
    config.model_channels = args.model_channels
    config.autocast = bool(args.autocast)
    config.use_checkpoint = bool(args.use_checkpoint)
    config.dropout = args.dropout
    config.lrate = args.lrate
    config.resume = args.resume
    config.reset_epoch = args.reset_epoch
    config.guide_w = args.guide_w
    config.ema = args.ema
    config.scale_path = args.scale_path
    config.beta_schedule = args.beta_schedule
    #config.amp_loss_weight = args.amp_loss_weight
    #config.sample = args.sample

    config.stride = args.stride #(2,2) if config.dim == 2 else (2,2,1)
    config.dim = len(config.stride) #args.dim
    #print(config.stride, config.dim)
    config.num_redshift = args.num_redshift
    config.squish = args.squish

    config.img_shape = (config.channel, config.HII_DIM, config.num_redshift) if config.dim == 2 else (config.channel, config.HII_DIM, config.HII_DIM, config.num_redshift)
    config.num_res_blocks = args.num_res_blocks

    config.run_name = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%d%H%M%S")) # the unique name of each experiment
    config.save_name += f"-N{config.num_image}-device_count{local_world_size}-node{total_nodes}-{config.run_name}"

    if not dist.is_initialized():
        ddp_setup()
   
    ############################ training ################################
    if args.train:
        config.dataset_name = args.train
        train(config)
        config.resume = config.save_name + f"-epoch{config.n_epoch}.pt"

    ############################ sampling ################################
    if os.path.exists(config.resume) and args.sample:
        num_new_img_per_gpu = args.num_new_img_per_gpu#200#4#200
        max_num_img_per_gpu = args.max_num_img_per_gpu#40#2#20
        params_pairs = [
            (4.4, 131.341),
            (5.6, 19.037),
            (4.699, 30),
            (5.477, 200),
            (4.8, 131.341),
        ]
        generate_samples(config, num_new_img_per_gpu, max_num_img_per_gpu, params_pairs)
    else:
        print(f'🆘 os.path.exists({config.resume}) = {os.path.exists(config.resume)} 🆘')

    if dist.is_initialized():
        dist.barrier()
        print(f"🚥 cuda:{os.environ['LOCAL_RANK']}|{os.environ['RANK']} dist.destroy_process_group {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚥")#, flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        #dist.destroy_process_group()
