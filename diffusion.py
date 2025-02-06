# %%
import logging
#logging.getLogger("torch").setLevel(logging.ERROR)
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
from torch.utils.tensorboard import SummaryWriter
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

from load_h5 import Dataset4h5
from context_unet import ContextUnet

from huggingface_hub import notebook_login

import torch.multiprocessing as mp
#from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import argparse
import socket
import sys
from datetime import timedelta
from time import time

from torch.cuda.amp import autocast, GradScaler
from random import getrandbits

import subprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %%
def ddp_setup(rank: int, world_size: int, master_addr, master_port):
    """
    Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
    """

    #print("inside ddp_setup")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    #print("ddp_setup, rank =", rank)
    init_process_group(
            backend="nccl", 
            init_method=f"tcp://{master_addr}:{master_port}", 
            rank=rank, 
            world_size=world_size,
            timeout=timedelta(minutes=20)
            )

# %%
# notebook_login()

# %% [markdown]
# # Add noise:
# 
# \begin{align*}
# x_t &\sim \mathcal N\left(\sqrt{1-\beta_t}\ x_{t-1},\ \beta_t \right) \\
# x_t &\equiv \sqrt{1-\beta_t}\ x_{t-1} + \sqrt{\beta_t}\ \epsilon\\
# \epsilon &\sim \mathcal N(0,1)\\
# \alpha_t & \equiv 1 - \beta_t\\
# & ...\\
# x_t &= \sqrt{\bar {\alpha_t}} x_0 + \epsilon\ \sqrt{1 - \bar{\alpha_t}}\\
# \bar {\alpha_t} &\equiv \prod_{i=1}^t \alpha_i\\
# &= \exp\left({\ln{\prod_{i=1}^t \alpha_i}}\right)\\
# &= \exp\left({\sum_{i=1}^t\ln{ \alpha_i}}\right)
# \end{align*}

# %%
class DDPMScheduler(nn.Module):
    def __init__(self, betas: tuple, num_timesteps: int, img_shape: list, device='cpu', config=None):#, dtype=torch.float16,
        super().__init__()
        #self.dtype = dtype#torch.float16 if self.use_fp16 else torch.float32
        
        beta_1, beta_T = betas
        assert 0 < beta_1 <= beta_T <= 1, "ensure 0 < beta_1 <= beta_T <= 1"
        self.device = device
        self.num_timesteps = num_timesteps
        self.img_shape = img_shape
        self.beta_t = torch.linspace(beta_1, beta_T, self.num_timesteps) #* (beta_T-beta_1) + beta_1
        #self.beta_t = self.beta_t.to(self.dtype)
        self.beta_t = self.beta_t.to(self.device)

        # self.drop_prob = drop_prob
        # self.cond = cond
        self.alpha_t = 1 - self.beta_t
        # self.bar_alpha_t = torch.exp(torch.cumsum(torch.log(self.alpha_t), dim=0))
        self.bar_alpha_t = torch.cumprod(self.alpha_t, dim=0)
        # self.use_fp16 = use_fp16
        self.config = config

    def add_noise(self, clean_images):
        shape = clean_images.shape
        expand = torch.ones(len(shape)-1, dtype=int)
        # ts_expand = ts.view(ts.shape[0], *expand.tolist())
        # expand = [1 for i in range(len(shape)-1)]

        noise = torch.randn_like(clean_images).to(self.device)
        ts = torch.randint(0, self.num_timesteps, (shape[0],)).to(self.device)
                
        # test_expand = test.view(test.shape[0],*expand)
        # extend_dim = [None for i in range(shape.dim()-1)]
        noisy_images = (
            clean_images * torch.sqrt(self.bar_alpha_t[ts]).view(shape[0], *expand.tolist())
            + noise * torch.sqrt(1-self.bar_alpha_t[ts]).view(shape[0], *expand.tolist())
            )
        # print(x_t.shape)

        return noisy_images, noise, ts

    def sample(self, nn_model, params, device, guide_w = 0):
        n_sample = len(params) #params.shape[0]
        # print("params.shape[0], len(params)", params.shape[0], len(params))
        x_i = torch.randn(n_sample, *self.img_shape)#.to(self.dtype)
        x_i = x_i.to(device)
        #print(f"#1 x_i.device = {x_i.device}")
        # print("x_i.shape =", x_i.shape)
        # print("x_i.shape =", x_i.shape)
        if guide_w != -1:
            c_i = params
            #uncond_tokens = torch.zeros(int(n_sample), params.shape[1]).to(device)
            # uncond_tokens = torch.tensor(np.float32(np.array([0,0]))).to(device)
            # uncond_tokens = uncond_tokens.repeat(int(n_sample),1)
            #c_i = torch.cat((c_i, uncond_tokens), 0)
            #c_i = c_i.to(self.dtype)

        x_i_entire = [] # keep track of generated steps in case want to plot something
        # print("self.num_timesteps =", self.num_timesteps)
        # for i in range(self.num_timesteps, 0, -1):
        # print(f'sampling!!!')
        pbar_sample = tqdm(total=self.num_timesteps, file=sys.stderr, disable=True)
        pbar_sample.set_description(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} sampling")
        for i in reversed(range(0, self.num_timesteps)):
            # print(f'sampling timestep {i:4d}',end='\r')
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample)
            #t_is = t_is.to(self.dtype)

            z = torch.randn(n_sample, *self.img_shape).to(device) if i > 0 else torch.tensor(0.)
            #z = z.to(self.dtype)

            if guide_w == -1:
                # eps = nn_model(x_i, t_is, return_dict=False)[0]
                eps = nn_model(x_i, t_is)#.to(self.dtype)
                # x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            else:
                # double batch
                #print(f"#2 x_i.device = {x_i.device}")
                #x_i = x_i.repeat(2, *torch.ones(len(self.img_shape), dtype=int).tolist())
                #t_is = t_is.repeat(2)

                # split predictions and compute weighting
                # print("nn_model input shape", x_i.shape, t_is.shape, c_i.shape)
                #print(f"sample, i = {i}, x_i.dtype = {x_i.dtype}, c_i.dtype = {c_i.dtype}")
                eps = nn_model(x_i, t_is, c_i)#.to(self.dtype)
                #eps1 = eps[:n_sample]
                #eps2 = eps[n_sample:]
                #eps = eps1 + guide_w*(eps1 - eps2)
                # eps = (1+guide_w)*eps1 - guide_w*eps2
                #x_i = x_i[:n_sample]
                # x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            
            # print("x_i.shape =", x_i.shape)
            #print(f"before, x_i.dtype = {x_i.dtype}, beta_t.dtype = {self.beta_t.dtype}, eps.dtype = {eps.dtype}, alpha_t.dtype = {self.alpha_t.dtype}, z.dtype = {z.dtype}")
            x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            #print(f"after, x_i.dtype = {x_i.dtype}, beta_t.dtype = {self.beta_t.dtype}, eps.dtype = {eps.dtype}, alpha_t.dtype = {self.alpha_t.dtype}, z.dtype = {z.dtype}")

            pbar_sample.update(1)
            
            # store only part of the intermediate steps
            # if i%20==0:# or i==0:# or i<8:
            #     x_i_entire.append(x_i.detach().cpu().numpy())
        x_i_entire = np.array(x_i_entire)
        x_i = x_i.detach().cpu().numpy()
        return x_i, x_i_entire


# ddpm_scheduler = DDPMScheduler((1e-4,0.02),10)
# noisy_images, noise, ts = ddpm_scheduler.add_noise(images)

# %%
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
    startat = 512 #-num_redshift

    channel = 1
    img_shape = (channel, HII_DIM, num_redshift) if dim == 2 else (channel, HII_DIM, HII_DIM, num_redshift)

    ranges_dict = dict(
        params = {
            0: [4, 6], # ION_Tvir_MIN
            1: [10, 250], # HII_EFF_FACTOR
            },
        images = {
            # 0: [-338, 54],#[0, 80], # brightness_temp
            0: [-387, 86],
            }
        )

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
    output_dir = "./outputs/"
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
    str_len = 140
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

        #gpu_info = get_gpu_info(config.device)
        if config.resume and os.path.exists(config.resume):
            # resume_file = os.path.join(config.output_dir, f"{config.resume}")
            # self.nn_model.load_state_dict(torch.load(config.resume)['unet_state_dict'])
            # print(f"resumed nn_model from {config.resume}")
            self.nn_model.module.load_state_dict(torch.load(config.resume)['unet_state_dict'])
            #self.nn_model.module.to(config.dtype)
            print(f"📝 {config.run_name} cuda:{torch.cuda.current_device()}/{self.config.global_rank} resumed nn_model from {config.resume} with {sum(x.numel() for x in self.nn_model.module.parameters())} parameters, {datetime.now().strftime('%d-%H:%M:%S.%f')} 📝".center(self.config.str_len,'+'))
        else:
            print(f"🚀 {config.run_name} cuda:{torch.cuda.current_device()}/{self.config.global_rank} initialized nn_model randomly with {sum(x.numel() for x in self.nn_model.module.parameters())} parameters, {datetime.now().strftime('%d-%H:%M:%S.%f')} 🚀".center(self.config.str_len,'+'))

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
            #self.ema_model = DDP(self.ema_model, device_ids=[self.ddpm.device], find_unused_parameters=True)

            if config.resume and os.path.exists(config.resume):
                #self.ema_model = ContextUnet(n_param=config.n_param, image_size=config.HII_DIM, dim=config.dim, stride=config.stride).to(config.device, dropout=config.dropout)#, dtype=config.dtype
                self.ema_model.load_state_dict(torch.load(config.resume)['ema_unet_state_dict'])
                print(f"{config.run_name} cuda:{torch.cuda.current_device()}/{self.config.global_rank} resumed ema_model from {config.resume} with {sum(x.numel() for x in self.ema_model.parameters())} parameters, {datetime.now().strftime('%d-%H:%M:%S.%f')}".center(self.config.str_len,'+'))
                #print(f"resumed ema_model from {config.resume}")
            else:
                self.ema_model = copy.deepcopy(self.nn_model.module).eval().requires_grad_(False)
                print(f"{config.run_name} cuda:{torch.cuda.current_device()}/{self.config.global_rank} initialized ema_model randomly with {sum(x.numel() for x in self.ema_model.parameters())} parameters, {datetime.now().strftime('%d-%H:%M:%S.%f')}".center(self.config.str_len,'+'))


        self.optimizer = torch.optim.AdamW(self.nn_model.module.parameters(), lr=config.lrate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer = self.optimizer,
                T_max = int(config.num_image / config.batch_size * config.n_epoch / config.gradient_accumulation_steps),
                )

        self.ranges_dict = config.ranges_dict
        self.scaler = GradScaler()

    def load(self):
        dataset = Dataset4h5(
            self.config.dataset_name, 
            num_image=self.config.num_image,
            idx = 'range',#"random",#
            HII_DIM=self.config.HII_DIM, 
            num_redshift=self.config.num_redshift,
            startat=self.config.startat,
            #drop_prob=self.config.drop_prob, 
            dim=self.config.dim,
            ranges_dict=self.ranges_dict,
            num_workers=min(1,len(os.sched_getaffinity(0))//self.config.world_size),
            str_len = self.config.str_len,
            )
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank}: Dataset4h5 done")

        dataloader_start = time()
        self.dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,#False, 
            num_workers=len(os.sched_getaffinity(0))//self.config.world_size,
            pin_memory=True,
            persistent_workers=True,
            # sampler=DistributedSampler(dataset),
            )
        if len(self.dataloader) % self.config.gradient_accumulation_steps != 0:
            raise ValueError(f"len(self.dataloader) % self.config.gradient_accumulation_steps = {len(self.dataloader) % self.config.gradient_accumulation_steps} instead of 0. Make sure len(dataloader)={len(self.dataloader)} is dividable by gradient_accumulation_steps={self.config.gradient_accumulation_steps}.")

        dataloader_end = time()
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} dataloader costs {dataloader_end-dataloader_start:.3f}s")

        del dataset

    def transform(self, img, idx):
        #flip along x or y or both
        flip_xy = [i+1 for i in range(2) if getrandbits(1)]
        img[idx] = torch.flip(img[idx], dims=flip_xy) 
        # flip diagonally 
        if getrandbits(1):
            #img = img.transpose(2,3)
            img[idx] = img[idx].clone().transpose(1,2)
            #print(f"transform: img.shape={img.shape}, idx={idx}, flip_xy={flip_xy}, w/ transpose")
        #else:
            #print(f"transform: img.shape={img.shape}, idx={idx}, flip_xy={flip_xy}, w/o tranpose")
        return img

    def train(self):
        ###################      
        ## training loop ##
        ###################
        # plot_unet = True

        self.load()
        #self.accelerator = Accelerator(
        #    mixed_precision=self.config.mixed_precision,
        #    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        #    log_with="tensorboard",
        #    project_dir=os.path.join(self.config.output_dir, "logs"),
            # distributed_type="MULTI_GPU",
        #)
        # print("!!!!!!!!!!!!!!!!!!!self.accelerator.device:", self.accelerator.device)
        # if self.accelerator.is_main_process:
        if self.config.global_rank == 0: # or torch.cuda.current_device() == 0:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            if self.config.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.config.hub_model_id or Path(self.config.output_dir).name, exist_ok=True
                ).repo_id
            #self.accelerator.init_trackers(f"{self.config.run_name}")
            self.config.logger = SummaryWriter(f"logs/{self.config.run_name}") 

        # print("!!!!!!!!!!!!!!!!, before prepare, self.dataloader.sampler =", self.dataloader.sampler)
        #model_start = time()
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} model: {self.nn_model.device}", f"{time()-model_start:.3f}s")
        #print(f"optimizer: {self.optimizer.state_dict()}")
        #dataloader_start = time()
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} dataloader: {next(iter(self.dataloader))[0].device}", f"{time()-dataloader_start:.3f}s")
        #lr_start = time()
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} lr_scheduler: {self.lr_scheduler.optimizer is self.optimizer}", f"{time()-lr_start:.3f}s")
        #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} print costs {print_end-print_start:.3f}s")
        if torch.distributed.is_initialized():
            #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} torch.distributed.is_initialized")
            torch.distributed.barrier()
        else:
            print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} torch.distributed.is_initialized False!!!!!!!!!!!!!!!") 

        global_step = 0
        for ep in range(self.config.n_epoch):
            self.ddpm.train()
            pbar_train = tqdm(total=len(self.dataloader), file=sys.stderr, disable=True)#, mininterval=self.config.pbar_update_step)#, disable=True)#not self.accelerator.is_local_main_process)
            pbar_train.set_description(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.config.global_rank} Epoch {ep}")
            epoch_start = time()
            #print(f"🚀 ep={ep}")
            for i, (x, c) in enumerate(self.dataloader):
                #print(f"i={i}")
                if self.config.dim == 3:
                    #x = self.transform(x)
                    for idx in range(len(x)):
                        x = self.transform(x, idx)

                x = x.to(self.config.device)#.to(self.config.dtype)
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
                    loss = loss / self.config.gradient_accumulation_steps

                    #print(f"ep = {ep}, loss = {loss}")
                    if torch.isnan(loss).any():
                        raise ValueError(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.config.global_rank} Epoch {ep}, loss: {loss}")

                #if self.config.global_rank == 0:
                #    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                #    print(result.stdout)#, flush=True)

                #del noise, noise_pred
                #torch.cuda.empty_cache()

                #if self.config.global_rank == 0:
                #    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                #    print(result.stdout)#, flush=True)

                # scaler backward propogation
                self.scaler.scale(loss).backward()
                #loss.backward()
                #print(f"ep = {ep}, after backward")

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
                    step=global_step
                )
                pbar_train.set_postfix(**logs)

                #self.accelerator.log(logs, step=global_step)
                if self.config.global_rank == 0:
                    self.config.logger.add_scalar("MSE", logs["loss"], global_step = global_step)
                    self.config.logger.add_scalar("learning_rate", logs["lr"], global_step = global_step)
                global_step += 1

            if (i+1) % self.config.gradient_accumulation_steps != 0:
                print(f"(i+1)%self.config.gradient_accumulation_steps = {(i+1)%self.config.gradient_accumulation_steps}, i = {i}, scg = {self.config.gradient_accumulation_steps}".center(self.config.str_len,'-'))
            # if ep == config.n_epoch-1 or (ep+1)*config.save_period==1:
            self.save(ep)
            print(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.config.global_rank} Epoch{ep}:{i+1}/{len(self.dataloader)} costs {(time()-epoch_start)/60:.2f} min", flush=True)

        #print(f"🆘 global_rank = {self.config.global_rank}, after save(ep) 🆘", flush=True)
        if dist.is_initialized():
            dist.barrier()

        #print(f"🆘 global_rank = {self.config.global_rank}, before del self.nn_model 🆘", flush=True)
        del self.nn_model
        #print(f"🆘 global_rank = {self.config.global_rank}, after del self.nn_model 🆘", flush=True)

        if self.config.ema:
            del self.ema_model

    def save(self, ep):
        # save model
        # if self.accelerator.is_main_process:
        if ep == self.config.n_epoch-1 or (ep+1) % self.config.save_period == 0:
            #save_name = self.config.save_name+f"-N{self.config.num_image}-device_count{self.config.world_size}-node{int(os.environ['SLURM_NNODES'])}-epoch{ep}-{self.config.run_name}"
            #global config_resume
            #config_resume = save_name
            #print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank}", "save_name copied to config_resume =", config_resume)

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
                            'unet_state_dict': self.nn_model.module.state_dict(),
                            'ema_unet_state_dict': self.ema_model.state_dict() if self.config.ema else None,
                            }
                        save_name = self.config.save_name + f"-epoch{ep+1}"
                        torch.save(model_state, save_name)
                        print(f'🌟 cuda:{torch.cuda.current_device()}/{self.config.global_rank} saved model at ' + save_name)
                        # print('saved model at ' + config.save_dir + f"model_epoch_{ep}_test_{config.run_name}.pth")

    # def rescale(self, value, type='params', to_ranges=[0,1]):
    #     for i, from_ranges in self.ranges_dict[type].items():
    #         value[i] = (value[i] - from_ranges[0])/(from_ranges[1]-from_ranges[0]) # normalize
    #         value[i] = 
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
            # params_backup = params.numpy().copy()
        # else:
        params_backup = params.numpy().copy()
        params_normalized = self.rescale(params, self.ranges_dict['params'], to=[0,1])

        print(f"{socket.gethostbyname(socket.gethostname())} cuda:{torch.cuda.current_device()}/{self.config.global_rank} sampling {num_new_img_per_gpu} images with normalized params = {params_normalized}, {datetime.now().strftime('%d-%H:%M:%S.%f')}")
        params_normalized = params_normalized.repeat(num_new_img_per_gpu,1)
        assert params_normalized.dim() == 2, "params_normalized must be a 2D torch.tensor"
        # print("params =", params)

        self.nn_model.module.eval()
        if self.config.ema:
            self.ema_model.eval()

        sample_start = time()
        with torch.no_grad():
            with autocast(enabled=self.config.autocast):
            #with autocast():
                x_last, x_entire = self.ddpm.sample(
                    nn_model=self.nn_model.module, 
                    params=params_normalized.to(self.config.device), 
                    device=self.config.device, 
                    guide_w=self.config.guide_w
                    )
                if self.config.ema:
                    x_last_ema, x_entire_ema = self.ddpm.sample(
                        nn_model=self.ema_model, 
                        params=params_normalized.to(self.config.device), 
                        device=self.config.device, 
                        guide_w=self.config.guide_w
                        )
        #print(f"x_last.dtype = {x_last.dtype}")
        if save:    
            # np.save(os.path.join(self.config.output_dir, f"{self.config.run_name}{'ema' if ema else ''}.npy"), x_last)
            savetime = datetime.now().strftime("%d%H%M%S")
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

            for ema in range(self.config.ema + 1):
                elapsed_time = (time()-sample_start)/(self.config.ema + 1)
                savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]:.3f}-zeta{params_backup[1]:.3f}-device{self.config.global_rank}-{os.path.basename(self.config.resume)}-{savetime}-ema{ema}")
                np.save(savename, x_last if ema==0 else x_last_ema)
                print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} saved {x_last.shape} to {os.path.basename(savename)} with {elapsed_time/60:.2f} min", flush=True)

                if entire:
                    #savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]:.3f}-zeta{params_backup[1]:.3f}-device{self.config.global_rank}-{os.path.basename(self.config.resume)}-{savetime}-ema{ema}_entire")
                    savename += '_entire'
                    np.save(savename, x_entire)
                    print(f"cuda:{torch.cuda.current_device()}/{self.config.global_rank} saved images of shape {x_entire.shape} to {savename}")
        # else:
        #return x_last
# %%

#num_train_image_list = [6000]#[60]#[8000]#[1000]#[100]#
def train(rank, world_size, local_world_size, master_addr, master_port, config):
    global_rank = rank + local_world_size * int(os.environ["SLURM_NODEID"])
    ddp_setup(global_rank, world_size, master_addr, master_port)
    torch.cuda.set_device(rank)
    #print(f"rank = {rank}, global_rank = {global_rank}, world_size = {world_size}, local_world_size = {local_world_size}")

    #config = TrainConfig()
    config.device = f"cuda:{rank}"
    config.world_size = local_world_size
    config.global_rank = global_rank 
    #print("before dppm21cm")
    ddpm21cm = DDPM21CM(config)
    ddpm21cm.train()
    print(f"🆘 global_rank = {global_rank}, ddpm21cm.train is over 🆘", flush=True)

    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
        print(f"🆘 global_rank = {global_rank}, destroy_process_group starts 🆘", flush=True)
        dist.destroy_process_group()
    print(f"🆘 global_rank = {global_rank}, destroy_process_group is over 🆘", flush=True)


def generate_samples(rank, world_size, local_world_size, master_addr, master_port, config, num_new_img_per_gpu, max_num_img_per_gpu, params_pairs):
    global_rank = rank + local_world_size * int(os.environ["SLURM_NODEID"])
    ddp_setup(global_rank, world_size, master_addr, master_port)
    torch.cuda.set_device(rank)

    config.device = f"cuda:{rank}"
    config.world_size = local_world_size
    config.global_rank = global_rank

    ddpm21cm = DDPM21CM(config)

    for params in params_pairs:
        if global_rank == 0:
            print(f"⛳️ sampling, {params}, ip = {socket.gethostbyname(socket.gethostname())}, local_world_size = {local_world_size}, world_size = {world_size}, {datetime.now().strftime('%d-%H:%M:%S.%f')} ⛳️".center(config.str_len,'#'), flush=True)

        for _ in range(num_new_img_per_gpu // max_num_img_per_gpu):    
            #print(f"rank = {rank}, global_rank = {global_rank}, world_size = {world_size}, local_world_size = {local_world_size}")
            ddpm21cm.sample(
                params=params, 
                num_new_img_per_gpu=max_num_img_per_gpu,
                )
                
        if num_new_img_per_gpu % max_num_img_per_gpu:
            ddpm21cm.sample(
                params=params, 
                num_new_img_per_gpu=num_new_img_per_gpu % max_num_img_per_gpu,
                )

    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=False, help="whether to train the model", default=False)
    parser.add_argument("--sample", type=int, required=False, help="whether to sample", default=0)
    parser.add_argument("--resume", type=str, required=False, help="filename of the model to resume", default=False)
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
    parser.add_argument("--lrate", type=float, required=False, default=1e-4)
    parser.add_argument("--dim", type=int, required=False, default=3)
    parser.add_argument("--num_redshift", type=int, required=False, default=64)
    parser.add_argument("--num_res_blocks", type=int, required=False, default=3)
    parser.add_argument("--model_channels", type=int, required=False, default=96)
    parser.add_argument("--stride", type=int, nargs="+", required=False, default=(2,2,1))
    parser.add_argument("--guide_w", type=int, required=False, default=0)
    parser.add_argument("--ema", type=int, required=False, default=0)

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
    config.guide_w = args.guide_w
    config.ema = args.ema
    #config.sample = args.sample

    config.stride = args.stride #(2,2) if config.dim == 2 else (2,2,1)
    config.dim = len(config.stride) #args.dim
    #print(config.stride, config.dim)
    config.num_redshift = args.num_redshift
    config.img_shape = (config.channel, config.HII_DIM, config.HII_DIM) if config.dim == 2 else (config.channel, config.HII_DIM, config.HII_DIM, config.num_redshift)
    config.num_res_blocks = args.num_res_blocks

    config.run_name = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%d%H%M%S")) # the unique name of each experiment
    config.save_name += f"-N{config.num_image}-device_count{local_world_size}-node{total_nodes}-{config.run_name}"
    
    #print("before args.train, config.resume =", config.resume)
    ############################ training ################################
    if args.train:
        config.dataset_name = args.train
        print(f"⛳️ training, ip = {socket.gethostbyname(socket.gethostname())}, local_world_size = {local_world_size}, world_size = {world_size}, {datetime.now().strftime('%d-%H:%M:%S.%f')} ⛳️".center(config.str_len,'#'))
        mp.spawn(
                train, 
                args=(world_size, local_world_size, master_addr, master_port, config), 
                nprocs=local_world_size, 
                join=True,
                )
        #print(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
        config.resume = config.save_name + f"-epoch{config.n_epoch}"
        #print("in args.train, config.resume =", config.resume)

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

        #for params in params_pairs:
        mp.spawn(
                generate_samples, 
                args=(world_size, local_world_size, master_addr, master_port, config, num_new_img_per_gpu, max_num_img_per_gpu, torch.tensor(params_pairs)), 
                nprocs=local_world_size, 
                join=True,
                )


