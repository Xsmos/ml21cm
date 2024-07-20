# %% [markdown]
# ## 改編ContextUnet及相關代碼，使其首先對二維的情況適用。並於diffusers.Unet2DModel作比較並加以優化。最後再改寫爲3維的情形。
# - 經試用diffusers的Unet2DModel，發現loss從0.3降到0.2但仍然很高，説明存在非Unet2DModel的問題可以優化
# - 改用diffusers的DDMPScheduler和DDPMPipeline后，loss降低至0.1以下，有時甚至可以低至0.004，可見我的代碼問題主要出在DDPM部分。DDPMScheduler部分比較簡短，似乎沒有問題，所以問題應該在DDPMPipeline裏某一部分代碼是我代碼欠缺的。
# - 我在DDPMScheduler部分有一個typo，導致beta_t一直很小，修正后loss從0.2能降低至0.02, 維持在0.1以下
# - 用diffusers的DDPMScheduler似乎效果要好一些，loss總是比我的DDPMScheduler要小一點。儅epoch為19時，前者的loss約0.02，後者loss約0.07。而且前者還支持3維圖像的加噪，不如直接用別人的輪子。但我想知道爲什麽我的loss會高一些。
# - 我意識到別人的DDPMScheduler在sample函數中沒有兼容輸入參數，所以歸根結底還是需要我的DDPMscheduler。不過我可以先用別人的來debug我的ContextUnet.
# - 我需要將我的ContextUnet擴展兼容不同維度的照片，畢竟我本身也需要和原文獻對比完了再拓展到三維的情形
# - 我已將我的ContextUnet轉成了2維的模式，與diffusers.Unet2DModel的loss=0.037相比，我的Unet的loss=0.07。同時我的Unet生成的圖像看上去很奇怪，説明我的Unet也有問題。我需要將代碼退回原Unet，並檢查問題所在。
# - 我將紅移方向的像素的數量限制在了64.以此比較兩個Unet的差別。經比較：\
# Unet2DModel loss：0.03, 0.0655, 0.05, 0.02, 0.05\
# ContextUnet loss: 0.1, 0.16, 0.1, 0.2186, 0.06
# - 我把ContextUnet退回到了原作者的版本，結果loss=0.05，輸出的照片也不錯。我主要的改動是改回了他原用的normalization函數，其中還有個參數swish。有時間我可以研究一下具體是哪裏影響了訓練的結果。另外我發現了要想tensorboard的圖綫獨立美觀，需要把他們放在不同的文件夾下
# - 經過驗證，GroupNorm比batchNorm效果要好
# - 已擴展爲接受不同維度的情形
# - 融合cond, guide_w, drop_out這些參數
# - 生成的21cm圖像該暗的地方不夠暗，似乎換成MNIST的數字圖像就沒問題
# - 我用diffusion模型生成MNIST的數字時發現，儘管生成的數據的範圍也存在負數數值，如-0.1,但畫出來的圖像卻是理想的黑色。數據的分佈與21cm的結果的分佈沒多大差別，我現在打算把代碼退回到21cm的情形
# - 我統一了ddpm21cm這個module，能統一實現訓練和生成樣本，但目前有個bug， sample時總是會cuda out of memory，然而單獨resume model並sample就不會。
# - 解決了，問題出在我忘了寫with torch.no_grad():
# - 接下來就是生成800個lightcones，與此同時研究如何計算global signal以及power spectrum
# - 儅訓練圖片的數量達到5000時，生成的圖片與檢測數據的相似程度很高
# - it takes 62 mins to generated 8 images with shape of (64,64,64), which is even slower than simulation, which takes ~5 mins for each image. Besides, the batch_size during training and num of images to be generated are limited to be 2 and 8, respectively.
# - the slowerness can be solved by using multi-GPUs, and the limited-num-of-images can be solved by multi-accuracy, multi-GPUs.
# - In addtion, the performance of DDPM can looks better compared to computation-intensive simulations. 
# 1 GPU, batch_size = 10, num_image = 3200, 50s for each epoch
# 4 GPU, batch_size = 10, num_image = 3200, 

# %%
from dataclasses import dataclass
import h5py
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
# from torchvision import transforms
# from diffusers import UNet2DModel#, UNet3DConditionModel
# from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
import datetime
from pathlib import Path
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher, Accelerator
from huggingface_hub import create_repo, upload_folder

from load_h5 import Dataset4h5
from context_unet import ContextUnet

from huggingface_hub import notebook_login

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# %%
def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
#   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ddp_setup, rank =", rank)
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

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
    def __init__(self, betas: tuple, num_timesteps: int, img_shape: list, device='cpu', dtype=torch.float32):
        super().__init__()
        
        beta_1, beta_T = betas
        assert 0 < beta_1 <= beta_T <= 1, "ensure 0 < beta_1 <= beta_T <= 1"
        self.device = device
        self.num_timesteps = num_timesteps
        self.img_shape = img_shape
        self.beta_t = torch.linspace(beta_1, beta_T, self.num_timesteps) #* (beta_T-beta_1) + beta_1
        self.beta_t = self.beta_t.to(self.device)

        # self.drop_prob = drop_prob
        # self.cond = cond
        self.alpha_t = 1 - self.beta_t
        # self.bar_alpha_t = torch.exp(torch.cumsum(torch.log(self.alpha_t), dim=0))
        self.bar_alpha_t = torch.cumprod(self.alpha_t, dim=0)
        # self.use_fp16 = use_fp16
        self.dtype = dtype#torch.float16 if self.use_fp16 else torch.float32

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
        x_i = torch.randn(n_sample, *self.img_shape).to(device)
        # print("x_i.shape =", x_i.shape)
        # print("x_i.shape =", x_i.shape)
        if guide_w != -1:
            c_i = params
            uncond_tokens = torch.zeros(int(n_sample), params.shape[1]).to(device)
            # uncond_tokens = torch.tensor(np.float32(np.array([0,0]))).to(device)
            # uncond_tokens = uncond_tokens.repeat(int(n_sample),1)
            c_i = torch.cat((c_i, uncond_tokens), 0)

        x_i_entire = [] # keep track of generated steps in case want to plot something
        # print("self.num_timesteps =", self.num_timesteps)
        # for i in range(self.num_timesteps, 0, -1):
        # print(f'sampling!!!')
        pbar_sample = tqdm(total=self.num_timesteps)
        pbar_sample.set_description(f"device {torch.cuda.current_device()} sampling")
        for i in reversed(range(0, self.num_timesteps)):
            # print(f'sampling timestep {i:4d}',end='\r')
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample)

            z = torch.randn(n_sample, *self.img_shape).to(device) if i > 0 else 0

            if guide_w == -1:
                # eps = nn_model(x_i, t_is, return_dict=False)[0]
                eps = nn_model(x_i, t_is)
                # x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            else:
                # double batch
                x_i = x_i.repeat(2, *torch.ones(len(self.img_shape), dtype=int).tolist())
                t_is = t_is.repeat(2)

                # split predictions and compute weighting
                # print("nn_model input shape", x_i.shape, t_is.shape, c_i.shape)
                eps = nn_model(x_i, t_is, c_i)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = eps1 + guide_w*(eps1 - eps2)
                # eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:n_sample]
                # x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            
            # print("x_i.shape =", x_i.shape)
            x_i = 1/torch.sqrt(self.alpha_t[i])*(x_i-eps*self.beta_t[i]/torch.sqrt(1-self.bar_alpha_t[i])) + torch.sqrt(self.beta_t[i])*z
            
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
    push_to_hub = True 
    hub_model_id = "Xsmos/ml21cm"
    hub_private_repo = False
    dataset_name = "/storage/home/hcoda1/3/bxia34/scratch/LEN128-DIM64-CUB8.h5"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    world_size = torch.cuda.device_count()
    # repeat = 2

    # dim = 2
    dim = 2
    stride = (2,2) if dim == 2 else (2,2,2)
    num_image = 1000#2000#20000#15000#7000#25600#3000#10000#1000#10000#5000#2560#800#2560
    batch_size = 50#1#2#50#20#2#100 # 10
    n_epoch = 50#30#120#5#4# 10#50#20#20#2#5#25 # 120
    HII_DIM = 64
    num_redshift = 64#512#128#64#512#256#256#64#512#128
    channel = 1
    img_shape = (channel, HII_DIM, num_redshift) if dim == 2 else (channel, HII_DIM, HII_DIM, num_redshift)

    ranges_dict = dict(
        params = {
            0: [4, 6], # ION_Tvir_MIN
            1: [10, 250], # HII_EFF_FACTOR
            },
        images = {
            0: [0, 80], # brightness_temp
            }
        )

    num_timesteps = 1000#1000 # 1000, 500; DDPM time steps
    # n_sample = 24 # 64, the number of samples in sampling process
    n_param = 2
    guide_w = 0#-1#0#-1#0#-1#0.1#[0,0.1] #[0,0.5,2] strength of generative guidance
    drop_prob = 0#0.28 # only takes effect when guide_w != -1
    ema=False # whether to use ema
    ema_rate=0.995

    # seed = 0
    # save_dir = './outputs/'

    save_period = np.infty#.1 # the period of sampling
    # general parameters for the name and logger    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    lrate = 1e-4
    lr_warmup_steps = 0#5#00
    output_dir = "./outputs/"
    save_name = os.path.join(output_dir, 'model_state')
    # save_period = 1 #10 # the period of saving model
    # cond = True # if training using the conditional information
    # lr_decay = False #True# if using the learning rate decay
    resume = False # if resume from the trained checkpoints
    # params_single = torch.tensor([0.2,0.80000023])
    # params = torch.tile(params_single,(n_sample,1)).to(device)
    # params =  params
    # data_dir = './data' # data directory

    use_fp16 = False
    dtype = torch.float16 if use_fp16 else torch.float32
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1

    # date = datetime.datetime.now().strftime("%m%d-%H%M")
    # run_name = f'{date}' # the unique name of each experiment

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
class DDPM21CM:
    def __init__(self, config):
        # print(
        #     "torch.cuda.is_available() =", torch.cuda.is_available(), 
        #     "torch.cuda.device_count() =", torch.cuda.device_count(),
        #     "torch.cuda.is_initialized() =", torch.cuda.is_initialized(),
        #     "torch.cuda.current_device() =", torch.cuda.current_device()
        # )
        # config = TrainConfig()
        # date = datetime.datetime.now().strftime("%m%d-%H%M")
        config.run_name = datetime.datetime.now().strftime("%m%d-%H%M") # the unique name of each experiment
        self.config = config
        # dataset = Dataset4h5(config.dataset_name, num_image=config.num_image, HII_DIM=config.HII_DIM, num_redshift=config.num_redshift, drop_prob=config.drop_prob, dim=config.dim)
        # # self.shape_loaded = dataset.images.shape
        # # print("shape_loaded =", self.shape_loaded)
        # self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        # del dataset
        self.ddpm = DDPMScheduler(betas=(1e-4, 0.02), num_timesteps=config.num_timesteps, img_shape=config.img_shape, device=config.device, dtype=config.dtype)

        # initialize the unet
        self.nn_model = ContextUnet(n_param=config.n_param, image_size=config.HII_DIM, dim=config.dim, stride=config.stride, dtype=config.dtype)

        # nn_model = ContextUnet(n_param=1, image_size=28)
        self.nn_model.train()
        # print("self.ddpm.device =", self.ddpm.device)
        self.nn_model.to(self.ddpm.device)
        self.nn_model = DDP(self.nn_model, device_ids=[self.ddpm.device])
        # print("nn_model.device =", ddpm.device)
        # number of parameters to be trained

        if config.resume and os.path.exists(config.resume):
            # resume_file = os.path.join(config.output_dir, f"{config.resume}")
            # self.nn_model.load_state_dict(torch.load(config.resume)['unet_state_dict'])
            # print(f"resumed nn_model from {config.resume}")
            self.nn_model.module.load_state_dict(torch.load(config.resume)['unet_state_dict'])
            print(f"device {torch.cuda.current_device()} resumed nn_model from {config.resume}")

        self.number_of_params = sum(x.numel() for x in self.nn_model.parameters())
        print(f" Number of parameters for nn_model: {self.number_of_params} ".center(100,'-'))

        # whether to use ema
        if config.ema:
            self.ema = EMA(config.ema_rate)
            if config.resume and os.path.exists(config.resume):
                self.ema_model = ContextUnet(n_param=config.n_param, image_size=config.HII_DIM, dim=config.dim, stride=config.stride, dtype=config.dtype).to(config.device)
                self.ema_model.load_state_dict(torch.load(config.resume)['ema_unet_state_dict'])
                print(f"resumed ema_model from {config.resume}")
            else:
                self.ema_model = copy.deepcopy(self.nn_model).eval().requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=config.lrate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=int(config.num_image / config.world_size / config.batch_size * config.n_epoch),
            # num_training_steps=(len(self.dataloader) * config.n_epoch),
        )

        self.ranges_dict = config.ranges_dict

    def load(self):
        dataset = Dataset4h5(self.config.dataset_name, num_image=self.config.num_image, HII_DIM=self.config.HII_DIM, num_redshift=self.config.num_redshift, drop_prob=self.config.drop_prob, dim=self.config.dim, ranges_dict=self.ranges_dict)
        # self.shape_loaded = dataset.images.shape
        # print("shape_loaded =", self.shape_loaded)
        # print(f"load, current_device() = {torch.cuda.current_device()}")
        self.dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=1,#len(os.sched_getaffinity(0)), 
            pin_memory=True,
            persistent_workers=True,
            sampler=DistributedSampler(dataset),
            )

        del dataset
        # self.accelerate(self.config)
        # print("!!!!!!!!!!!!!!!!, self.dataloader.sampler =", self.dataloader.sampler)
        # del dataset

    # def accelerate(self):

    def train(self):
        ###################      
        ## training loop ##
        ###################
        # plot_unet = True

        self.load()
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.config.output_dir, "logs"),
            # distributed_type="MULTI_GPU",
        )
        # print("!!!!!!!!!!!!!!!!!!!self.accelerator.device:", self.accelerator.device)
        # if self.accelerator.is_main_process:
        if torch.cuda.current_device() == 0:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            if self.config.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.config.hub_model_id or Path(self.config.output_dir).name, exist_ok=True
                ).repo_id
            self.accelerator.init_trackers(f"{self.config.run_name}")


        # print("!!!!!!!!!!!!!!!!, before prepare, self.dataloader.sampler =", self.dataloader.sampler)
        # self.nn_model, self.optimizer, self.dataloader, self.lr_scheduler = \
        #     self.accelerator.prepare(
        #     self.nn_model, self.optimizer, self.dataloader, self.lr_scheduler
        #     )
        self.nn_model, self.optimizer, self.lr_scheduler = \
            self.accelerator.prepare(
            self.nn_model, self.optimizer, self.lr_scheduler
            )

        # print("!!!!!!!!!!!!!!!!, after prepare, self.dataloader.sampler =", self.dataloader.sampler)
        # print("!!!!!!!!!!!!!!!!, after prepare, self.dataloader.batch_sampler =", self.dataloader.batch_sampler)
        # print("!!!!!!!!!!!!!!!!, after prepare, self.dataloader.DistributedSampler =", self.dataloader.DistributedSampler)

        global_step = 0
        for ep in range(self.config.n_epoch):
            self.ddpm.train()
            self.dataloader.sampler.set_epoch(ep)

            pbar_train = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process)
            pbar_train.set_description(f"device {torch.cuda.current_device()}, Epoch {ep}")
            for i, (x, c) in enumerate(self.dataloader):
                # print(f"device {torch.cuda.current_device()}, x[:,0,:2,0,0] =", x[:,0,:2,0,0])
                with self.accelerator.accumulate(self.nn_model):
                    x = x.to(self.config.device)
                    # print("x = x.to(self.config.device), x.dtype =", x.dtype)
                    # x = x.to(self.config.dtype)
                    # print("x = x.to(self.dtype), x.dtype =", x.dtype)
                    xt, noise, ts = self.ddpm.add_noise(x)
                    
                    if self.config.guide_w == -1:
                        noise_pred = self.nn_model(xt, ts)
                    else:
                        c = c.to(self.config.device)
                        noise_pred = self.nn_model(xt, ts, c)

                    # print("noise_pred = self.nn_model(xt, ts, c), noise_pred.dtype =", noise_pred.dtype)
                    
                    loss = F.mse_loss(noise, noise_pred)
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.nn_model.parameters(), 1)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # ema update
                if self.config.ema:
                    self.ema.step_ema(self.ema_model, self.nn_model)

                pbar_train.update(1)
                logs = dict(
                    loss=loss.detach().item(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    step=global_step
                )
                pbar_train.set_postfix(**logs)

                self.accelerator.log(logs, step=global_step)
                global_step += 1

            # if ep == config.n_epoch-1 or (ep+1)*config.save_period==1:
            self.save(ep)

        del self.nn_model
        if self.config.ema:
            del self.ema_model
        torch.cuda.empty_cache()

    def save(self, ep):
        # save model
        # if self.accelerator.is_main_process:
        if torch.cuda.current_device() == 0:
            if ep == self.config.n_epoch-1 or (ep+1) % self.config.save_period == 0:
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
                            # 'ema_unet_state_dict': self.ema_model.state_dict(),
                            }
                        save_name = self.config.save_name+f"-N{self.config.num_image}-device_count{torch.cuda.device_count()}-epoch{ep}"
                        torch.save(model_state, save_name)
                        print(f'device {torch.cuda.current_device()} saved model at ' + save_name)
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

    def sample(self, params:torch.tensor=None, num_new_img_per_gpu=192, ema=False, entire=False, save=True):
        # n_sample = params.shape[0]
        # file = self.config.resume

        print(f"device {torch.cuda.current_device()}, sample, params = {params}")
        if params is None:
            params = torch.tensor([4.4, 131.341])
            # params_backup = params.numpy().copy()
        # else:
        params_backup = params.numpy().copy()
        params_normalized = self.rescale(params, self.ranges_dict['params'], to=[0,1])

        print(f"device {torch.cuda.current_device()} sampling {num_new_img_per_gpu} images with normalized params = {params_normalized}")
        params_normalized = params_normalized.repeat(num_new_img_per_gpu,1)
        assert params_normalized.dim() == 2, "params_normalized must be a 2D torch.tensor"
        # print("params =", params)
        # print("params =", params)
        # print("len(params) =", len(params))
        # model = self.ema_model if ema else self.nn_model
        # del self.ema_model, self.nn
        # params = torch.tile(params, (n_sample,1)).to(device)

        # nn_model = ContextUnet(n_param=self.config.n_param, image_size=self.config.HII_DIM, dim=self.config.dim, stride=self.config.stride).to(self.config.device)
        # if ema:
        #     self.nn_model.module.load_state_dict(torch.load(file)['ema_unet_state_dict'])
        # else:
        #     self.nn_model.module.load_state_dict(torch.load(file)['unet_state_dict'])
        # print(f"device {torch.cuda.current_device()} resumed nn_model from {file}")
        # nn_model = ContextUnet(n_param=1, image_size=28)
        # nn_model.train()
        # self.nn_model.to(self.ddpm.device)
        self.nn_model.eval()

        # self.ema_model = ContextUnet(n_param=config.n_param, image_size=config.HII_DIM, dim=config.dim, stride=config.stride).to(config.device)
        # self.ema_model.load_state_dict(torch.load(os.path.join(config.output_dir, f"{config.resume}"))['ema_unet_state_dict'])
        # print(f"resumed ema_model from {config.resume}")

        with torch.no_grad():
            x_last, x_entire = self.ddpm.sample(
                nn_model=self.nn_model, 
                params=params_normalized.to(self.config.device), 
                device=self.config.device, 
                guide_w=self.config.guide_w
                )

        if save:    
            # np.save(os.path.join(self.config.output_dir, f"{self.config.run_name}{'ema' if ema else ''}.npy"), x_last)
            savetime = datetime.datetime.now().strftime("%m%d-%H%M")
            savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]}-zeta{params_backup[1]}-N{self.config.num_image}-device{torch.cuda.current_device()}-{savetime}{'ema' if ema else ''}.npy")
            print(f"saving {savename} ...")
            np.save(savename, x_last)

            if entire:
                savename = os.path.join(self.config.output_dir, f"Tvir{params_backup[0]}-zeta{params_backup[1]}-N{self.config.num_image}-device{torch.cuda.current_device()}-{savetime}{'ema' if ema else ''}_entire.npy")
                print(f"saving {savename} ...")
                np.save(savename, x_entire)
        # else:
        return x_last
# %%

num_train_image_list = [7000]

def train(rank, world_size):
    config = TrainConfig()
    config.world_size = world_size

    ddp_setup(rank, world_size)
    
    #[3200]#[200]#[1600,3200,6400,12800,25600]
    for i, num_image in enumerate(num_train_image_list):
        config.num_image = num_image
        # config.world_size = world_size
        
        ddpm21cm = DDPM21CM(config)
        # print(f" num_image = {ddpm21cm.config.num_image} ".center(50, '-'))
        print(f"run_name = {ddpm21cm.config.run_name}")
        ddpm21cm.train()
        destroy_process_group()

        
if __name__ == "__main__":
    world_size = 1#torch.cuda.device_count()
    print(f" training, world_size = {world_size} ".center(100,'-'))
    # torch.multiprocessing.set_start_method("spawn")
    # args = (config, nn_model, ddpm, optimizer, dataloader, lr_scheduler)

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    # notebook_launcher(ddpm21cm.train, num_processes=1, mixed_precision='fp16')


# %%

# def generate_samples(ddpm21cm, num_new_img_per_gpu, max_num_img_per_gpu, rank, world_size, params):
#     # samples = []
#     for _ in range(num_new_img_per_gpu // max_num_img_per_gpu):    
#         sample = ddpm21cm.sample(
#             params=params, 
#             num_new_img_per_gpu=max_num_img_per_gpu
#             )

#         print(f"device {torch.cuda.current_device()} generated sample of shape: {sample.shape}")

#         # samples.append(sample)
#         # ddpm21cm.sample(params=torch.tensor((5.6, 19.037)), num_new_img_per_gpu=max_num_img_per_gpu)
#         # ddpm21cm.sample(params=torch.tensor((4.699, 30)), num_new_img_per_gpu=max_num_img_per_gpu)
#         # ddpm21cm.sample(params=torch.tensor((5.477, 200)), num_new_img_per_gpu=max_num_img_per_gpu)
#         # ddpm21cm.sample(params=torch.tensor((4.8, 131.341)), num_new_img_per_gpu=max_num_img_per_gpu)
#     # samples = np.concatenate(samples, axis=0)

#     # samples_list = [np.empty_like(samples) for _ in range(world_size)]
#     # dist.all_gather_object(samples_list, samples)

#     # if rank == 0:
#     #     all_samples = np.concatenate(samples_list, axis=0)
#     #     return all_samples
#     # else:
#     #     return None

def generate_samples(rank, world_size, config, num_new_img_per_gpu, max_num_img_per_gpu, return_dict, params):
    ddp_setup(rank, world_size)
    ddpm21cm = DDPM21CM(config)

    # generate_samples(ddpm21cm, num_new_img_per_gpu, max_num_img_per_gpu, rank, world_size, params)

    # samples = []
    for _ in range(num_new_img_per_gpu // max_num_img_per_gpu):    
        sample = ddpm21cm.sample(
            params=params, 
            num_new_img_per_gpu=max_num_img_per_gpu
            )
            
        print(f"device {torch.cuda.current_device()} generated sample of shape: {sample.shape}")

    # print(f"device {torch.cuda.current_device()}, rank = {rank}, keys = {return_dict.keys()}, samples.shape = {np.shape(samples)}")
    # if rank == 0:
    #     return_dict['samples'] = samples
    # print(f"device {torch.cuda.current_device()}, rank = {rank}, keys = {return_dict.keys()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # print(f" sampling, world_size = {world_size} ".center(100,'-'))
    # num_train_image_list = [1600,3200,6400,12800,25600]
    # num_train_image_list = [5000]
    num_new_img_per_gpu = 200
    max_num_img_per_gpu = 40

    # params = torch.tensor([4.4, 131.341])

    # print("config = TrainConfig()")
    config = TrainConfig()
    config.world_size = world_size
    # print("config.world_size = world_size")

    for num_image in num_train_image_list:
        config.num_image = num_image
        config.resume = f"./outputs/model_state-N{num_image}-device_count{torch.cuda.device_count()}-epoch{config.n_epoch-1}"

        # print("ddpm21cm = DDPM21CM(config)")
        manager = mp.Manager()
        return_dict = manager.dict()

        params_pairs = [
            (4.4, 131.341),
            (5.6, 19.037),
            (4.699, 30),
            (5.477, 200),
            (4.8, 131.341),
        ]
        for params in params_pairs:
            print(f" sampling for {params}, world_size = {world_size} ".center(100,'-'))
            mp.spawn(generate_samples, args=(world_size, config, num_new_img_per_gpu, max_num_img_per_gpu, return_dict, torch.tensor(params)), nprocs=world_size, join=True)

        # print("---"*30)
        # print(f"device {torch.cuda.current_device()}, keys = {return_dict.keys()}")
        # if "samples" in return_dict:
        #     samples = return_dict["samples"]
        #     print(f"device {torch.cuda.current_device()} generated samples shape: {samples.shape}")


# %%
# ls -lth outputs | head

# # %%
# def plot_grid(samples, c=None, row=1, col=2):
#     print("samples.shape =", samples.shape)
#     for j in range(samples.shape[4]):
#         plt.figure(figsize = (12,6), dpi=400)
#         for i in range(len(samples)):
#             plt.subplot(row,col,i+1)
#             plt.imshow(samples[i,0,:,:,j], cmap='gray')#, vmin=-1, vmax=1)
#             plt.xticks([])
#             plt.yticks([])
#         # plt.suptitle(f"ION_Tvir_MIN = {c[0][0]}, HII_EFF_FACTOR = {c[0][1]}")
#             # plt.show()
#         # plt.suptitle('simulations')
#         plt.tight_layout()
#         plt.subplots_adjust(wspace=0, hspace=0)
#         plt.savefig(f"test3D-{j:03d}.png")
#         plt.close()
#         # plt.show()
    
# data = np.load("outputs/Tvir4.400000095367432-zeta131.34100341796875-N1000.npy")
# # print(data.shape)
# plot_grid(data)
# plt.imshow(data)

# %%
# config = TrainConfig()
# def plot(filename, row=4, col=6):
#     samples = np.load(filename)
#     params = filename.split('guide_w')[-1][:-4]
#     print("plotting", samples.shape, params)
#     plt.figure(figsize = (8,8))
#     for i in range(24):
#         plt.subplot(row,col,i+1)
#         plt.imshow(samples[i,0,:,:], cmap='gray')#, vmin=-1, vmax=1)
#         plt.xticks([])
#         plt.yticks([])
#         # plt.show()
#     plt.suptitle(params)
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0, hspace=0) 
#     plt.show()
#     # plt.savefig('outputs/'+params+'.png')
#     # plt.close()
#     # plt.imshow(images[0,0])
#     # plt.show()

# %%


