#!/usr/bin/env python
# coding: utf-8

# # load dataset

# In[1]:
import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(f"sys.path.append(parent_dir): {parent_dir}")
sys.path.append(parent_dir)

from utils.load_h5 import Dataset4h5, ranges_dict
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import h5py
import matplotlib as mpl
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
from scipy.linalg import sqrtm
# print("before torch")
import torch
# print("after torch")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from kymatio.torch import Scattering2D
from sklearn.preprocessing import PowerTransformer
import joblib

import gc
# print("before summary writer")
# from torch.utils.tensorboard import SummaryWriter
# print("after summary writer")
import multiprocessing
import matplotlib.ticker as ticker
from numpy import interp
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import FormatStrFormatter

from typing import List
import argparse

# print("before spawn")
multiprocessing.set_start_method('spawn', force=True)
# print("after spawn")

#ranges_dict = dict(
#    params = {
#        0: [4, 6], # ION_Tvir_MIN
#        1: [10, 250], # HII_EFF_FACTOR
#        },
#    images = {
#        0: [-387, 86], # brightness_temp
#        # 0: [-338, 54], # brightness_temp
#        }
#    )


# In[2]:


def load_h5_as_tensor(dir_name='LEN128-DIM64-CUB8.h5', num_image=256, num_redshift=32, HII_DIM=64, scale_path=False, dim=3, startat=0):
    # print("dataset = Dataset4h5(")
    dir_name = os.path.join(os.environ['SCRATCH'], dir_name)
    dataset = Dataset4h5(dir_name, num_image=num_image, num_redshift=num_redshift, HII_DIM=HII_DIM, scale_path=scale_path, dim=dim, startat=startat)

    # print("with h5py.File(dir_name)")
    with h5py.File(dir_name) as f:
        # print(f.keys())
        # print(f['params'])
        # print(f['redshifts_distances'])
        los = f['redshifts_distances'][:,startat:startat+dataset.num_redshift]

    # print("dataloader = DataLoader(")
    dataloader = DataLoader(dataset, batch_size=800)
    
    # print("x, c = next(iter(dataloader))")
    x, c = next(iter(dataloader))
    # print("x.shape =", x.shape)
    # print("c.shape =", c.shape)
    #print("x.min() =", x.min())
    #print("x.max() =", x.max())
    #print(f"loaded x.shape = {x.shape}")
    return x, c, los


# In[3]:

os.environ["SLURM_NODEID"] = '0'

import matplotlib

def get_eor_cmap(vmin=-150, vmax=30):
    name = f"EoR-{vmin}-{vmax}"
    negative_segments = 4
    positive_segments = 2
    neg_frac = abs(vmin) / (vmax - vmin)
    neg_seg_size = neg_frac / negative_segments
    pos_frac = abs(vmax) / (vmax - vmin)
    pos_seg_size = pos_frac / positive_segments

    EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list(
        name,
        [
            (0, 'white'),
            (0.33, 'yellow'),
            (0.5, 'orange'),
            (0.68, 'red'),
            (0.83333333, 'black'),
            (0.9, 'blue'),
            (1, 'cyan')])
    
    try:
        matplotlib.colormaps.register(cmap=EoR_colour)
    except ValueError:
        matplotlib.colormaps.unregister(name)
        matplotlib.colormaps.register(cmap=EoR_colour)

    return name

vmin = -150#Tb_all.min()
vmax = 30#Tb_all.max()
# print(vmin, vmax)
cmap = get_eor_cmap(vmin, vmax)

def plot_grid(samples, c, row=8, col=12, idx=0, los=None, savename=None, figsize=(16, 4.5)): # (64,128)
    print(f"plot_grid: samples.shape = {samples.shape}")

    fig, axes = plt.subplots(row, col, figsize=figsize, dpi=100)#, constrained_layout=True)
    plt.subplots_adjust(wspace=0, hspace=-0.001)
    axes = axes.flatten()

    for ax in axes[row//2*col:]:  # 选择第 row//2 行的所有列
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.003, pos.width, pos.height])  # 手动下移

    #print(samples.shape)
    for i in range(row*col):
        if i >= samples.shape[0]:
            axes[i].axis("off")
            continue
        if samples.ndim == 5:
            #im = axes[i].imshow(samples[i,0,:,:,idx], cmap=cmap, vmin=vmin, vmax=vmax)
            im = axes[i].imshow(samples[i,0,:,idx,:], cmap=cmap, vmin=vmin, vmax=vmax)
        elif samples.ndim == 4:
            im = axes[i].imshow(samples[i,0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].axis("off")
        
    cbar_ax = fig.add_axes([0.90, 0.128, 0.01, 0.737]) 
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Brightness Temperature (mK)', fontsize=10) 
    
    plt.suptitle(f"ION_Tvir_MIN = {c[0][0]:.3f}, HII_EFF_FACTOR = {c[0][1]:.3f},\nz = [{los[0,0]:.2f}, {los[0,-1]:.2f}] {savename}")
    plt.colormaps()
    
    if savename is None:
        plt.show()
    else:
        savename = f"Tvir_zeta-{c[0][0]:.3f}_{c[0][1]:.3f}_{savename}.png"
        plt.savefig(savename, bbox_inches='tight',)
        print(f"Image saved to {savename}")
    plt.close()
    gc.collect()
    
import cv2
import glob
import os

def png2mp4(
    image_folder = ".",  # 你的图片所在的目录
    image_format = "x0_*.png",  # 你的图片格式
    output_video = "x0.mp4",  # 生成的视频文件
):
    # 读取所有匹配的 PNG 图片（按文件名排序）
    images = sorted(glob.glob(os.path.join(image_folder, image_format)))
    
    if not images:
        print("未找到匹配的 PNG 图片")
        exit()
    
    # 读取第一张图片以获取宽高
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    
    # 创建 VideoWriter（H.264 编码）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 也可以用 'XVID' 或 'avc1'
    fps = 10  # 设置帧率（10帧/秒）
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    # 逐帧写入视频
    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)
    
    # 释放资源
    video.release()
    print(f"video is saved as {output_video}")

# In[17]:


def calc_ps(field, L):
    ng = field.shape[0]
    ru = np.fft.fftn(field)
    
    ru *= (L/ng)**field.ndim

    if field.ndim == 3:
        ru = ru[0:ng//2+1, 0:ng//2+1, 0:ng//2+1]
    elif field.ndim ==2:
        ru = ru[0:ng//2+1, 0:ng//2+1]

    # ru *= (2/ng)**field.ndim
    ru = np.abs(ru)**2

    kx = np.fft.rfftfreq(ng) * ng / L
    ky = kx.copy()
    kz = kx.copy()

    kmin = 1/L
    kmax = 0.5*ng/L
    
    kbins = np.arange(kmin, kmax, kmin)
    Nk = len(kbins)
    
    if field.ndim == 3:
        k_nd = np.meshgrid(kx, ky, kz, indexing="ij")
        # print("field.ndim == 3:")
    elif field.ndim == 2:
        k_nd = np.meshgrid(kx, ky, indexing="ij")
        # print("field.ndim == 2:")
    
    k = np.sqrt(np.sum(np.array(k_nd)**2,axis=0))

    # hist, edges = np.histogram(k, weights=ru, bins=Nk)
    # Pk = ng * hist / kbins**(field.ndim - 1)
    
    Pk = np.array([np.mean(ru[(k >= kbins[i]) & (k < kbins[i+1])]) for i in range(len(kbins)-1)])
    kbins = (kbins[:-1] + kbins[1:])/2

    if field.ndim == 3:
        Pk *= (kbins**3) / (2*np.pi**2)
    elif field.ndim == 2:
        Pk *= (kbins**2) / (4*np.pi**2)

    return kbins, Pk


# In[18]:


# 示例三维密度场
# Nx, Ny, Nz = 64, 64, 512  # 密度场的大小，长方体
# box_size = (128.0, 128.0, 1024.0)  # 盒子大小（单位Mpc/h），对应于 (Lx, Ly, Lz)
# plt.figure(figsize=(6, 4), dpi=100)
# k_vals_all = []
def x2Pk(x):
    print(f"x2Pk, x.shape = {x.shape}")
    Pk_vals_all = []
    for i in range(x.shape[0]):
        startat=512
        if x.ndim == 4:
            # density_field = x[i,0,:,x.shape[-1]//2:x.shape[-1]//2+64]
            density_field = x[i,0,:,startat:startat+64]
        elif x.ndim == 5:
            # density_field = x[i,0,:,:,x.shape[-1]//2:x.shape[-1]//2+64]
            density_field = x[i,0,:,0,startat:startat+64]
        if density_field.ndim == 3:
            Nx, Ny, Nz = density_field.shape
            box_size = 128#(128.0, 128.0, 1024.0) #512#
        elif density_field.ndim == 2:
            Nx, Ny = density_field.shape
            box_size = 128#(128.0, 1024.0) #512#

        # 计算物质功率谱
        k_vals, Pk_vals = calc_ps(density_field, box_size)
        # k_vals_all.append(k_vals)
        Pk_vals_all.append(Pk_vals)

    Pk_vals_all = np.array(Pk_vals_all)
    return k_vals, Pk_vals_all


# def rescale(x, ranges=ranges_dict['images']):
#     #x = (x + 1) / 2 * (ranges[0][1]-ranges[0][0]) + ranges[0][0]
#     x = x * ranges[0][1] + ranges[0][0]
#     return x
    
def x2Tb(x):
    #print('x.shape =', x.shape, 'x.ndim =', x.ndim)
    if x.ndim == 4:
        Tb = x[:,0].mean(axis=1)
    elif x.ndim == 5:
        Tb = x[:,0].mean(axis=(1,2))
    return Tb

def load_x_ml(fname_pattern0, fname_pattern1, ema = 0, outputs_dir = "../training/outputs"):
    # num = 7200
    x_ml = []
    fnames = [fname for fname in os.listdir(outputs_dir) if fname_pattern0 in fname and fname_pattern1 in fname and f'-ema{ema}' in fname]
    print("fname pattern:", fname_pattern0, fname_pattern1, "; len(fnames) =", len(fnames), ";\nfnames[0] =", fnames[0])
    # print("fname:",fnames)
    # print()
    for fname in fnames:
    #    if ema and 'ema1' not in fname:
    #        continue
    #    if not ema and 'ema1' in fname:
    #        continue
        data = np.load(os.path.join(outputs_dir, fname))
        # print(fname)
        x_ml.append(data)

    x_ml = np.concatenate(x_ml, axis=0)
    pt = joblib.load(f"../utils/power_transformer_25600.pkl")
    original_shape = x_ml.shape
    x_ml = pt.inverse_transform(x_ml.reshape(-1, original_shape[-1]))
    # x_ml = rescale(x_ml)
    x_ml = torch.from_numpy(x_ml.reshape(*original_shape))
    print(f"loaded x_ml.shape = {x_ml.shape}")
    return x_ml


def plot_global_signal(x_pairs, params, los, sigma_level=68.27, alpha=0.2, interval = 10, lw = 0.6, y_eps = 0.2, savename=None):
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(8,6), dpi=100, gridspec_kw={'height_ratios': [1.5,.5,.5,.5]})
    
    for i, (x0, x1) in enumerate(x_pairs):
        # print(Tb0.shape)
        Tb0 = x2Tb(x0)
        Tb1 = x2Tb(x1)
        
        y0 = np.median(Tb0, axis=0)
        y1 = np.median(Tb1, axis=0)

        Tb0_perc = np.percentile(Tb0, [100-sigma_level, sigma_level], axis=0)
        ax[0].fill_between(los[1,:Tb0_perc.shape[-1]], Tb0_perc[0], Tb0_perc[1], alpha=alpha, facecolor=f"C{i}", edgecolor='black')
        # ax[0].plot(los[1], Tb0[:4].T, lw=0.5)
        Tb1_perc = np.percentile(Tb1, [100-sigma_level, sigma_level], axis=0)
        yerr_lower = y1 - Tb1_perc[0]
        yerr_upper = Tb1_perc[1] - y1
        ax[0].errorbar(los[1,:Tb0_perc.shape[-1]][::interval], y1[::interval], yerr=[yerr_lower[::interval], yerr_upper[::interval]], linestyle='-', c=f"C{i}", marker='|', markersize=1, linewidth=lw)#, label='diffusion')

        ax[0].plot(los[1,:Tb0_perc.shape[-1]], y0, linestyle=':', c=f"C{i}", lw=3*lw)
        ax[1].plot(los[1,:Tb0_perc.shape[-1]][abs(y0)>y_eps], ((y1-y0)/abs(y0))[abs(y0)>y_eps], label=f'{np.array(params[i])}', c=f"C{i}", lw=lw)

        sigma0 = 0.5*(Tb0_perc[1]-Tb0_perc[0])
        sigma1 = 0.5*(Tb1_perc[1]-Tb1_perc[0])
        ax[2].plot(los[1,:Tb0_perc.shape[-1]][sigma0>1.5*y_eps], ((y1-y0)/sigma0)[sigma0>1.5*y_eps], label=f'{np.array(params[i])}', c=f"C{i}", lw=lw)
        
        ax[3].plot(los[1,:Tb0_perc.shape[-1]][sigma0>1.5*y_eps], (sigma1/sigma0-1)[sigma0>1.5*y_eps], c=f"C{i}", lw=lw)

    ax[0].set_ylabel(r'$\langle T_b \rangle$ [mK]')
    ax[0].grid()
    
    ax1_handles, ax1_labels = ax[1].get_legend_handles_labels()
    
    legend_line1 = Line2D([0], [0], linestyle=':', color='black')
    legend_line2 = Line2D([0], [0], linestyle='-', color='black', marker='|', markersize=8)
    
    # 创建自定义图例条目
    legend_elements = [
        (Patch(facecolor='black', edgecolor='black', alpha=alpha),legend_line1), 
        (legend_line2),
    ]
    legend_labels = ['21cmfast', 'diffusion']
    # 添加自定义图例
    ax[0].legend(
        legend_elements + ax1_handles, 
        legend_labels + ax1_labels, 
        handler_map={tuple: HandlerTuple(ndivide=None)}, 
    )

    ax[1].set_ylabel(r"$\epsilon_{rel}$")
    ax[1].set_yscale("symlog", linthresh=0.1)

    ax[1].grid()
    
    ax1_sec = ax[1].secondary_xaxis('top')
    ax1_sec.set_xticklabels([])

    ax[2].set_ylabel(r"$\epsilon_{std}$")
    
    ax[3].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.1f}'))
    ax[3].set_xlabel('distance [Gpc]')
    ax[2].grid()
    ax[3].grid()
    
    ax2_sec = ax[2].secondary_xaxis('top')
    ax2_sec.set_xticklabels([])

    ax3_sec = ax[3].secondary_xaxis('top')
    ax3_sec.set_xticklabels([])
    ax[3].set_ylabel(r"$\epsilon_{\sigma}$")
    
    ax_twin = ax[3].secondary_xaxis('bottom')               # 创建共享 y 轴的第二个 x 轴
    ax_twin.set_xlim(ax[3].get_xlim())       # 设置副 x 轴的范围与主 x 轴相同
    ax_twin.set_xlabel('redshift')           # 设置副 x 轴标签
    ax_twin.xaxis.set_major_locator(ticker.MaxNLocator(10))  # 这里5表示最多显示5个刻度
    ax_twin.set_xticks(ax_twin.get_xticks())                  # 设置刻度为 z 的值
    z_ticks = interp(ax_twin.get_xticks(), los[1], los[0])
    ax_twin.set_xticklabels([f"{ztick:.1f}" for ztick in z_ticks])
    ax_twin.spines['bottom'].set_position(('outward', 40))  # 将副 x 轴向外移动 40 像素
    
    for axis in ax:
        axis.tick_params(axis='y', labelsize=10)  # 设置所有子图的 y 轴刻度标签字体大小为 8

    plt.subplots_adjust(hspace=0)
    if savename == None:
        plt.show()
    else:
        savename = f"global_Tb_{savename}.png"
        plt.savefig(savename, bbox_inches='tight',)
        print(f'Image saved to {savename}')


# In[35]:


def plot_power_spectrum(x_pairs, params, los, sigma_level=68.27, alpha=0.2, redshift=None, savename=None):
    
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(8,6), dpi=100)
    
    for i, (x0, x1) in enumerate(x_pairs):
        k_vals, Pk0 = x2Pk(x0)
        k_vals, Pk1 = x2Pk(x1)
        y0 = np.median(Pk0, axis=0)
        y1 = np.median(Pk1, axis=0)
        
        Pk0_perc = np.percentile(Pk0, [100-sigma_level, sigma_level], axis=0)
        ax[0].fill_between(k_vals, Pk0_perc[0], Pk0_perc[1], alpha=alpha, facecolor=f"C{i}", edgecolor='black')

        ax[0].plot(k_vals, y0, linestyle=':', c=f"C{i}")#, label='sim')

        Pk1_perc = np.percentile(Pk1, [100-sigma_level, sigma_level], axis=0)
        yerr_lower = y1 - Pk1_perc[0]
        yerr_upper = Pk1_perc[1] - y1
        ax[0].errorbar(k_vals, y1, yerr=[yerr_lower, yerr_upper], linestyle='-', c=f"C{i}", marker='|', markersize=1, linewidth=1)#, label='diffusion')
        
        ax[1].plot(k_vals, (y1-y0)/y0, label=f'{np.array(params[i])}', c=f"C{i}")

        sigma = 0.5*(Pk0_perc[1]-Pk0_perc[0])
        ax[2].plot(k_vals, (y1-y0)/sigma, label=f'{np.array(params[i])}', c=f"C{i}")

        ax[3].plot(k_vals, (Pk1_perc[1]-Pk1_perc[0])/2/sigma-1, c=f"C{i}")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\Delta^2(k)$ [mK$^2$]')
    ax[0].grid()
    
    legend_line1 = Line2D([0], [0], linestyle=':', color='black')
    legend_line2 = Line2D([0], [0], linestyle='-', color='black', marker='|', markersize=10)

    # 创建自定义图例条目
    legend_elements = [
        (Patch(facecolor='black', edgecolor='black', alpha=alpha),legend_line1), 
        (legend_line2),
    ]
    # 添加自定义图例
    ax[0].legend(legend_elements, ['21cmfast', 'diffusion'], handler_map={tuple: HandlerTuple(ndivide=None)})

    ax[0].set_title(r"power spectrum of $T_b$ at z = "+f"{los[0].mean():.2f}")
        # plt.xlim(xmin=0.01)
        # ax[0].legend()

    ax[1].set_xscale('log')
        # ax[1].hlines(0,0.01,0.3)
        # ax[1].hlines(0.1,0.01,0.3)
        # ax[1].hlines(-0.1,0.01,0.3)
    ax[1].set_ylabel(r"$\epsilon_{rel}$")
    # ax[1].set_xlabel('k [Mpc$^{-1}$]')
    ax[1].grid()
    ax1_sec = ax[1].secondary_xaxis('top')
    ax1_sec.set_xticklabels([])
    ax[1].legend()

    ax[2].set_xscale('log')
    ax[2].set_ylabel(r"$\epsilon_{std}$")
    # ax[3].set_xlabel('k [Mpc$^{-1}$]')
    ax[2].grid()
    ax2_sec = ax[2].secondary_xaxis('top')
    ax2_sec.set_xticklabels([])

    ax[3].set_xscale('log')
    ax[3].set_ylabel(r"$\epsilon_{\sigma}$")
    ax[3].set_xlabel('k [Mpc$^{-1}$]')
    ax[3].grid()
    ax3_sec = ax[3].secondary_xaxis('top')
    ax3_sec.set_xticklabels([])

    plt.subplots_adjust(hspace=0)

    if savename == None:
        plt.show()
    else:
        savename = f"power_spectrum_{savename}.png"
        plt.savefig(savename, bbox_inches='tight',)
        print(f'Image saved to {savename}')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sort_S2_by_l(S2, jthetas, L):
    # print("S2.shape =", S2.shape)
    S2_sorted = np.empty((L, S2.shape[0], S2.shape[1]//L))
    jthetas_sorted = np.empty((L, S2.shape[1]//L, 2, 2))
    # jthetas_sorted = [[]]*L
    num_updated = np.zeros(L, dtype=int)

    # print("jthetas =", jthetas)
    for i, jtheta in enumerate(jthetas):
        index = (jtheta[1][1]-jtheta[1][0]) % L
        # print("index =", index)
        # print("num_updated =", num_updated)
        S2_sorted[index, :, num_updated[index]] = S2[:,i]
        # print("sort_S2_by_l", jthetas_sorted[index*L+num_updated[index]], np.array(jtheta))
        jthetas_sorted[index, num_updated[index]][0] = jtheta[0]
        jthetas_sorted[index, num_updated[index]][1] = jtheta[1]
        # print("S2_sorted[:, index*L+num_updated[index]] =", S2_sorted[:, index*L+num_updated[index]])
        # jthetas_sorted[index] = jtheta
        # print("jtheta =", jtheta)
        # print("jthetas_sorted.shape =", jthetas_sorted.shape)
        # print("left =", jthetas_sorted[index*L+num_updated[index],:])
        # print("right =", np.array(jtheta))
        # jthetas_sorted[index*L+num_updated[index],:] = np.array(jtheta)
        # print("index, S2_sorted.shape =", index, np.shape(S2_sorted))
        num_updated[index] += 1
    # print("index*L+num_updated[index]", index*L+num_updated[index])
    # print("i =", i)
    # print("sort_S2_by_l, num_updated =", num_updated)

    S2_sorted = np.array(S2_sorted)
    # print("S2_sorted.shape", S2_sorted.shape)
    # print("S2_sorted", S2_sorted)
    jthetas_sorted = np.array(jthetas_sorted)
    # print('sort_S2_by_l S2_sorted.shape',S2_sorted.shape)
    # print('sort_S2_by_l jthetas_sorted.shape', jthetas_sorted.shape)
    return S2_sorted, jthetas_sorted


def calculate_sorted_S2(x, S, J, L, jthetas):
    S_all = np.mean(S(x.to(device))[:,0].cpu().numpy(), axis=(2,3))
    # print("calculate_sorted_S2, S2.shape =", S_all.shape)
    # print("calculate_sorted_S2, jthetas.shape =", jthetas.shape)
    # print("calculate_sorted_S2, jthetas[21:41] =", jthetas[21:41])
    # print("calculate_sorted_S2, jthetas[-1] =", jthetas[-1])

    ############################################################
    for j1 in range(J-1):
        for j2 in range(j1+1, J):
            # if j2>j1:
            # print("j1", j1, "j2", j2)
            index = [jtheta[0] == (j1,j2) for jtheta in jthetas]
            # print(index)
            # print(jthetas[index])
            # cache = S2[:,index]
            # print(index)
            if (j1,j2) == (0,1):
                S2 = S_all[:,index]
                jthetas_2 = np.array(jthetas[index])
                # index_reduced = index
            else:
                S2 = np.concatenate((S2, S_all[:,index]), axis = 1)
                jthetas_2 = np.concatenate([jthetas_2, np.array(jthetas[index])], axis = 0)

    S2_sorted, jthetas_sorted = sort_S2_by_l(S2, jthetas_2, L)
    # print(index_reduced.shape)
    return S2_sorted, jthetas_sorted

def calculate_reduced_S2(x_pairs, params, J=5, L=4, M=64, N=64):
    S2_reduced_list = []
    jthetas_reduced_list = []
    for i, (x0, x1) in enumerate(x_pairs):
        #print(f"#{i}: x0.shape = {x0.shape}, x1.shape = {x1.shape}")
        # get jthetas and S
        startat=512
        if x0.ndim == 4:
            x0 = x0[...,startat:startat+64]
            x1 = x1[...,startat:startat+64]
        elif x0.ndim == 5:
            x0 = x0[...,0,startat:startat+64]
            x1 = x1[...,0,startat:startat+64]

        if i == 0:
            S = Scattering2D(J, (M, N), L=L, out_type='list').to(device)
            jthetas = []
            for dicts in S(x0.to(device)):
                jthetas.append([dicts['j'], dicts['theta']])
            # print(jthetas[0])
            # print(jthetas[1])
            # print(jthetas[2])
            # print(jthetas[3])
            # print(jthetas[-2])
            # print(jthetas[-1])
            jthetas = np.array(jthetas, dtype=object)
            S = Scattering2D(J, (M, N), L=L).to(device)
            # print("type(dicts[j])", type(dicts['j']), dicts['j'])
        # print("plot_scattering_transform_2 jthetas.shape", jthetas.shape)
        # print(jthetas[0], jthetas[1], jthetas[160])
        S2_reduced_0, jthetas_reduced_0 = calculate_sorted_S2(x0, S, J, L, jthetas)
        S2_reduced_1, jthetas_reduced_1 = calculate_sorted_S2(x1, S, J, L, jthetas)
        # print("S2_reduced.shape =", S2_reduced.shape)
        S2_reduced_list.append((S2_reduced_0, S2_reduced_1))
        jthetas_reduced_list.append((jthetas_reduced_0, jthetas_reduced_1))

    return S2_reduced_list, jthetas_reduced_list


# In[44]:


def average_single_S2_over_l(S2, jthetas, L=4):
    # print("average_single_S2_over_l, shape =", S2.shape, jthetas.shape)
    S2_reshape = np.array(np.array_split(S2, S2.shape[2]//L, axis=2))
    # jthetas_reshape = jthetas.reshape(jthetas.shape[0],jthetas.shape[1]//L, L,jthetas.shape[2],jthetas.shape[3])
    jthetas_reshape = np.array(np.array_split(jthetas, jthetas.shape[1]//L, axis=1))
    # print("average_single_S2_over_l, shape =", S2_reshape.shape, jthetas_reshape.shape)
    # print("---"*30)
    # print(jthetas_reshape[0])
    # print("---"*30)
    S2_average = np.average(S2_reshape, axis=3)
    # jthetas_average = np.average(jthetas_reshape, axis=2)
    jthetas_average = jthetas_reshape[:,:,0,0,:]
    # print(S2_average.shape, jthetas_average.shape)
    # print(jthetas_average)
    # print("---"*30)
    S2_transpose = S2_average.transpose(2,1,0)
    # print(S2_transpose.shape)
    # print("---"*30)
    S2_combine = S2_transpose.reshape(S2_transpose.shape[0], -1)
    # print(S2_combine.shape)
    # print("---"*30)
    j1j2 = jthetas_average.transpose(1,0,2).reshape(-1, 2)
    # j1j2 = np.tile(jthetas_average, (S2_combine.shape[1]//jthetas_average.shape[0],1))
        # S2_average.append(())
    return S2_combine, j1j2.astype(int)

def average_S2_over_l(x_pairs, params, J, L, M, N):

    S2_list, jthetas_list = calculate_reduced_S2(x_pairs, params, J, L, M, N)

    S2_average = []
    j1j2_average = []
    for i in range(len(S2_list)):
        S2_sim = S2_list[i][0]
        jthetas_sim = jthetas_list[i][0]
        S2_combine_sim, j1j2_sim = average_single_S2_over_l(S2_sim, jthetas_sim, L)

        S2_ml = S2_list[i][1]
        jthetas_ml = jthetas_list[i][1]
        S2_combine_ml, j1j2_ml = average_single_S2_over_l(S2_ml, jthetas_ml, L)

        S2_average.append((S2_combine_sim, S2_combine_ml))
        j1j2_average.append((j1j2_sim, j1j2_ml))
    return np.array(S2_average), np.array(j1j2_average)

# In[46]:


def plot_scattering_transform_2(x_pairs, params, los, sigma_level=68.27, alpha=0.2, J=5, L=4, M=64, N=64, savename=None):
    # S2_reduced, jthetas_reduced = calculate_reduced_S2(x_pairs, params, J, L, M, N)
    S2, j1j2 = average_S2_over_l(x_pairs, params, J, L, M, N)
    #print("S2.shape, j1j2.shape =", S2.shape, j1j2.shape)
    # plt.figure(dpi=200, figsize=(12,4))
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(12,6), dpi=100)
    ax[0].set_title(f"reduced scattering coefficients at z = {los[0].mean():.2f}")
    # S2 = S2[..., :20]
    #print("S2.min() =", S2.min())
    S2 = np.log10(S2)
    # j1j2 = j1j2[..., :S2.shape[3], :]
    j1j2 = j1j2[0,0]
    for i in range(len(S2)):
        S2_sim = S2[i][0]
        S2_ml = S2[i][1]

        y0 = np.median(S2_sim, axis=0)
        y1 = np.median(S2_ml, axis=0)
        # print(y0.shape)
        ax[0].plot(np.arange(y0.shape[0]), y0, lw=1, c=f"C{i}", linestyle=':')
        # plt.plot(np.median(S2_ml, axis=0), lw=1)

        S2_sim_perc = np.percentile(S2_sim, [100-sigma_level, sigma_level], axis=0)
        # S2_ml_perc = np.percentile(S2_ml, [100-sigma_level, sigma_level], axis=0)
        ax[0].fill_between(np.arange(S2_sim.shape[1]), S2_sim_perc[0], S2_sim_perc[1], alpha=alpha, facecolor=f"C{i}", edgecolor='black')
        # plt.fill_between(np.arange(S2_ml.shape[1]), S2_ml_perc[0], S2_ml_perc[1], alpha=alpha)

        S2_ml_perc = np.percentile(S2_ml, [100-sigma_level, sigma_level], axis=0)
        yerr_lower = y1 - S2_ml_perc[0]
        yerr_upper = S2_ml_perc[1] - y1
        ax[0].errorbar(np.arange(y1.shape[0]), y1, yerr=[yerr_lower, yerr_upper], linestyle='-', c=f"C{i}", marker='|', markersize=1, linewidth=1)#, label='diffusion')

        ax[1].plot(np.arange(y0.shape[0]), ((y1-y0)/y0), label=f'{np.array(params[i])}', c=f"C{i}")

        sigma = (S2_sim_perc[1]-S2_sim_perc[0])/2
        ax[2].plot(np.arange(y0.shape[0]), (y1-y0)/sigma, label=f'{np.array(params[i])}', c=f"C{i}")

        # ax[3].plot(np.arange(y0.shape[0]), (S2_sim_perc[1]-S2_sim_perc[0])/sigma-1, c=f"C{i}")
        ax[3].plot(np.arange(y0.shape[0]), (S2_ml_perc[1]-S2_ml_perc[0])/2/sigma-1, c=f"C{i}")

    legend_line1 = Line2D([0], [0], linestyle=':', color='black')
    legend_line2 = Line2D([0], [0], linestyle='-', color='black', marker='|', markersize=10)

    # 创建自定义图例条目
    legend_elements = [
        (Patch(facecolor='black', edgecolor='black', alpha=alpha),legend_line1), 
        (legend_line2),
    ]
    # 添加自定义图例
    ax[0].legend(legend_elements, ['21cmfast', 'diffusion'], handler_map={tuple: HandlerTuple(ndivide=None)})

    ax[0].set_ylabel(r'$\log{S_2}$')
    ax[0].grid()
    j1j2_period = j1j2.shape[0]//L

    # plt.text()
    ax[0].vlines(np.arange(0-0.5, j1j2.shape[0]-0.5+j1j2_period,j1j2_period), ax[0].get_ylim()[0], ax[0].get_ylim()[1], colors='grey', alpha=0.8, linestyles=':')
    ax[1].vlines(np.arange(0-0.5, j1j2.shape[0]-0.5+j1j2_period,j1j2_period), ax[1].get_ylim()[0], ax[1].get_ylim()[1], colors='grey', alpha=0.8, linestyles=':')
    ax[2].vlines(np.arange(0-0.5, j1j2.shape[0]-0.5+j1j2_period,j1j2_period), ax[2].get_ylim()[0], ax[2].get_ylim()[1], colors='grey', alpha=0.8, linestyles=':')
    ax[3].vlines(np.arange(0-0.5, j1j2.shape[0]-0.5+j1j2_period,j1j2_period), ax[3].get_ylim()[0], ax[3].get_ylim()[1], colors='grey', alpha=0.8, linestyles=':')

    ax1_sec = ax[1].secondary_xaxis('top')
    ax1_sec.set_xticklabels([])
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel(r'$\epsilon_{rel}$')

    # ax[2].set_xticks(np.arange(j1j2.shape[0]), j1j2, rotation=90)
    # ax[2].set_xlabel(r'$(j_1,\ j_2)$')
    ax[2].grid()
    # ax[2].set_ylim(ymin=-1.1)
    ax[2].set_ylabel(r'$\epsilon_{std}$')
    # print(ax[2].get_xticks())
    ax2_sec = ax[2].secondary_xaxis('top')
    ax2_sec.set_xticklabels([])
    # ax[2].set_xlim((0,19))

    ax[3].set_xticks(np.arange(j1j2.shape[0]), j1j2, rotation=90)
    ax[3].set_xlabel(r'$(j_1,\ j_2)$')
    ax[3].grid()
    # ax[2].set_ylim(ymin=-1.1)
    ax[3].set_ylabel(r'$\epsilon_{\sigma}$')
    # print(ax[2].get_xticks())
    ax3_sec = ax[3].secondary_xaxis('top')
    ax3_sec.set_xticklabels([])
    # ax[3].set_xlim((0,19))

    for i in range(L):
        if i*j1j2_period < ax[3].get_xlim()[1]:
            ax[3].text(x=i*j1j2_period, y=0.35+ax[3].get_ylim()[0], s=r"$(l_2-l_1)\%L$="+f"{i}")

    plt.subplots_adjust(hspace=0)

    if savename == None:
        plt.show()
    else:
        savename = f"scattering_coefficients_{savename}.png"
        plt.savefig(savename, bbox_inches='tight',)
        print(f'Image saved to {savename}')


def evaluate(
    what2plot: List[str] = ['grid', 'global_signal', 'power_spectrum', 'scatter_transform'],
    device_count: int = 4,
    node: int = 8,
    jobID: int = 35912978,
    epoch: int = 120,
    use_ema: int = 0,
    ):

    print(f"device = {device}")
    config = f"device_count{device_count}-node{node}-{jobID}-epoch{epoch}"

    for ema in range(use_ema+1):
        print('🚀')
        save_name = f"{jobID}_ema{ema}"

        x0_ml = load_x_ml(f"Tvir4.400-zeta131.341", config, ema = ema)
        x1_ml = load_x_ml(f"Tvir5.600-zeta19.037", config, ema = ema)
        x2_ml = load_x_ml(f"Tvir4.699-zeta30.000", config, ema = ema)
        x3_ml = load_x_ml(f"Tvir5.477-zeta200.000", config, ema = ema)
        x4_ml = load_x_ml(f"Tvir4.800-zeta131.341", config, ema = ema)

        print(f"x0_ml.shape = {x0_ml.shape}")
        dim = x0_ml[0,0].ndim 
        if dim == 2:
            num_image, _, HII_DIM, num_redshift = x0_ml.shape
        elif dim == 3:
            num_image, _, HII_DIM, _, num_redshift = x0_ml.shape

        x0, c0, los = load_h5_as_tensor('LEN128-DIM64-CUB16-Tvir4.4-zeta131.341-0812-104709.h5',num_image=num_image,num_redshift=num_redshift,dim=dim)
        x1, c1, los = load_h5_as_tensor('LEN128-DIM64-CUB16-Tvir5.6-zeta19.037-0812-104704.h5',num_image=num_image,num_redshift=num_redshift,dim=dim)
        x2, c2, los = load_h5_as_tensor('LEN128-DIM64-CUB16-Tvir4.699-zeta30-0812-104322.h5',num_image=num_image,num_redshift=num_redshift,dim=dim)
        x3, c3, los = load_h5_as_tensor('LEN128-DIM64-CUB16-Tvir5.477-zeta200-0812-104013.h5',num_image=num_image,num_redshift=num_redshift,dim=dim)
        x4, c4, los = load_h5_as_tensor('LEN128-DIM64-CUB16-Tvir4.8-zeta131.341-0812-103813.h5',num_image=num_image,num_redshift=num_redshift,dim=dim)

        x_pairs = [
                (x0, x0_ml),
                (x1, x1_ml),
                (x2, x2_ml),
                (x3, x3_ml),
                (x4, x4_ml),
                ]

        params = [
                c0[0],
                c1[0],
                c2[0],
                c3[0],
                c4[0],
                ]

        if 'grid' in what2plot:

            if x0.shape[-1] == 64:
                row, col = 4, 13
            elif x0.shape[-1] == 128:
                row, col = 8, 12
            elif x0.shape[-1] == 256:
                row, col = 8, 6
            elif x0.shape[-1] == 1024:
                row, col = 9, 2

            plot_grid(torch.cat((x0[:row//2 * col], x0_ml), dim=0), c=c0, los=los, savename = save_name, row=row, col=col)
            plot_grid(torch.cat((x1[:row//2 * col], x1_ml), dim=0), c=c1, los=los, savename = save_name, row=row, col=col)
            plot_grid(torch.cat((x2[:row//2 * col], x2_ml), dim=0), c=c2, los=los, savename = save_name, row=row, col=col)
            plot_grid(torch.cat((x3[:row//2 * col], x3_ml), dim=0), c=c3, los=los, savename = save_name, row=row, col=col)
            plot_grid(torch.cat((x4[:row//2 * col], x4_ml), dim=0), c=c4, los=los, savename = save_name, row=row, col=col)

        if 'global_signal' in what2plot:
            plot_global_signal(
                    x_pairs = x_pairs,
                    params = params,
                    los = los,
                    savename = save_name,
                    # sigma_level=100,
                    )

        if 'power_spectrum' in what2plot:
            plot_power_spectrum(
                    x_pairs = x_pairs,
                    params = params,
                    los = los,
                    savename = save_name,
                    # sigma_level=100,
                    )

        if 'scatter_transform' in what2plot:
            plot_scattering_transform_2(
                    x_pairs = x_pairs,
                    params = params,
                    los = los,
                    savename = save_name,
                    # sigma_level=100,
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobID", type=int, required=True)
    args = parser.parse_args()

    evaluate(
            what2plot = ['grid', 'global_signal', 'power_spectrum', 'scatter_transform'],
            device_count = 4,
            node = 8,
            jobID = args.jobID,
            epoch = 120,
            use_ema = 0,
            )
