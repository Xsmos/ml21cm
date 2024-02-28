# Packages 
# import warnings
# warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import glob
import h5py
import time
from datetime import timedelta

# We change the default level of the logger so that
# we can see what's happening with caching.
import sys, os
import logging
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c

# For interacting with the cache
from py21cmfast import cache_tools

# Cache for intermediate process
cache_direc = "/storage/home/hcoda1/3/bxia34/scratch/_cache"

if not os.path.exists(cache_direc):
    os.mkdir(cache_direc)

p21c.config['direc'] = cache_direc

str_pad_len = 80
str_pad_type = '-'

# Parallize
import multiprocessing
from multiprocessing import Pool
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size =1

class Dataset():
    def __init__(self, **kwargs):
        """
        Generate dataset by 21cmFAST in parallel.
        Input: kwargs = {'param1': [min, max], 'param2': [min, max], ...}
        Output: hdf5 storing images and params.
        """
        self.kwargs = kwargs
        print(f"kwargs = {self.kwargs}".center(str_pad_len, str_pad_type))

        self.sample_normalized_params(dimension=len(self.kwargs), num_groups=19)
        self.denormalize()
        self.define_default_params()

    def sample_normalized_params(self, dimension=2, num_groups=9):
        """
        sample and scatter to other nodes
        """
        if rank == 0:
            sampler = qmc.LatinHypercube(d=dimension, strength=2)
            sample = sampler.random(n=num_groups)
            send_data = sample[:int(sample.shape[0]//size * size),:]
            send_data = send_data.reshape(size, int(send_data.shape[0]/size), send_data.shape[1])
            print(f"Process {rank} scatters data (shape = {send_data.shape}) to {size} nodes".center(str_pad_len,str_pad_type))
        else:
            send_data = None
        self.recv_data = comm.scatter(send_data, root=0)
        print(f"Process {rank}/{size} recvs data (shape = {self.recv_data.shape})".center(str_pad_len, str_pad_type))

    def denormalize(self):
        """
        denormalize data received, and return self.params_node which stores params for each node.
        """
        self.params_node = {}
        for i, kind in enumerate(self.kwargs):
            x = self.recv_data.T[i]
            k = self.kwargs[kind][1]-self.kwargs[kind][0]
            b = self.kwargs[kind][0]
            self.params_node[kind] = k*x + b

    def define_default_params(self, params: dict) -> None:
        self.redshift = 11.93 
        self.user_params = {
            "HII_DIM":60, 
            "BOX_LEN":150, 
            # "USE_INTERPOLATION_TABLE":True
            }
        self.cosmo_params = dict(
            SIGMA_8 = 0.810,
            hlittle = 0.677,
            OMm = 0.310,
            OMb = 0.0490,
            POWER_INDEX = 0.967,
            )
        self.astro_params = dict(
            ION_Tvir_MIN = 5,#params['ION_Tvir_MIN'],
            HII_EFF_FACTOR = 100,#params['HII_EFF_FACTOR'],
            )

    def update_params(self):
        params_list = ["user_params", "cosmo_params", "astro_params"]
        for params in params_list:
            for key in params_cpu:
                if key is in self.__dict__[params]:
                    self.__dict__[params][key] = self.params_cpu[key]


    def generate_brightness_temp(self, params_node_value):
        # All parameters
        generate_brightness_temp_start = time.perf_counter()

        pid_cpu = multiprocessing.current_process().pid
        self.random_seed = np.random.randint(1,2**32) + pid_cpu

        self.params_cpu = {key: params_node_value[i] for (i, key) in enumerate(params_node.keys())}
        self.update_params()

        # Simulation
        coeval = p21c.run_coeval(
            redshift = self.redshift,
            user_params = self.user_params,
            cosmo_params = p21c.CosmoParams(self.cosmo_params),
            astro_params = p21c.AstroParams(self.astro_params),
            random_seed = self.random_seed
        )

        cache_pattern = os.path.join(cache_direc, f"*{coeval.random_seed}*")
        for filename in glob.glob(cache_pattern):
            # print(filename)
            os.remove(filename)

        generate_brightness_temp_end = time.perf_counter()
        time_elapsed = generate_brightness_temp_end - generate_brightness_temp_start
        print(f'cpu {pid_cpu} in {pid_node}, seed {self.random_seed}, {params_cpu}, cost {timedelta(seconds=time_elapsed)}')
        
        return coeval.brightness_temp

    def run_parallel(self):
        pid_node = os.getpid()
        CPU_num = len(os.sched_getaffinity(pid_node))
        print(f"node {pid_node}: {CPU_num} CPUs are working on {np.shape(list(params_node.values()))[-1]} groups of params".center(str_pad_len,str_pad_type))

        # run p21c.run_coeval in parallel on multi-CPUs
        Pool_start = time.perf_counter()
        with Pool(CPU_num) as p:
            images_node = np.array(p.map(generate_brightness_temp, np.array(list(params_node.values())).T))
        Pool_end = time.perf_counter()
        time_elapsed = Pool_end - Pool_start
        print(f"images {self.images_node.shape} generated by node {pid_node} with {timedelta(seconds=time_elapsed)}".center(str_pad_len,str_pad_type))

        # save images, params as .h5 file
        self.save(images_node, np.array(list(params_node.values())).T, 'test_save_func.h5')

    # Save as hdf5
    def save(images, params, save_direc_name="./images_params.h5"):
        # if os.path.exists(save_direc_name):
        #     os.remove(save_direc_name)
        HII_DIM = images.shape[-1]
        with h5py.File(save_direc_name, 'a') as f:
            if 'images' not in f.keys():
                f.create_dataset(
                    'images', 
                    data=images, 
                    maxshape=(None, HII_DIM, HII_DIM, HII_DIM)
                )
                f.create_dataset(
                    'params',
                    data = params,
                    maxshape = (None, params.shape[-1]))
                # f.create_dataset(
                #     'random_seed',
                #     data=random_seed,
                #     maxshape=(None,1)
                # )
            else:
                # print(image.shape)
                new_size = f['images'].shape[0] + images.shape[0]
                f['images'].resize(new_size, axis=0)
                f['images'][-images.shape[0]:] = images
                f['params'].resize(new_size, axis=0)
                f['params'][-images.shape[0]:] = params
                # f['random_seed'].resize(new_size, axis=0)
                # f['random_seed'][-1] = random_seed