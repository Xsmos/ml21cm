# %%
# Packages 
import warnings
warnings.simplefilter('ignore')

import sys, os

# import py21cmfast as p21c

# default_cache_direc = p21c.config['direc']
# # if rank == 0:
# #     if os.path.exists(default_cache_direc) and not os.path.exists(default_cache_direc/'wisdoms'):
# if not os.path.exists(default_cache_direc/'wisdoms') and rank == 0:
#     os.mkdir(default_cache_direc/'wisdoms')
#     print("wisdoms created !!!!!!!!!!!!!!!!!!")

# print("wisdoms created")
import shutil

import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import glob
import h5py
import fcntl
import time
from time import sleep

# Parallize
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1

# py21cmfast will create ~/21cmFAST-cache/wisdoms automatically.
# To avoid conflicts between processes, it's necessary to do:
default_cache_direc = None
if rank == 0:
    import py21cmfast as p21c
    default_cache_direc = p21c.config['direc']
    if not os.path.exists(default_cache_direc/'wisdoms'):
        os.mkdir(default_cache_direc/'wisdoms')

# print("'comm' in globals():", 'comm' in globals(), rank)
if 'comm' in globals():
    default_cache_direc = comm.bcast(default_cache_direc, root=0)
    import py21cmfast as p21c
# print(rank, "default_cache_direc =", default_cache_direc)

# if rank 
    # while True:
    #     if os.path.exists(default_cache_direc/'wisdoms'):
    #         import py21cmfast as p21c
    #         break
    #     else:
    #         sleep(0.1)

str_pad_len = 80
str_pad_type = '-'
cache_direc = "_cache" + str(rank)

class Generator():
    def __init__(self, params_ranges, **kwargs):
        """
        Generate dataset by 21cmFAST in parallel.
        Input: params_ranges = {'param1': [min, max], 'param2': [min, max], ...}
        Output: hdf5 storing images and params.
        """
        self.params_ranges = params_ranges.copy()
        # self.kwargs['num_images'] = num_images
        self.define_kwargs(kwargs)

        # print parameters information
        if rank == 0:
            self.print_kwargs_params()

        #normalized_params = self.sample_normalized_params()
        #self.denormalize(normalized_params)
        # self.define_default_params()

    @property
    def params_ranges(self):
        if not hasattr(self, '_params_ranges'):
            self._params_ranges = "Error."
        return self._params_ranges

    @params_ranges.setter
    def params_ranges(self, value):
        self._params_ranges = value
        for key, value in self._params_ranges.items():
            if type(value) != list:
                self._params_ranges[key] = [value]

    def print_kwargs_params(self):
        if self.kwargs['verbose'] >= 1:
            print(f" Mission: Generate {self.kwargs['num_images']} images by {size}*{len(os.sched_getaffinity(os.getpid()))} CPUs ".center(str_pad_len, '#'))#, str_pad_type))
            print(f" params: ".center(int(str_pad_len/2),str_pad_type)+f" ranges: ".center(int(str_pad_len/2),str_pad_type))
            for key in self.params_ranges:
                print(f"{key}".center(int(str_pad_len/2))+f"[{self.params_ranges[key][0]}, {self.params_ranges[key][-1]}]".center(int(str_pad_len/2)))
            # print(f"params_ranges = {self.params_ranges}".center(str_pad_len, str_pad_type))
        
        if self.kwargs['verbose'] >= 2:
            # print(f"**kwargs will be passed to qmc.LatinHypercube".center(str_pad_len, str_pad_type))
            print(f" kwargs: ".center(int(str_pad_len/2), str_pad_type)+f" values: ".center(int(str_pad_len/2),str_pad_type))
            
            for key in self.kwargs:
                print(f"{key}".center(int(str_pad_len/2))+f"{self.kwargs[key]}".center(int(str_pad_len/2)))


    def define_kwargs(self, kwargs):
        self.kwargs = dict(
            # local params for Generator.__init__()
            p21c_run = 'lightcone',
            num_images = 9,
            fields = ['brightness_temp',],
            verbose = 2,
            seed = None,
            # cache_direc = "_cache",

            # strength param of scipy.stats.qmc.LatinHypercube():
            strength = 1,
            
            # redshift param of py21cmfast.run_coeval():
            redshift = [8,9,10],
            max_redshift = 20,
            
            # user_params of py21cmfast.run_coeval():
            HII_DIM = 60, 
            BOX_LEN = 150,
            USE_INTERPOLATION_TABLES = True,
            
            # cosmo_params of py21cmfast.run_coeval():
            SIGMA_8 = 0.810,
            hlittle = 0.677,
            OMm = 0.310,
            OMb = 0.0490,
            POWER_INDEX = 0.967,
        )

        # update
        # for key in kwargs:
        #     self.kwargs[key] = kwargs[key]
        self.kwargs = self.kwargs | kwargs

        if self.kwargs['p21c_run'] == 'coeval':
            if type(self.kwargs['redshift']) != list:
                self.kwargs['redshift'] = [self.kwargs['redshift']]
        elif self.kwargs['p21c_run'] == 'lightcone':
            if type(self.kwargs['redshift']) == list:
                self.kwargs['redshift'] = self.kwargs['redshift'][0]
        else:
            raise TypeError(f"'p21c_run' must be either 'coeval' or 'lightcone', instead of '{self.kwargs['p21c_run']}'.")

        if type(self.kwargs['fields']) != list:
            self.kwargs['fields'] = [self.kwargs['fields']]
 
        if self.kwargs['num_images'] < size:
            if self.kwargs['verbose'] > 0: print(f"num_images {self.kwargs['num_images']} must be >= the number of nodes {size}.")
            self.kwargs['num_images'] = size

        if not os.path.exists(cache_direc):
            os.mkdir(cache_direc)
        p21c.config['direc'] = cache_direc

    def sample_normalized_params(self):
        """
        sample and scatter to other nodes
        """
        # dimension=len(self.params_ranges), num_images=self.kwargs['num_images']
        np.random.seed(self.kwargs['seed'])
        if rank == 0:
            # np.random.seed(self.kwargs['seed'])
            sampler = qmc.LatinHypercube(d=len(self.params_ranges), strength=self.kwargs['strength'], seed=np.random.default_rng(self.kwargs['seed']))
            sample = sampler.random(n=self.kwargs['num_images'])
            #send_data = sample[:int(sample.shape[0]//size * size),:]
            #send_data = send_data.reshape(size, int(send_data.shape[0]/size), send_data.shape[1])
            send_data = np.array_split(sample, size, axis=0)

            if self.kwargs['verbose'] >= 2:
                print(f" Process {rank} scatters data {sample.shape} to {size} nodes ".center(str_pad_len,str_pad_type))
        else:
            send_data = None
        
        # print("'comm' in globals():", 'comm' in globals(), rank)
        if 'comm' in globals():
            recv_data = comm.scatter(send_data, root=0)
        else:
            recv_data = send_data
        #if self.kwargs['verbose'] >= 2:
        #    print(f"Process {rank} recvs data {recv_data.shape}".center(str_pad_len))#, str_pad_type))

        return recv_data


    def denormalize(self, normalized_data):
        """
        denormalize data received, and return self.params_node which stores params for each node.
        """
        self.params_node = {}
        for i, kind in enumerate(self.params_ranges):
            x = normalized_data.T[i]
            k = self.params_ranges[kind][-1]-self.params_ranges[kind][0]
            b = self.params_ranges[kind][0]
            self.params_node[kind] = k*x + b


    def return_coeval_or_lightcone(self, kwargs_params_cpu, random_seed):
        if self.kwargs['p21c_run'] == 'coeval':
            coevals_cpu = p21c.run_coeval(
                redshift = kwargs_params_cpu['redshift'],
                user_params = kwargs_params_cpu,
                cosmo_params = p21c.CosmoParams(kwargs_params_cpu),
                astro_params = p21c.AstroParams(kwargs_params_cpu),
                random_seed = random_seed,
            )
            dict_cpu = self.coevals2dict(coevals_cpu)

        elif self.kwargs['p21c_run'] == 'lightcone':
            lightcone_cpu = p21c.run_lightcone(
                redshift = kwargs_params_cpu['redshift'],
                max_redshift = kwargs_params_cpu['max_redshift'],
                lightcone_quantities = kwargs_params_cpu['fields'],
                user_params = kwargs_params_cpu,
                cosmo_params = p21c.CosmoParams(kwargs_params_cpu),
                astro_params = p21c.AstroParams(kwargs_params_cpu),
                random_seed = random_seed,
            )
            self.kwargs['node_redshifts'] = lightcone_cpu.node_redshifts
            print(rank, "self.kwargs['node_redshifts'][0] =", self.kwargs['node_redshifts'][0])
            dict_cpu = self.lightcone2dict(lightcone_cpu)
        
        return dict_cpu


    def pool_run(self, params_node_value):
        # All parameters
        pool_run_start = time.perf_counter()

        pid_cpu = multiprocessing.current_process().pid

        random_seed = int(params_node_value[-1])
        params_cpu = {key: params_node_value[i] for (i, key) in enumerate(self.params_node.keys())}
        # self.update_params()

        # concantenate parameters and kwargs
        kwargs_params_cpu = self.kwargs | params_cpu

        # Simulation
        # coevals_cpu = p21c.run_coeval(
        #     redshift = kwargs_params_cpu['redshift'],
        #     user_params = kwargs_params_cpu,
        #     cosmo_params = p21c.CosmoParams(kwargs_params_cpu),
        #     astro_params = p21c.AstroParams(kwargs_params_cpu),
        #     random_seed = random_seed
        # )
        # dict_cpu = self.coevals2dict(coevals_cpu)
        dict_cpu = self.return_coeval_or_lightcone(kwargs_params_cpu,random_seed)

        # Clear cache
        cache_pattern = os.path.join(cache_direc, f"*r{random_seed}.h5")
        for filename in glob.glob(cache_pattern):
            # print(filename)
            os.remove(filename)
        #if len(os.listdir(cache_direc)) == 0:
        #    os.rmdir(cache_direc)

        pool_run_end = time.perf_counter()
        
        time_elapsed = time.strftime("%M:%S", time.gmtime(pool_run_end - pool_run_start))
        # time_elapsed = pool_run_end - pool_run_start
        
        #print("verbose =", self.kwargs['verbose']) 
        if self.kwargs['verbose'] > 2:
            print(f'{time_elapsed}, cpu {pid_cpu}-{rank}, params {list(params_cpu.values())}, seed {random_seed}')
        #print("os.getpid", os.getpid(), "multiprocessing.parent_process().pid", multiprocessing.parent_process().pid, "multiprocessing.current_process().pid", multiprocessing.current_process().pid)

        return dict_cpu

    def lightcone2dict(self, lightcone_cpu):
        images_cpu = {}
        for i, field in enumerate(self.kwargs['fields']):
            images_cpu[field] = lightcone_cpu.lightcones[field]
            # for j, coeval in enumerate(coevals_cpu):
                # images_cpu[field].append(coeval.__dict__[field])
        # print(images_cpu.keys())
        # print(images_cpu.values())
        return images_cpu
        

    def coevals2dict(self, coevals_cpu):
        images_cpu = {}
        for i, field in enumerate(self.kwargs['fields']):
            images_cpu[field] = []
            for j, coeval in enumerate(coevals_cpu):
                images_cpu[field].append(coeval.__dict__[field])
        # print(images_cpu.keys())
        # print(images_cpu.values())
        return images_cpu

    def clear_cache_direc(self):
        # print(cache_direc, "starts")
        if len(os.listdir(cache_direc)) == 0:
            os.rmdir(cache_direc)
        # print(cache_direc, "ends")
        
        # node 0 waits all processes finishing to remove default_cache_direc
        # print("'comm' in globals():", 'comm' in globals(), rank)

        if 'comm' in globals():
            recv_data = comm.gather(rank, root=0)
        
        if rank == 0:
            if len(os.listdir(default_cache_direc)) == 1:
                shutil.rmtree(default_cache_direc)
            # print("clear_cache_direc starts")
            # if os.path.exists(default_cache_direc):
            #     if len(os.listdir(default_cache_direc/'wisdoms')) == 0:
            #         shutil.rmtree(default_cache_direc)
            #         print("removed default_cache_direc!!!!!!!!!!!!!!!!!")
            # print("clear_cache_direc ends")


    def run(self, save_direc_name='images_params.h5'):
        #if rank == 0:
        normalized_params = self.sample_normalized_params()
        self.denormalize(normalized_params)

        pid_node = os.getpid()
        CPU_num = len(os.sched_getaffinity(pid_node))
        
        if self.kwargs['verbose'] >= 2:
            print(f" node {rank}: {CPU_num} CPUs, params.shape {np.array(list(self.params_node.values())).T.shape} ".center(str_pad_len,str_pad_type))

        # run p21c.run_coeval in parallel on multi-CPUs
        Pool_start = time.perf_counter()

        with Pool(CPU_num) as p:
            iterable = np.array(list(self.params_node.values()))
            #print(iterable)
            # np.random.seed(self.kwargs['seed'])
            random_seeds = np.random.randint(1,2**32, size = iterable.shape[-1])
            iterable = np.vstack((iterable, random_seeds)).T
            #print(iterable)
            # images_node = np.array(p.map(self.pool_run, iterable))
            dict_node = p.map(self.pool_run, iterable)
        
        images_node, images_node_MB = self.dict2images(dict_node)

        Pool_end = time.perf_counter()
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(Pool_end - Pool_start))

	# # gather images_node from different node
    #     images_all = {}
    #     for field in self.kwargs['fields']:
    #         images_all[field] = comm.gather(images_node[field], root=0)
    #     #print("images_all:", images_all)
    #     params_seeds = comm.gather(iterable, root=0)
    #     #print("params_seeds:", params_seeds)

    #     if rank == 0:
    #         for i, field in enumerate(self.kwargs['fields']):
    #             #print("images_all[field]", images_all[field])
    #             images_all[field] = np.concatenate(images_all[field], axis=0)
    #         params_seeds = np.concatenate(params_seeds, axis=0)
    #         self.save(images_all, params_seeds, save_direc_name)
            
        # save images, params as .h5 file
        async_save_time = self.async_save(images_node, iterable, save_direc_name)

        if self.kwargs['verbose'] >= 1:
            print(f"{time_elapsed}, node {rank}: {images_node_MB} MB images {[np.shape(images) for images in images_node.values()]} ->{async_save_time}-> {os.path.basename(save_direc_name)}".center(str_pad_len))
        
        self.clear_cache_direc()
        #print("os.getpid", os.getpid(), "multiprocessing.current_process().pid", multiprocessing.current_process().pid)
        #return images_node, self.params_node, rank

    # def dict2images_backup(self, dict_node):
    #     #print("dict_node len =", len(dict_node))
    #     images_node = []
    #     images_node_MB = []
    #     for field in self.kwargs['fields']:
    #         images = []
    #         for dict_cpu in dict_node:
    #             images.append(dict_cpu[field])
    #         images = np.array(images)
    #         print("images.shape:", images.shape)
    #         images_node_nbytes.append(images.nbytes)
    #         images_node.append(images) 
    #     return images_node, images_node_nbytes

    def dict2images(self, dict_node):
        images_node = {}
        images_node_MB = []
        for field in self.kwargs['fields']:
            images_node[field] = []
            for dict_cpu in dict_node:
                images_node[field].append(dict_cpu[field])
            images_node[field] = np.array(images_node[field])
            images_node_MB.append(round(images_node[field].nbytes / 1024**2))

        return images_node, images_node_MB

    def async_save(self, images_node, params_seeds, save_direc_name):
        try_start = time.perf_counter()
        while True:
            try:
                save_start = time.perf_counter()
                self.save(images_node, params_seeds, save_direc_name)
                save_end = time.perf_counter()
                # time_elapsed = time.strftime("%H:%M:%S", time.gmtime(try_end - try_start))
                save_time = save_end - save_start
                try_time = save_start - try_start
                return f"{try_time:.1f}s/{save_time:.2f}s"
                # break
            except BlockingIOError:
                # print(f"Hold on, {rank}/{size} wait a minute ...")
                sleep(0.1)

    # Save as hdf5
    def save(self, images_node, params_seeds, save_direc_name):
        # if os.path.exists(save_direc_name):
        #     os.remove(save_direc_name)
        # HII_DIM = images.shape[-1]
        # self.images_node_nbytes = 0
        with h5py.File(save_direc_name, 'a') as f:
        #f = h5py.File(save_direc_name, 'a')
        #try:
        #    fd = f.id.get_vfd_handle()
        #    fcntl.flock(fd, fcntl.LOCK_EX)
            if 'kwargs' not in f.keys():
                #grp = f.create_group('kwargs')
                print("self.kwargs =", self.kwargs)
                keys = list(self.kwargs)
                values = [str(value) for value in self.kwargs.values()]
                data = np.transpose(list((keys, values)))
                data = data.tolist()
                f.create_dataset('kwargs', data=data)

            if 'params_seeds' not in f.keys():
                grp = f.create_group('params_seeds') 
                grp['keys'] = list(self.params_ranges) + ['seed']
                grp.create_dataset(
                    'values',
                    data = params_seeds,
                    maxshape = tuple((None,) + params_seeds.shape[1:]),
                    )
            else:
                new_size = f['params_seeds']['values'].shape[0] + params_seeds.shape[0]
                f['params_seeds']['values'].resize(new_size, axis=0)
                f['params_seeds']['values'][-params_seeds.shape[0]:] = params_seeds

            for field in self.kwargs['fields']:
                images = images_node[field]
                if field not in f.keys():
                    f.create_dataset(
                        field, 
                        data=images, 
                        maxshape= tuple((None,) + images.shape[1:])
                    )
                else:
                    # print(image.shape)
                    new_size = f[field].shape[0] + images.shape[0]
                    f[field].resize(new_size, axis=0)
                    f[field][-images.shape[0]:] = images
        #finally:
        #    fcntl.flock(fd, fcntl.LOCK_UN)
        #    f.close()

if __name__ == '__main__':
    # training set, (25600, 64, 64, 64)
    save_direc = "/storage/home/hcoda1/3/bxia34/scratch/"

    params_ranges = dict(
        ION_Tvir_MIN = [4,6],
        HII_EFF_FACTOR = [10, 250],
        )
    kwargs = dict(
        p21c_run = 'coeval',
        seed = 1, fields = 'brightness_temp',
        HII_DIM=64, BOX_LEN=100,
        verbose=3, redshift=[8,10]
        )
    generator = Generator(params_ranges, num_images=32, **kwargs)
    generator.run(save_direc_name=os.path.join(save_direc, "train.h5"))

    # testing set, (5*800, 64, 64, 64)
    params_list = [(4.4,131.341),(5.6,19.037)]#, (4.699,30), (5.477,200), (4.8,131.341)]

    kwargs = dict(
        # p21c_run = 'coeval',
        fields = ['brightness_temp', 'density'],
        HII_DIM=64, BOX_LEN=60,
        verbose=3, redshift=9,
        max_redshift = 15,
        num_images=16,
        )
    for T_vir, zeta in params_list:
        params_ranges = dict(
        ION_Tvir_MIN = T_vir, # single number is ok,
        HII_EFF_FACTOR = [zeta], # list of single number is ok,
        )
        generator = Generator(params_ranges, **kwargs)
        generator.run(save_direc_name=os.path.join(save_direc,"test.h5"))
