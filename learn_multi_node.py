import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Replace with master node's IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class MyDiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        # self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
        self.fc1 = torch.nn.Linear(32 * 6 * 6, 128)
        # self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        # x = torch.nn.functional.relu(self.conv2(x))
        # x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(rank, world_size):
    setup(rank, world_size)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Rank {rank}, Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"Rank {rank}, GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Rank {rank}, No GPUs available")

    cleanup()

if __name__ == "__main__":
    world_size = 1  # Number of nodes
    mp.spawn(main, args=(world_size,), nprocs=torch.cuda.device_count(), join=True)
