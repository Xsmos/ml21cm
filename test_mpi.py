from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# recv_data = None

if rank == 0:
    send_data = range(2)
    print(f"process {rank} scatter data {send_data} to other process.")
else:
    send_data = None

recv_data = comm.scatter(send_data, root=0)
print(f"Process {rank} recv data {recv_data}...")
