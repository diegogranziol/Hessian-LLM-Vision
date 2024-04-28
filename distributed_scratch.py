import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def ddp_gpu_memory_usage(rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Retrieve and print memory usage information for the current GPU
    device = torch.device(f"cuda:{rank}")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory - (cached_memory - allocated_memory)

    print(f"Rank {rank}, Total GPU Memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Rank {rank}, Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Rank {rank}, Cached Memory: {cached_memory / (1024 ** 3):.2f} GB")
    print(f"Rank {rank}, Free Memory: {free_memory / (1024 ** 3):.2f} GB")

    # Clean up the distributed environment
    dist.destroy_process_group()


def main():
    world_size = 8  # Number of GPUs
    mp.spawn(ddp_gpu_memory_usage, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
