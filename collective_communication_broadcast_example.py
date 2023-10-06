import ray
import cupy
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size):
        self.rank = rank
        # Initialize data based on rank
        if self.rank == 0:
            self.data = cupy.array([1, 2, 3, 4], dtype=cupy.float32)
            print(f"Rank {self.rank}: Initialized data with values [1, 2, 3, 4].")
        else:
            self.data = cupy.zeros((4,), dtype=cupy.float32)
            print(f"Rank {self.rank}: Initialized data with zeros.")
        
        # Initialize the collective group
        col.init_collective_group(world_size=world_size, rank=self.rank, backend="nccl", group_name="default")
        print(f"Rank {self.rank}: Initialized collective group.")

    def broadcast_data(self):
        # Perform the broadcast operation
        col.broadcast(self.data, src_rank=0, group_name="default")
        print(f"Rank {self.rank}: Completed broadcast operation.")
        return self.data

def main():
    ray.init()  # Ensure Ray is initialized

    world_size = 2
    workers = [Worker.remote(rank, world_size) for rank in range(world_size)]

    # Broadcast data from rank 0 to all other ranks
    results = ray.get([worker.broadcast_data.remote() for worker in workers])

    for rank, data in enumerate(results):
        print(f"Rank {rank} data:", data)

if __name__ == "__main__":
    main()

