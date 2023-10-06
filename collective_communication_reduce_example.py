import ray
import cupy
import os
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank):
        self.rank = rank
        # Initialize the data array with 10 numbers
        self.data = cupy.array([self.rank + i for i in range(10)], dtype=cupy.float32)
        print(f"(Worker pid={os.getpid()}) Rank {self.rank}: Initializing...")
        col.init_collective_group(world_size=2, rank=self.rank, backend="nccl", group_name="default")
        print(f"(Worker pid={os.getpid()}) Rank {self.rank}: Initialized data with values {self.data}.")

    def reduce_data(self):
        print(f"(Worker pid={os.getpid()}) Rank {self.rank}: Starting reduce operation...")
        col.reduce(self.data, dst_rank=0, group_name="default")
        print(f"(Worker pid={os.getpid()}) Rank {self.rank}: Completed reduce operation.")
        return self.data if self.rank == 0 else None

def main():
    ray.init()  
    print("Ray initialized.")

    # Create workers
    print("Creating workers...")
    workers = [Worker.remote(rank=i) for i in range(2)]
    print("Workers created.")

    # Perform reduce operation
    print("Starting reduce operation across workers...")
    reduced_data = ray.get([worker.reduce_data.remote() for worker in workers])

    # Print reduced data from rank 0
    print(f"Reduced data on Rank 0: {reduced_data[0]}")

if __name__ == "__main__":
    main()
