"""

Does not work for NVIDIA-SMI 440.33.01 -- CUDA Version: 10.2  Ray version 2.7

"""


import ray
import cupy as cp
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data = cp.array([cp.random.randint(1000, 10000)], dtype=cp.float32)
        print(f"(Worker pid={ray.worker.global_worker.worker_id}) Rank {self.rank}: Generated random number {self.data}.")
        col.init_collective_group(world_size=self.world_size, rank=self.rank, backend="nccl", group_name="default")

    def reduce_scatter_data(self):
        result = cp.zeros((1,), dtype=cp.float32)
        print(f"(Worker pid={ray.worker.global_worker.worker_id}) Rank {self.rank}: Starting reduce_scatter operation...")
        col.reduce_scatter(result, [self.data], op="PRODUCT", group_name="default")
        return result

def main():
    ray.init()  # Ensure Ray is initialized
    print("Ray initialized.")
    world_size = 2
    print("Creating workers...")
    workers = [Worker.remote(rank, world_size) for rank in range(world_size)]
    print("Workers created.")
    print("Starting reduce_scatter operation across workers...")
    results = ray.get([worker.reduce_scatter_data.remote() for worker in workers])
    for rank, result in enumerate(results):
        print(f"Rank {rank} reduced_scatter result: {result}")

if __name__ == "__main__":
    main()
