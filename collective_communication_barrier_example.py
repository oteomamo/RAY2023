import ray
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank):
        self.rank = rank
        self.group_name = "default"
        # Initialize collective group for 2 GPUs
        col.init_collective_group(world_size=2, rank=self.rank, backend="nccl")
        print(f"(Worker {self.rank}) Collective group initialized.")

    def run_barrier(self):
        print(f"(Worker {self.rank}) Before barrier.")
        col.barrier(group_name=self.group_name)
        print(f"(Worker {self.rank}) After barrier.")

def main():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")

    # Create 2 workers
    workers = [Worker.remote(i) for i in range(2)]
    print("Workers created.")

    # Run barrier operation across workers
    print("Starting barrier operation across workers...")
    _ = ray.get([worker.run_barrier.remote() for worker in workers])

    ray.shutdown()

if __name__ == "__main__":
    main()

