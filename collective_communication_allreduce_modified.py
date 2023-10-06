import ray
import cupy
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.buffer = cupy.ones((10,), dtype=cupy.float32)

    def setup(self, world_size, rank):
        col.init_collective_group(world_size, rank, "nccl", "default")
        return True

    def compute(self):
        col.allreduce(self.buffer, "default")
        return self.buffer

ray.init()  # Ensure Ray is initialized

# Create two actors A and B
A = Worker.remote()
B = Worker.remote()

# Initialize the collective group for both actors
world_size = 2
_ = ray.get([A.setup.remote(world_size, 0), B.setup.remote(world_size, 1)])

# Invoke allreduce remotely
results = ray.get([A.compute.remote(), B.compute.remote()])
print(results)
