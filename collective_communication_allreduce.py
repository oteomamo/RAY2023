import ray
import cupy
import ray.util.collective as col


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.buffer = cupy.ones((10,), dtype=cupy.float32)

    def compute(self):
        col.allreduce(self.buffer, "default")
        return self.buffer

# Create two actors A and B and create a collective group following the previous example...
A = Worker.remote()
B = Worker.remote()
# Invoke allreduce remotely
ray.get([A.compute.remote(), B.compute.remote()])
