import ray
import ray.util.collective as collective
import cupy as cp

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.send = cp.ones((4, ), dtype=cp.float32)
        self.recv = cp.zeros((4, ), dtype=cp.float32)

    def setup(self, world_size, rank, group_name):
        collective.init_collective_group(world_size, rank, "nccl", group_name)
        return True

    def compute(self, group_name):
        collective.allreduce(self.send, group_name)
        return self.send

    def destroy(self, group_name):
        collective.destroy_collective_group(group_name)

ray.init()  # Ensure Ray is initialized

# imperative
num_workers = 2
workers = []
init_rets = []
group_name = "default"
for i in range(num_workers):
    w = Worker.remote()
    workers.append(w)
    init_rets.append(w.setup.remote(num_workers, i, group_name))
_ = ray.get(init_rets)
results = ray.get([w.compute.remote(group_name) for w in workers])

# declarative
workers = []  # Resetting the workers list for the declarative approach
for i in range(num_workers):
    w = Worker.remote()
    workers.append(w)

_options = {
    "group_name": "177",
    "world_size": 2,
    "ranks": [0, 1],
    "backend": "nccl"
}
collective.create_collective_group(workers, **_options)
results = ray.get([w.compute.remote(_options["group_name"]) for w in workers])

