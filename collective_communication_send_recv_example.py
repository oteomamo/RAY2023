import ray
import cupy
import ray.util.collective as col

@ray.remote(num_gpus=1)
class Sender:
    def __init__(self):
        print("Sender: Initializing...")
        self.data = cupy.array([1, 2, 3, 4], dtype=cupy.float32)
        col.init_collective_group(world_size=2, rank=0, backend="nccl", group_name="default")

    def send_data(self):
        print("Sender: Sending data...")
        col.send(self.data, dst_rank=1, group_name="default")
        return "Data sent!"

@ray.remote(num_gpus=1)
class Receiver:
    def __init__(self):
        print("Receiver: Initializing...")
        self.data = cupy.zeros((4,), dtype=cupy.float32)
        col.init_collective_group(world_size=2, rank=1, backend="nccl", group_name="default")

    def receive_data(self):
        print("Receiver: Receiving data...")
        col.recv(self.data, src_rank=0, group_name="default")
        return self.data

def main():
    ray.init(num_gpus=2)  # Ensure Ray is initialized with 2 GPUs

    # Check available resources
    print("Available resources:", ray.available_resources())

    # Create actors
    sender = Sender.remote()
    receiver = Receiver.remote()

    # Send and receive data concurrently
    send_future = sender.send_data.remote()
    recv_future = receiver.receive_data.remote()

    # Wait for both tasks to complete
    done_refs, _ = ray.wait([send_future, recv_future], num_returns=2)

    send_status = ray.get(done_refs[0])
    received_data = ray.get(done_refs[1])

    print(send_status)
    print("Received data:", received_data)

main()
