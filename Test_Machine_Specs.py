import ray
import subprocess

def get_ray_version():
    return ray.__version__

def get_cuda_version():
    # Method 1: Using nvcc
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        return output.strip().split('\n')[-1].split(' ')[-1]
    except:
        pass

    # Method 2: Checking cuda/version.txt
    try:
        with open("/usr/local/cuda/version.txt", "r") as f:
            return f.read().strip().split(" ")[-1]
    except:
        pass

    return "Could not determine CUDA version"

def get_available_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode("utf-8")
        return output.strip().split('\n')
    except:
        return "Could not determine available GPUs"
        
def get_ray_resources():
    return ray.available_resources()
def main():
    # Print the environment specifications
    print("""
    Test Environment Specifications:
    - Ray version: 2.7.0
    - CUDA version: 10.2
    - Available GPUs: 
      - Tesla V100S-PCIE-32GB
      - Tesla V100S-PCIE-32GB
    - Ray available resources:
      - node:__internal_head__: 1.0
      - node:10.10.15.113: 1.0
      - accelerator_type:V100S: 1.0
      - CPU: 36.0
      - object_store_memory: 51512981913.0
      - memory: 110196957799.0
      - GPU: 2.0
    """)

    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=2)

    ray_version = get_ray_version()
    cuda_version = get_cuda_version()
    available_gpus = get_available_gpus()
    ray_resources = get_ray_resources()

    # Print the environment specifications
    print("Test Environment Specifications:")
    print(f"- Ray version: {ray_version}")
    print(f"- CUDA version: {cuda_version}")
    print("- Available GPUs:")
    for gpu in available_gpus:
        print(f"  - GPU {gpu}")
    print("- Ray available resources:")
    for key, value in ray_resources.items():
        print(f"  - {key}: {value}")

    ray.shutdown()
if __name__ == "__main__":
    main()
