import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("当前设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import pycuda.driver as cuda
import pycuda.autoinit

# 获取GPU数量
device_count = cuda.Device.count()
print("GPU数量:", device_count)

# 显示每个GPU的信息
for i in range(device_count):
    device = cuda.Device(i)
    print(f"GPU {i}: {device.name()}")
    print("计算能力:", device.compute_capability())
    print("总内存:", device.total_memory() / 1024**3, "GB")

import tensorflow as tf

# 检查GPU是否可用
print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", tf.config.list_physical_devices('GPU'))

# 显示GPU信息
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU名称:", gpu.name)
