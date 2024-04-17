import GPUtil
from time import sleep

while True:
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU: {gpu.name}, GPU Util: {gpu.load*100}%, GPU Mem Util: {gpu.memoryUtil*100}%")
    sleep(1)
