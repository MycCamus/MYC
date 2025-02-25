import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备编号
print(torch.cuda.is_available())  # 输出: False 或 True
print(torch.cuda.device_count())  # 输出: 0 或 GPU数量
print(os.environ['CUDA_VISIBLE_DEVICES'])  # 输出: None 或 GPU设备编号


# # 安全地访问环境变量
# cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
# print(cuda_visible_devices)  # 输出: None 或 GPU设备编号
