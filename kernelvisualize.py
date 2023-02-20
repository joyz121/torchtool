from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from pathlib import Path
import torch.nn as nn
import torch
import os
from model import Net
kernel_num = -1
vis_max = 1
FILE = Path(__file__).resolve() #current directory
ROOT = FILE.parents[0]  #root directory
log_dir = os.path.join(ROOT, "results")
writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")
model_dir= os.path.join(ROOT, 'model.pt')
model=torch.load(model_dir)
model.eval()
#可视化模型
# 取前两层卷积核
for sub_module in model.modules():
    if not isinstance(sub_module, nn.Conv2d):
        continue
    if kernel_num >= vis_max:
        break
    kernel_num += 1
    kernels = sub_module.weight
    c_out, c_int, k_h, k_w = tuple(kernels.shape)  # 输出通道数,输入通道数,卷积核宽,卷积核高
    print(kernels.shape)
    for o_idx in range(c_out):
        kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # 获得(3, h, w), 但是make_grid需要 BCHW，这里拓展C维度变为（3， 1， h, w）
        kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=8)  # 将卷积核于网格中可视化
        # nrow:每一行显示的图像数
        writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)
    #     名称，图片，第几张图片
    kernel_all = kernels.view(-1, 1, k_h, k_w)  # 3, h, w
    kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
    writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=620)
    print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))
