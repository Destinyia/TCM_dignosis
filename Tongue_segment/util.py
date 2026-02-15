import os
import cv2
import torch
import numpy as np
import torchvision
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metrics(object):
    def __init__(self, tag, log_dir, keys, writer):
        self.tag = tag
        self.keys = keys
        self.data = {k:AverageMeter() for k in keys}
        self.writer = writer
        with open(os.path.join(log_dir, '%s.csv'%tag), 'w') as f:
            f.write(','.join(['epoch'] + keys) + '\n')
        
    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self, key):
        return self.data.get(key, AverageMeter()).avg

    def reset(self):
        for k in self.keys:
            self.data[k].reset()
    
    def update(self, items, n=1):
        for k, v in items.items():
            self.data[k].update(v, n)
    
    def overwrite(self, items, n=1):
        for k, v in items.items():
            self.data[k].reset()
            self.data[k].update(v, n)
            
    def write(self, step=0):
        for k in self.keys:
            self.writer.add_scalar(tag="%s/%s" % (self.tag, k), 
                                   scalar_value=self.data[k].avg, 
                                   global_step=step)

def save_seg_images(images, masks, outputs, pred_mask, file_path):
    B = images.shape[0]//4
    plt.figure(figsize=(16, B))
    for i in range(B):
        for j in range(4):
            plt.subplot(B,16,i*16+j+1)
            plt.yticks([])
            plt.imshow(images[i*4+j].permute(1, 2, 0).numpy())#.astype(np.float32)
            plt.subplot(B,16,i*16+j+5)
            plt.yticks([])
            plt.imshow(masks[i*4+j].permute(1, 2, 0).numpy())
            plt.subplot(B,16,i*16+j+9)
            plt.yticks([])
            plt.imshow(pred_mask[i*4+j].permute(1, 2, 0).numpy())
            plt.subplot(B,16,i*16+j+13)
            plt.yticks([])
            plt.imshow(outputs[i*4+j].permute(1, 2, 0).detach().numpy())
            plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def tensor_to_images(tensor):
    """
    将模型的输出Tensor转换为图像列表。
    参数:
    - tensor (torch.Tensor): 应该是一个四维的Tensor，例如[batch_size, channels, height, width]
    
    返回:
    - images (list): 包含PIL图像的列表
    """
    images = tensor.detach().cpu()  # 确保tensor在CPU上，并且无需梯度计算
    images = images.permute(0, 2, 3, 1)  # 重排维度为[batch_size, height, width, channels]
    images = [img.numpy() for img in images]  # 转换为numpy数组
    return images

def show_images(images, cols=4):
    """
    使用matplotlib显示图像列表。
    参数:
    - images (list): 图像列表
    - cols (int): 每行显示的图像数量
    """
    rows = (len(images) + cols - 1) // cols  # 计算需要多少行
    fig = plt.figure(figsize=(cols * 10, rows * 10))  # 创建一个足够大的图形
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image, interpolation='nearest')  # 显示图像
        ax.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()

def save_model(model, name, epoch, exp_path='runs/exp', intervals=20):
    save_epoch = epoch - epoch % intervals
    save_epoch = f'{name}_epoch{save_epoch}-{save_epoch+intervals-1}.pt'
    torch.save(model.state_dict(), os.path.join(exp_path, save_epoch))

def load_model_weights(model, weight_path):
    try:
        # 加载权重文件
        state_dict = torch.load(weight_path)
        result = model.load_state_dict(state_dict, strict=False)
        
        # 检查未加载和多余的键
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
        
        if missing_keys:
            print("以下层未能加载权重：", missing_keys)
        if unexpected_keys:
            print("权重文件中包含未使用的键：", unexpected_keys)
        
    except Exception as e:
        print("加载权重时出错：", str(e))

# IoU计算函数
def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    if union == 0:
        return 1.0  # 如果没有前景物体，IoU 为 1.0
    else:
        return intersection / union

# Dice系数计算函数
def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    
    if total == 0:
        return 1.0  # 如果没有前景物体，Dice 为 1.0
    else:
        return (2 * intersection) / total
