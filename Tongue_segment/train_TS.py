import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from models.unet import ResUNet, UNet, CBAMUNet, ResUNet1
from models.loss import DiceLoss
from data.TongueDataset import TongueDataset, TongueSegDataset, RandomTransform
from util import Metrics, save_seg_images, save_model, load_model_weights, calculate_iou, calculate_dice
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=5)

def test(teacher, student, 
         dataloader, device):
    teacher.eval()  # 设置模型为评估模式
    student.eval()
    iou_scores = []
    dice_scores = []
    
    progress_bar = tqdm(dataloader, desc=f'   test', leave=False)
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            images_teacher = downsample(images) # 400x400
            
            # 教师网络
            with torch.no_grad():
                teacher_output = teacher(images_teacher)  # [B, 1, 400, 400]
                # teacher_output_upsampled = upsample(teacher_output)  # [B, 1, 800, 800]
                teacher_masks = (upsample(teacher_output) > 0.5).float()
                
            # 前向传播
            outputs = student(images)
            pred_masks = (outputs > 0.5).float().squeeze(1)
            pred_masks = pred_masks.cpu().numpy()
            teacher_masks = teacher_masks.cpu().numpy()
            
            # 计算每个batch的IoU和Dice
            for pred_mask, teacher_mask in zip(pred_masks, teacher_masks):
                iou = calculate_iou(pred_mask, teacher_mask)
                dice = calculate_dice(pred_mask, teacher_mask)
                iou_scores.append(iou)
                dice_scores.append(dice)
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                'GPU': f"{memory_allocated:.0f} MB",
                'IoU': f"{iou:.5f}",
                'Dice': f"{dice:.5f}",
                })
    
    # 计算平均IoU和Dice
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    return mean_iou, mean_dice
    

def train(teacher, student, 
          train_loader, 
          val_loader, 
          optimizer, 
          num_epochs, 
          exp_path, 
          device):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
    os.mkdir(exp_path)
    writer = SummaryWriter(log_dir=exp_path)
    bce_loss_fn = nn.BCELoss()
    dice_loss_fn = DiceLoss()
    best_dice = 0
    
    teacher.eval()
    for epoch in range(1, num_epochs + 1):
        student.train()
        metrics = Metrics('train', exp_path,
                          keys=['BCE Loss', 'Dice Loss', 'Test IoU', 'Test Dice'], 
                          writer=writer)
        # running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            images_teacher = downsample(images) # 400x400

            # 教师网络
            with torch.no_grad():
                teacher_output = teacher(images_teacher)  # [B, 1, 400, 400]
                # teacher_output_upsampled = upsample(teacher_output)  # [B, 1, 800, 800]
                teacher_mask = (upsample(teacher_output) > 0.5).float().detach()
                
            # 前向传播
            optimizer.zero_grad()
            outputs = student(images)
            pred_mask = (outputs > 0.5).float()
            
            # 计算 Loss
            bce_loss = bce_loss_fn(outputs, teacher_mask)
            dice_loss = dice_loss_fn(outputs, teacher_mask)
            loss = bce_loss
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            metrics.update({'BCE Loss': bce_loss.item(),
                            'Dice Loss': dice_loss.item(),}, labels.size(0))
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({'GPU': f"{memory_allocated:.0f} MB",
                                      'BCE Loss': f"{bce_loss.item():.5f}",
                                      'Dice Loss': f"{dice_loss.item():.5f}"})
        
        # Epoch后
        test_iou, test_dice = test(teacher, student, val_loader, device)
        metrics.update({'Test IoU' : test_iou,
                        'Test Dice': test_dice})
        metrics.write(epoch)
        # 每个epoch结束后保存预测图/打印训练损失
        save_seg_images(images.cpu(), teacher_mask.cpu(), outputs.detach().cpu(), pred_mask.cpu(), os.path.join(exp_path, f'{epoch}.jpg'))
        save_model(student, 'unet', epoch, exp_path, intervals=10)
        # 保存最佳表现的模型
        if metrics['Test Dice'] > best_dice:
            best_dice = metrics['Test Dice']
            save_model(student, 'best', epoch, exp_path, intervals=1)
            
        print('Epoch: {}/{} | BCE loss: {:.5f} | Test IoU: {:.4f} | Test Dice: {:.4f}'.format(epoch, num_epochs,
                                                                                                metrics['BCE Loss'],
                                                                                                metrics['Test IoU'],
                                                                                                metrics['Test Dice'],
                                                                                                ))
        
def upsample(tensor, size=(640, 640)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

def downsample(tensor, size=(320, 320)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

if __name__ == '__main__':
    # 创建数据集实例
    train_set = TongueDataset('data/Tongue_train', transform=RandomTransform())
    test_set = TongueDataset('data/Tongue_test', transform=RandomTransform())
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    teacher = ResUNet1(layers=[2,2,2])
    load_model_weights(teacher, 'runs/res222_aspp04/unet_epoch100-109.pt')
    teacher.to(device)
    student = ResUNet1(layers=[2,2,2]).to(device)
    load_model_weights(student, 'weights/unet222.pt')
    train(teacher, student, train_loader, test_loader,
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9),
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001),
        num_epochs = 100,
        exp_path = 'runs/T222_S234',
        device = device
        )
