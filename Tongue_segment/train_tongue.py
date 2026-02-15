import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.unet import ResUNet, UNet, CBAMUNet, ResUNet2
from models.loss import DiceLoss
from data.TongueDataset import TongueDataset, TongueSegDataset, RandomTransform
from util import Metrics, save_seg_images, save_model, load_model_weights, calculate_iou, calculate_dice
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=5)

def l1_regularization(model):
    l1_loss = 0.0
    for name, param in model.named_parameters():
        # 只对fc层的权重进行L1正则化
        if 'fc' in name and 'weight' in name:  # 确保只对全连接层的权重进行正则化
            l1_loss += torch.sum(torch.abs(param))  # L1正则化项是参数绝对值的和
    return l1_loss

def test(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    mse_scores = 0
    iou_scores = []
    dice_scores = []
    mse_loss_fn = nn.L1Loss()
    
    progress_bar = tqdm(dataloader, desc=f'   test', leave=False)
    with torch.no_grad():  # 关闭梯度计算
        for images, true_masks, features in progress_bar:
            images = images.to(device)
            true_masks = true_masks.to(device)
            features = features[:, :33].to(device)
            features = features / 4
            
            # 前向传播
            outputs, pred = model(images)
            pred_masks = (outputs > 0.5).float().squeeze(1)
            pred_masks = pred_masks.cpu().numpy()
            true_masks = true_masks.cpu().numpy()
            mse = mse_loss_fn(pred, features).item()
            mse_scores += mse * features.size(0)
            
            # 计算每个batch的IoU和Dice
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                iou = calculate_iou(pred_mask, true_mask)
                dice = calculate_dice(pred_mask, true_mask)
                iou_scores.append(iou)
                dice_scores.append(dice)
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                'GPU': f"{memory_allocated:.0f} MB",
                'MSE': f"{mse:.5f}",
                'IoU': f"{iou:.5f}",
                'Dice': f"{dice:.5f}",
                })
    
    # 计算平均IoU和Dice
    mean_mse = mse_scores / len(iou_scores)
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    return mean_mse, mean_iou, mean_dice
    

def train(model, train_loader, val_loader, optimizer, num_epochs, exp_path, device):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
    os.mkdir(exp_path)
    writer = SummaryWriter(log_dir=exp_path)
    mse_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCELoss()
    dice_loss_fn = DiceLoss()
    best_dice = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        metrics = Metrics('train', exp_path,
                          keys=['L1 Norm', 'L1 Loss', 'BCE Loss', 'Dice Loss', 'Test MSE', 'Test IoU', 'Test Dice'], 
                          writer=writer)
        # running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        for images, masks, features in progress_bar:
            images, masks = images.to(device), masks.to(device)
            features = features[:, :33].to(device)
            features = features / 4

            # 前向传播
            optimizer.zero_grad()
            outputs, pred = model(images)
            pred_mask = (outputs > 0.5).float()
            # 计算 Loss
            l1_loss = mse_loss_fn(pred, features)
            bce_loss = bce_loss_fn(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            l1_norm = l1_regularization(model)
            loss = bce_loss*0.3 + dice_loss*0.1 + l1_loss + l1_norm * 0.005
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            metrics.update({'L1 Loss': l1_loss.item(),
                            'BCE Loss': bce_loss.item(),
                            'Dice Loss': dice_loss.item(),}, masks.size(0))
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({'GPU': f"{memory_allocated:.0f} MB",
                                      'L1 Norm': f"{l1_norm.item():.5f}",
                                      'L1 Loss': f"{l1_loss.item():.5f}",
                                      'BCE Loss': f"{bce_loss.item():.5f}",
                                      'Dice Loss': f"{dice_loss.item():.5f}"})
        
        # Epoch后
        test_mse, test_iou, test_dice = test(model, val_loader, device)
        metrics.update({'Test MSE' : test_mse,
                        'Test IoU' : test_iou,
                        'Test Dice': test_dice})
        metrics.write(epoch)
        # 每个epoch结束后保存预测图/打印训练损失
        save_seg_images(images.cpu(), masks.cpu(), outputs.detach().cpu(), pred_mask.cpu(), os.path.join(exp_path, f'{epoch}.jpg'))
        save_model(model, 'unet', epoch, exp_path, intervals=10)
        # 保存最佳表现的模型
        if metrics['Test Dice'] > best_dice:
            best_dice = metrics['Test Dice']
            save_model(model, 'best', epoch, exp_path, intervals=1)
            
        print('Epoch: {}/{} | BCE loss: {:.5f} | Test MSE: {:.4f} | Test IoU: {:.4f} | Test Dice: {:.4f}'.format(epoch, num_epochs,
                                                                                                metrics['BCE Loss'],
                                                                                                metrics['Test MSE'],
                                                                                                metrics['Test IoU'],
                                                                                                metrics['Test Dice'],
                                                                                                ))
        

if __name__ == '__main__':
    # 创建数据集实例
    # train_set = TongueSegDataset(image_path='./data/img', label_path='./data/gt', transform=RandomTransform())
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # test_set = TongueSegDataset(image_path='./data/img_test', label_path='./data/gt_test', transform=RandomTransform())
    # test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
    train_set = TongueDataset('data/Tongue_train', transform=RandomTransform())
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
    test_set = TongueDataset('data/Tongue_test', transform=RandomTransform())
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ResUNet2(num_classes=33).to(device)
    load_model_weights(model, 'weights/tongue05.pt')
    train(model, train_loader, test_loader,
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001),
        num_epochs = 100,
        exp_path = 'runs/tongue06_l1_norm',
        device = device
        )
