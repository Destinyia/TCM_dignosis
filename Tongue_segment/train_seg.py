import argparse
import os
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.TongueDataset import TongueDataset, TongueSegDataset, RandomTransform
from models.loss import DiceLoss
from models.unet import CBAMUNet, ResUNet, ResUNet1, ResUNet2, ResUNet2Dist, UNet
from util import (
    Metrics,
    apply_dense_crf,
    calculate_dice,
    calculate_iou,
    load_model_weights,
    save_model,
    save_seg_images,
)

np.set_printoptions(precision=5)


def _distance_maps(masks: torch.Tensor, mode: str, normalize: bool, device: torch.device) -> torch.Tensor:
    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError(f"scipy is required for distance map loss: {exc}")

    masks_np = masks.detach().cpu().numpy()
    dist_maps = []
    for i in range(masks_np.shape[0]):
        mask = masks_np[i, 0] > 0.5
        if mode == "boundary":
            dist_in = ndimage.distance_transform_edt(mask)
            dist_out = ndimage.distance_transform_edt(~mask)
            dist = dist_in - dist_out
        else:
            dist = ndimage.distance_transform_edt(mask)
        if normalize:
            maxv = np.max(dist)
            if maxv > 0:
                dist = dist / maxv
        dist_maps.append(dist)
    dist_np = np.stack(dist_maps, axis=0)[:, None, :, :]
    return torch.from_numpy(dist_np).float().to(device)


def _extract_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            return outputs[0], outputs[2]
        if len(outputs) >= 1:
            return outputs[0], None
    return outputs, None


def _apply_crf_batch(images: torch.Tensor, probs: torch.Tensor, args) -> torch.Tensor:
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    probs_np = probs.detach().cpu().numpy()
    refined = []
    for img, prob in zip(images_np, probs_np):
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        refined_prob = apply_dense_crf(
            img,
            prob[0],
            iterations=args.crf_iter,
            sxy_gaussian=args.crf_sxy_gaussian,
            compat_gaussian=args.crf_compat_gaussian,
            sxy_bilateral=args.crf_sxy_bilateral,
            srgb_bilateral=args.crf_srgb_bilateral,
            compat_bilateral=args.crf_compat_bilateral,
        )
        refined.append(refined_prob[None, ...])
    refined_np = np.stack(refined, axis=0)
    return torch.from_numpy(refined_np).float().to(probs.device)


def test(model, dataloader, device, args):
    model.eval()
    iou_scores = []
    dice_scores = []

    progress_bar = tqdm(dataloader, desc="   test", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            images = batch[0].to(device)
            true_masks = batch[1].to(device)

            outputs = model(images)
            seg_out, _ = _extract_outputs(outputs)
            probs = seg_out
            if args.use_crf:
                probs = _apply_crf_batch(images, probs, args)
            pred_masks = (probs > 0.5).float().squeeze(1).cpu().numpy()
            true_masks_np = true_masks.squeeze(1).cpu().numpy()

            for pred_mask, true_mask in zip(pred_masks, true_masks_np):
                iou = calculate_iou(pred_mask, true_mask)
                dice = calculate_dice(pred_mask, true_mask)
                iou_scores.append(iou)
                dice_scores.append(dice)
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                "GPU": f"{memory_allocated:.0f} MB",
                "IoU": f"{iou:.5f}",
                "Dice": f"{dice:.5f}",
            })

    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    return mean_iou, mean_dice


def train(model, train_loader, val_loader, optimizer, num_epochs, exp_path, device, args):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
    os.makedirs(exp_path, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_path)
    bce_loss_fn = nn.BCELoss()
    dice_loss_fn = DiceLoss()
    dist_loss_fn = nn.MSELoss()
    best_dice = 0.0
    warned_no_dist = False

    for epoch in range(1, num_epochs + 1):
        model.train()
        metrics = Metrics(
            "train",
            exp_path,
            keys=["BCE Loss", "Dice Loss", "Dist Loss", "Test IoU", "Test Dice"],
            writer=writer,
        )

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in progress_bar:
            images, masks = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            seg_out, dist_out = _extract_outputs(outputs)
            pred_mask = (seg_out > 0.5).float()

            bce_loss = bce_loss_fn(seg_out, masks)
            dice_loss = dice_loss_fn(seg_out, masks)
            if args.base_loss == "dice":
                base_loss = dice_loss
            elif args.base_loss == "bce_dice":
                base_loss = 0.5 * (bce_loss + dice_loss)
            else:
                base_loss = bce_loss

            dist_loss = torch.tensor(0.0, device=device)
            if args.distance_weight > 0:
                if dist_out is None:
                    if not warned_no_dist:
                        print("Warning: distance loss enabled but model has no distance head.")
                        warned_no_dist = True
                else:
                    dist_target = _distance_maps(
                        masks, args.distance_mode, args.distance_normalize, device
                    )
                    dist_loss = dist_loss_fn(dist_out, dist_target)

            loss = base_loss + args.distance_weight * dist_loss
            loss.backward()
            optimizer.step()

            metrics.update(
                {"BCE Loss": bce_loss.item(), "Dice Loss": dice_loss.item(), "Dist Loss": dist_loss.item()},
                masks.size(0),
            )
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                "GPU": f"{memory_allocated:.0f} MB",
                "BCE Loss": f"{bce_loss.item():.5f}",
                "Dice Loss": f"{dice_loss.item():.5f}",
                "Dist Loss": f"{dist_loss.item():.5f}",
            })

        test_iou, test_dice = test(model, val_loader, device, args)
        metrics.update({"Test IoU": test_iou, "Test Dice": test_dice})
        metrics.write(epoch)
        save_seg_images(
            images.cpu(),
            masks.cpu(),
            seg_out.detach().cpu(),
            pred_mask.cpu(),
            os.path.join(exp_path, f"{epoch}.jpg"),
        )
        save_model(model, "unet", epoch, exp_path, intervals=10)
        if metrics["Test Dice"] > best_dice:
            best_dice = metrics["Test Dice"]
            save_model(model, "best", epoch, exp_path, intervals=1)

        print(
            "Epoch: {}/{} | BCE loss: {:.5f} | Test IoU: {:.4f} | Test Dice: {:.4f}".format(
                epoch, num_epochs, metrics["BCE Loss"], metrics["Test IoU"], metrics["Test Dice"]
            )
        )


def _build_model(name: str):
    if name == "unet":
        return UNet()
    if name == "resunet":
        return ResUNet()
    if name == "resunet1":
        return ResUNet1()
    if name == "resunet2":
        return ResUNet2()
    if name == "resunet2dist":
        return ResUNet2Dist()
    if name == "cbamunet":
        return CBAMUNet()
    raise ValueError(f"Unknown model: {name}")


def _build_dataloaders(args):
    if args.dataset_type == "seg":
        train_set = TongueSegDataset(
            image_path=args.train_image_dir,
            label_path=args.train_mask_dir,
            transform=RandomTransform(),
        )
        val_set = TongueSegDataset(
            image_path=args.val_image_dir,
            label_path=args.val_mask_dir,
            transform=RandomTransform(),
        )
    else:
        train_set = TongueDataset(args.train_root, transform=RandomTransform())
        val_set = TongueDataset(args.val_root, transform=RandomTransform())

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train tongue segmentation model")
    parser.add_argument("--dataset-type", type=str, default="seg", choices=["seg", "tongue"])
    parser.add_argument("--train-image-dir", type=str, default="./data/img")
    parser.add_argument("--train-mask-dir", type=str, default="./data/gt")
    parser.add_argument("--val-image-dir", type=str, default="./data/img_test")
    parser.add_argument("--val-mask-dir", type=str, default="./data/gt_test")
    parser.add_argument("--train-root", type=str, default="data/Tongue_train")
    parser.add_argument("--val-root", type=str, default="data/Tongue_test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--model",
        type=str,
        default="resunet1",
        choices=["unet", "resunet", "resunet1", "resunet2", "resunet2dist", "cbamunet"],
    )
    parser.add_argument("--pretrained-weights", type=str, default="weights/unet222.pt")
    parser.add_argument("--exp-path", type=str, default="runs/tongue640")
    parser.add_argument("--base-loss", type=str, default="bce", choices=["bce", "dice", "bce_dice"])
    parser.add_argument("--distance-weight", type=float, default=0.0)
    parser.add_argument("--distance-mode", type=str, default="inside", choices=["inside", "boundary"])
    parser.add_argument("--distance-normalize", action="store_true", default=True)
    parser.add_argument("--no-distance-normalize", action="store_false", dest="distance_normalize")
    parser.add_argument("--use-crf", action="store_true")
    parser.add_argument("--crf-iter", type=int, default=5)
    parser.add_argument("--crf-sxy-gaussian", type=int, default=3)
    parser.add_argument("--crf-compat-gaussian", type=int, default=3)
    parser.add_argument("--crf-sxy-bilateral", type=int, default=80)
    parser.add_argument("--crf-srgb-bilateral", type=int, default=13)
    parser.add_argument("--crf-compat-bilateral", type=int, default=10)
    args = parser.parse_args()

    train_loader, val_loader = _build_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(args.model).to(device)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        load_model_weights(model, args.pretrained_weights)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        exp_path=args.exp_path,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
