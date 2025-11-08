"""
2025 DL Lab5: Object Detection on Pascal VOC
Training Script for Server Execution

Author: [Your Name], [Your SID]
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.amp import autocast, GradScaler
from src.yolo import getODmodel
from yolo_loss import YOLOv3Loss
from src.dataset import VocDetectorDataset, train_data_pipelines, test_data_pipelines, collate_fn
from src.eval_voc import evaluate
from src.config import GRID_SIZES, ANCHORS
from torch.optim.lr_scheduler import CosineAnnealingLR


def main():
    print("="*60)
    print("YOLOv3 Object Detection Training")
    print("="*60)
    
    # ========== Hyperparameters ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 50
    batch_size = 40
    learning_rate = 1e-3
    lambda_coord = 5.0
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_class = 1.0
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda coord: {lambda_coord}")
    print(f"  Lambda obj: {lambda_obj}")
    print(f"  Lambda noobj: {lambda_noobj}")
    print(f"  Lambda class: {lambda_class}")
    
    # ========== Data paths ==========
    file_root_train = './dataset/image/'
    annotation_file_train = './dataset/vocall_train.txt'
    file_root_val = './dataset/image/'
    annotation_file_val = './dataset/vocall_val.txt'
    
    # ========== Create datasets ==========
    print('\n' + '='*60)
    print('Loading datasets...')
    print('='*60)
    
    train_dataset = VocDetectorDataset(
        root_img_dir=file_root_train,
        dataset_file=annotation_file_train,
        train=True,
        transform=train_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
    )
    print(f'Loaded {len(train_dataset)} train images')
    
    val_dataset = VocDetectorDataset(
        root_img_dir=file_root_val,
        dataset_file=annotation_file_val,
        train=False,
        transform=test_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
    )
    
    # For computing val mAP
    eval_dataset = VocDetectorDataset(
        root_img_dir=file_root_val,
        dataset_file=annotation_file_val,
        train=False,
        transform=test_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )
    print(f'Loaded {len(val_dataset)} val images')
    
    # ========== Model Initialization ==========
    print('\n' + '='*60)
    print('Initializing model...')
    print('='*60)
    
    load_network_path = None  # Set to checkpoint path if resuming
    pretrained = True
    model = getODmodel(pretrained=pretrained).to(device)
    
    if load_network_path:
        print(f'Loading checkpoint from {load_network_path}')
        model.load_state_dict(torch.load(load_network_path))
    
    print(f'Model created with pretrained={pretrained}')
    
    # ========== Training setup ==========
    criterion = YOLOv3Loss(lambda_coord, lambda_obj, lambda_noobj, lambda_class, ANCHORS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    
    print(f'Using mixed precision: {use_amp}')
    
    # ========== Training loop ==========
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60)
    
    torch.cuda.empty_cache()
    best_val_loss = np.inf
    
    for epoch in range(num_epochs):
        # ===== Training phase =====
        model.train()
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'{"="*60}')
        
        train_loss = 0.0
        for i, (images, target) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            target = [t.to(device) for t in target]
            
            # Forward pass
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                pred = model(images)
                loss_dict = criterion(pred, target)
            
            # Backward pass with mixed precision support
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss_dict['total'].item()
            
            # Print progress
            if i % 50 == 0:
                outstring = f'Iter [{i+1}/{len(train_loader)}], Loss: '
                outstring += ', '.join(f"{key}={val.item():.3f}" for key, val in loss_dict.items())
                print(outstring)
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        print(f'\nTraining Loss: {train_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # ===== Validation phase =====
        print('\nValidating...')
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            for i, (images, target) in enumerate(val_loader):
                # Move to device
                images = images.to(device)
                target = [t.to(device) for t in target]
                
                # Forward pass
                pred = model(images)
                loss_dict = criterion(pred, target)
                val_loss += loss_dict['total'].item()
            
            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
        
        # ===== Save checkpoints =====
        # Save best model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            print(f'✓ New best val loss: {best_val_loss:.5f}')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_detector.pth')
        
        # Save periodic checkpoints
        if (epoch + 1) in [5, 10, 20, 30, 40]:
            checkpoint_path = f'checkpoints/detector_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'✓ Saved checkpoint: {checkpoint_path}')
        
        # Save latest model
        torch.save(model.state_dict(), 'checkpoints/detector.pth')
        
        # ===== Evaluate mAP =====
        if (epoch + 1) % 5 == 0:
            print('\n' + '-'*60)
            print('Evaluating mAP on validation set...')
            print('-'*60)
            val_aps = evaluate(model, eval_loader)
            mean_ap = np.mean(val_aps)
            print(f'Epoch {epoch+1} - mAP: {mean_ap:.4f}')
            print('-'*60)
    
    # ========== Training complete ==========
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.5f}')
    print('='*60)


if __name__ == '__main__':
    main()
