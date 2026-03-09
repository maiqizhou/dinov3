#!/usr/bin/env python3
"""
Linear Probing with DINOv3 ViT-B/16 Backbone

Trains a linear classifier on top of frozen DINOv3 features
for binary classification (normal vs tumor).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import json
from datetime import datetime


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen DINOv3 backbone."""
    
    def __init__(self, backbone, embed_dim, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Linear classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
    
    def forward(self, x):
        # Get features from backbone (frozen)
        with torch.no_grad():
            features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        return logits


def get_transforms():
    """Get image transforms compatible with DINOv3."""
    # Standard ImageNet normalization used by DINOv3
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    model.backbone.eval()  # Keep backbone in eval mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='DINOv3 Linear Probing')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset (ImageFolder format)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--repo-dir', type=str, required=True,
                        help='Path to DINOv3 repo directory')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to pretrained weights (.pth file)')
    parser.add_argument('--model-arch', type=str, default='vitb16',
                        choices=['vits16', 'vitb16', 'vitl16'],
                        help='Model architecture (vits16, vitb16, vitl16)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory with explicit check
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir):
        raise RuntimeError(f"Failed to create output directory: {args.output_dir}")
    print(f"Output directory created: {os.path.exists(args.output_dir)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model architecture config
    arch_config = {
        'vits16': {'hub_name': 'dinov3_vits16', 'embed_dim': 384},
        'vitb16': {'hub_name': 'dinov3_vitb16', 'embed_dim': 768},
        'vitl16': {'hub_name': 'dinov3_vitl16', 'embed_dim': 1024},
    }
    
    config = arch_config[args.model_arch]
    hub_name = config['hub_name']
    embed_dim = config['embed_dim']
    
    # Load backbone via torch.hub with local weights
    print(f"Loading DINOv3 {args.model_arch} from: {args.repo_dir}")
    print(f"Using weights: {args.weights}")
    
    backbone = torch.hub.load(
        args.repo_dir,
        hub_name,
        source='local',
        weights=args.weights,
        pretrained=True
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    print(f"Backbone embedding dimension: {embed_dim}")
    
    # Create model with linear head
    model = LinearClassifier(backbone, embed_dim=embed_dim, num_classes=2, freeze_backbone=True)
    model = model.to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Data transforms
    transform = get_transforms()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss and optimizer (only for classifier head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)  # Ensure dir exists
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.2f}%)")
        
        # Save checkpoint every epoch
        os.makedirs(args.output_dir, exist_ok=True)  # Ensure dir exists
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure dir exists
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.classifier.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
    }, final_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    config = {
        'repo_dir': args.repo_dir,
        'weights': args.weights,
        'model_arch': args.model_arch,
        'embed_dim': embed_dim,
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'val_split': args.val_split,
        'seed': args.seed,
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'timestamp': datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
