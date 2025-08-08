"""
FER2013数据集预处理工具
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml


class FER2013Dataset(Dataset):
    """FER2013数据集类"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据变换
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 情绪标签映射
        self.emotion_labels = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'sad': 5, 'surprise': 6
        }
        
        # 收集数据文件路径和标签
        self.data_paths = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """加载数据路径和标签"""
        for emotion_name, emotion_idx in self.emotion_labels.items():
            emotion_dir = os.path.join(self.data_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_file)
                        self.data_paths.append(img_path)
                        self.labels.append(emotion_idx)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.data_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def get_data_transforms(config):
    """获取数据变换"""
    
    # 训练集变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=config['augmentation']['rotation_range']),
        transforms.ColorJitter(
            brightness=config['augmentation']['brightness_range'],
            contrast=config['augmentation']['contrast_range']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集变换
    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(config):
    """创建数据加载器"""
    
    # 获取数据变换
    train_transform, val_transform = get_data_transforms(config)
    
    # 创建训练数据集
    train_dataset = FER2013Dataset(
        data_dir=config['data']['train_path'],
        transform=train_transform,
        is_train=True
    )
    
    # 创建测试数据集
    test_dataset = FER2013Dataset(
        data_dir=config['data']['test_path'],
        transform=val_transform,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, test_loader


def analyze_dataset(data_dir):
    """分析数据集分布"""
    emotion_counts = {}
    total_images = 0
    
    for emotion_name in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
        emotion_dir = os.path.join(data_dir, emotion_name)
        if os.path.exists(emotion_dir):
            count = len([f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            emotion_counts[emotion_name] = count
            total_images += count
    
    print("数据集分析:")
    print(f"总图像数: {total_images}")
    print("\n各情绪类别分布:")
    for emotion, count in emotion_counts.items():
        percentage = (count / total_images) * 100
        print(f"{emotion}: {count} ({percentage:.1f}%)")
    
    return emotion_counts


if __name__ == "__main__":
    # 加载配置
    with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 分析数据集
    print("训练集分析:")
    analyze_dataset(config['data']['train_path'])
    
    print("\n测试集分析:")
    analyze_dataset(config['data']['test_path'])
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(config)
    
    print(f"\n数据加载器创建成功:")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
