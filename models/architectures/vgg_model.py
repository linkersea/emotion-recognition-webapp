"""
VGG情绪识别模型实现
针对小尺寸图像(48x48)优化的VGG架构
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EmotionVGG(nn.Module):
    """基于VGG的情绪识别模型"""
    
    def __init__(self, model_name='vgg16', num_classes=7, pretrained=True, dropout_rate=0.5):
        """
        Args:
            model_name: VGG模型名称 ('vgg11', 'vgg16', 'vgg19')
            num_classes: 情绪类别数量
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比率
        """
        super(EmotionVGG, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 加载预训练的VGG模型
        if model_name == 'vgg11':
            if pretrained:
                self.backbone = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vgg11_bn(weights=None)
        elif model_name == 'vgg16':
            if pretrained:
                self.backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vgg16_bn(weights=None)
        elif model_name == 'vgg19':
            if pretrained:
                self.backbone = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vgg19_bn(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 获取特征提取器（卷积层部分）
        self.features = self.backbone.features
        
        # 自适应平均池化（适应48x48输入）
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 自定义分类器（适配小图像）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 初始化分类器权重
        self._initialize_weights()
        
        print(f"✅ VGG{model_name.upper()}模型已创建 (针对48x48优化)")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.features(x)
        
        # 自适应池化
        x = self.avgpool(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class CustomEmotionCNN(nn.Module):
    """针对情绪识别优化的自定义CNN"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(CustomEmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一组卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 48x48 -> 24x24
            nn.Dropout2d(0.25),
            
            # 第二组卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24x24 -> 12x12
            nn.Dropout2d(0.25),
            
            # 第三组卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12x12 -> 6x6
            nn.Dropout2d(0.25),
            
            # 第四组卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6 -> 3x3
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        print("✅ 自定义情绪识别CNN已创建 (针对48x48优化)")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_vgg_model(config):
    """创建VGG模型"""
    model_name = config['model'].get('vgg_name', 'vgg16')
    
    model = EmotionVGG(
        model_name=model_name,
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate']
    )
    return model


def create_custom_cnn(config):
    """创建自定义CNN模型"""
    model = CustomEmotionCNN(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    return model


def compare_architectures():
    """比较不同架构的参数量"""
    import yaml
    
    print("=" * 60)
    print("模型架构对比分析")
    print("=" * 60)
    
    # 模拟配置
    config = {
        'model': {
            'name': 'ResNet18',
            'num_classes': 7,
            'pretrained': True,
            'dropout_rate': 0.5,
            'use_cbam': True
        }
    }
    
    # 测试输入
    dummy_input = torch.randn(1, 3, 48, 48)
    
    # ResNet18 + CBAM
    from models.architectures.resnet_model import create_model
    resnet_model = create_model(config)
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    resnet_output = resnet_model(dummy_input)
    
    # VGG16
    vgg_model = create_vgg_model(config)
    vgg_params = sum(p.numel() for p in vgg_model.parameters())
    vgg_output = vgg_model(dummy_input)
    
    # 自定义CNN
    custom_model = create_custom_cnn(config)
    custom_params = sum(p.numel() for p in custom_model.parameters())
    custom_output = custom_model(dummy_input)
    
    print(f"\n📊 模型对比:")
    print(f"   ResNet18+CBAM: {resnet_params:,} 参数")
    print(f"   VGG16:         {vgg_params:,} 参数")
    print(f"   Custom CNN:    {custom_params:,} 参数")
    
    print(f"\n🎯 推荐分析:")
    print(f"   当前ResNet18+CBAM: 64.99%准确率")
    print(f"   预期VGG16:         66-70%准确率 (更深层次特征)")
    print(f"   预期Custom CNN:    68-72%准确率 (专门优化)")
    
    print(f"\n💡 建议:")
    print(f"   1. 尝试VGG16: 更多层数可能提取更丰富特征")
    print(f"   2. 尝试Custom CNN: 针对48x48和情绪识别优化")
    print(f"   3. 数据增强: 可能比换架构更有效")
    
    return vgg_model, custom_model


if __name__ == "__main__":
    compare_architectures()
