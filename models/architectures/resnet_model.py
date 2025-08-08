"""
情绪识别ResNet模型架构 - 集成CBAM注意力机制
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """CBAM通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(spatial_attention)


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EmotionResNet(nn.Module):
    """基于ResNet+CBAM的情绪识别模型"""
    
    def __init__(self, model_name='resnet18', num_classes=7, pretrained=True, dropout_rate=0.5, use_cbam=True):
        """
        Args:
            model_name: ResNet模型名称 ('resnet18', 'resnet34', 'resnet50')
            num_classes: 情绪类别数量
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比率
            use_cbam: 是否使用CBAM注意力机制
        """
        super(EmotionResNet, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_cbam = use_cbam
        
        # 加载预训练的ResNet模型
        if model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            feature_dim = 512
        elif model_name == 'resnet34':
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet34(weights=None)
            feature_dim = 512
        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 移除最后的平均池化和全连接层，保留特征图
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 添加CBAM注意力机制
        if use_cbam:
            self.cbam = CBAM(feature_dim)
            print(f"✅ CBAM注意力机制已启用 (reduction=16)")
        else:
            self.cbam = None
            print("❌ 未使用CBAM注意力机制")
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 初始化新层的权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # 应用CBAM注意力机制
        if self.cbam is not None:
            features = self.cbam(features)
        
        # 分类
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x):
        """获取特征向量（用于可视化分析）"""
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        return features


class EmotionResNetWithAttention(nn.Module):
    """带注意力机制的ResNet情绪识别模型"""
    
    def __init__(self, model_name='resnet18', num_classes=7, pretrained=True, dropout_rate=0.5):
        super(EmotionResNetWithAttention, self).__init__()
        
        # 基础ResNet骨干网络
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        
        # 移除最后的平均池化和全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        
        # 注意力权重
        attention_weights = self.attention(features)
        
        # 应用注意力
        attended_features = features * attention_weights
        
        # 分类
        output = self.classifier(attended_features)
        
        return output


def create_model(config):
    """根据配置创建模型"""
    architecture = config['model'].get('architecture', 'resnet').lower()
    
    if architecture == 'resnet':
        model = EmotionResNet(
            model_name=config['model']['name'].lower(),
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate'],
            use_cbam=config['model'].get('use_cbam', True)
        )
        print(f"🏗️ 使用ResNet架构: {config['model']['name']}")
        
    elif architecture == 'vgg':
        # 导入VGG模型
        try:
            from models.architectures.vgg_model import create_vgg_model
            model = create_vgg_model(config)
            print(f"🏗️ 使用VGG架构: {config['model']['name']}")
        except ImportError:
            print("❌ VGG模型未找到，回退到ResNet")
            model = EmotionResNet(
                model_name='resnet18',
                num_classes=config['model']['num_classes'],
                pretrained=config['model']['pretrained'],
                dropout_rate=config['model']['dropout_rate'],
                use_cbam=False
            )
            
    elif architecture == 'custom_cnn':
        try:
            from models.architectures.vgg_model import create_custom_cnn
            model = create_custom_cnn(config)
            print("🏗️ 使用自定义CNN架构")
        except ImportError:
            print("❌ 自定义CNN未找到，回退到ResNet")
            model = EmotionResNet(
                model_name='resnet18',
                num_classes=config['model']['num_classes'],
                pretrained=config['model']['pretrained'],
                dropout_rate=config['model']['dropout_rate'],
                use_cbam=False
            )
    else:
        print(f"❌ 未知架构 {architecture}，使用默认ResNet")
        model = EmotionResNet(
            model_name='resnet18',
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate'],
            use_cbam=True
        )
    
    return model


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open('../../configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = create_model(config)
    
    # 统计参数
    count_parameters(model)
    
    # 测试模型
    dummy_input = torch.randn(1, 3, 48, 48)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出概率分布: {torch.softmax(output, dim=1)}")
