"""
情绪识别推理模块
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architectures.resnet_model import create_model


class EmotionPredictor:
    """情绪识别预测器"""
    
    def __init__(self, model_path, config_path):
        """
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 根据模型文件名确定架构类型
        model_filename = os.path.basename(model_path)
        print(f"🔍 检测模型类型: {model_filename}")
        
        if 'VGG' in model_filename or 'vgg' in model_filename:
            print(f"📋 加载VGG架构")
            from models.architectures.vgg_model import EmotionVGG
            self.model = EmotionVGG(num_classes=self.config['model']['num_classes'])
            self.is_rgb_model = True  # VGG需要RGB输入
        elif 'CustomCNN' in model_filename or 'custom' in model_filename:
            print(f"📋 加载Custom CNN架构")
            from models.architectures.vgg_model import CustomEmotionCNN
            self.model = CustomEmotionCNN(num_classes=self.config['model']['num_classes'])
            self.is_rgb_model = False  # Custom CNN使用灰度输入
        else:
            print(f"📋 加载ResNet架构")
            from models.architectures.resnet_model import create_model
            self.model = create_model(self.config)
            self.is_rgb_model = False  # ResNet使用灰度输入
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查检查点格式
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        try:
            self.model.load_state_dict(model_state)
            print(f"✅ 模型权重加载成功")
        except Exception as e:
            print(f"⚠️ 权重加载警告: {e}")
            # 尝试兼容性加载
            model_dict = self.model.state_dict()
            filtered_state = {k: v for k, v in model_state.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_state)
            self.model.load_state_dict(model_dict)
            print(f"✅ 兼容性加载完成")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 情绪标签映射
        self.emotion_labels = self.config['emotion_labels']
        self.emotion_emojis = self.config['emotion_emojis']
        
        # 根据模型类型设置数据预处理
        if self.is_rgb_model:
            # VGG模型使用RGB预处理
            self.transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"📊 使用RGB预处理")
        else:
            # ResNet和Custom CNN使用灰度预处理
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            print(f"📊 使用灰度预处理")
        
        print(f"✅ 模型加载完成，使用设备: {self.device}")
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            # 从文件路径加载图像
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # 从numpy数组转换
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像格式")
        
        # 应用变换
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """预测单张图像的情绪"""
        try:
            with torch.no_grad():
                # 预处理图像
                image_tensor = self.preprocess_image(image)
                
                # 模型推理
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # 获取预测结果
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # 获取所有类别的概率
                all_probabilities = probabilities[0].cpu().numpy()
                
                # 安全地获取情绪标签
                if str(predicted_class) in self.emotion_labels:
                    predicted_emotion = self.emotion_labels[str(predicted_class)]
                else:
                    # 备用映射
                    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                    predicted_emotion = emotion_names[predicted_class] if predicted_class < len(emotion_names) else 'unknown'
                
                # 安全地获取emoji
                emoji = self.emotion_emojis.get(predicted_emotion, '😐')
                
                # 构建结果
                result = {
                    'predicted_emotion': predicted_emotion,
                    'predicted_class_id': predicted_class,
                    'confidence': confidence,
                    'emoji': emoji,
                    'all_probabilities': {}
                }
                
                # 安全地构建所有概率
                emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                for i, prob in enumerate(all_probabilities):
                    if i < len(emotion_names):
                        result['all_probabilities'][emotion_names[i]] = float(prob)
                
                return result
                
        except Exception as e:
            print(f"❌ 预测过程发生错误: {e}")
            # 返回默认结果
            return {
                'predicted_emotion': 'neutral',
                'predicted_class_id': 4,
                'confidence': 0.0,
                'emoji': '😐',
                'all_probabilities': {
                    'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0,
                    'neutral': 1.0, 'sad': 0.0, 'surprise': 0.0
                },
                'error': str(e)
            }
            
            return result
    
    def predict_batch(self, images):
        """批量预测图像情绪"""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def get_top_k_predictions(self, image, k=3):
        """获取top-k预测结果"""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取top-k结果
            top_k_probs, top_k_indices = torch.topk(probabilities, k)
            
            results = []
            for i in range(k):
                class_id = top_k_indices[0][i].item()
                prob = top_k_probs[0][i].item()
                emotion = self.emotion_labels[str(class_id)]
                
                results.append({
                    'emotion': emotion,
                    'probability': prob,
                    'emoji': self.emotion_emojis[emotion]
                })
            
            return results


class RealTimeEmotionDetector:
    """实时情绪检测器（用于摄像头输入）"""
    
    def __init__(self, model_path, config_path):
        self.predictor = EmotionPredictor(model_path, config_path)
        
        # 人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_and_predict(self, frame):
        """检测人脸并预测情绪"""
        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            
            # 预测情绪
            emotion_result = self.predictor.predict(face_roi)
            
            # 添加位置信息
            emotion_result['face_bbox'] = (x, y, w, h)
            results.append(emotion_result)
        
        return results
    
    def draw_emotion_on_frame(self, frame, results):
        """在帧上绘制情绪信息"""
        for result in results:
            x, y, w, h = result['face_bbox']
            emotion = result['predicted_emotion']
            confidence = result['confidence']
            emoji = result['emoji']
            
            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 绘制情绪信息
            label = f"{emotion} {emoji} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # 背景矩形
            cv2.rectangle(frame, (x, y-30), (x+label_size[0], y), (0, 255, 0), -1)
            
            # 文字
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame


def demo_single_image():
    """单张图像预测演示"""
    # 配置文件路径
    config_path = 'configs/config.yaml'
    model_path = 'models/saved_models/emotion_resnet_resnet18_best.pth'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    # 创建预测器
    predictor = EmotionPredictor(model_path, config_path)
    
    # 预测示例图像
    test_image_path = 'archive/test/happy/PrivateTest_10131363.jpg'
    
    if os.path.exists(test_image_path):
        result = predictor.predict(test_image_path)
        
        print("预测结果:")
        print(f"情绪: {result['predicted_emotion']} {result['emoji']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("所有类别概率:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")
    else:
        print(f"测试图像不存在: {test_image_path}")


def demo_webcam():
    """摄像头实时预测演示"""
    config_path = 'configs/config.yaml'
    model_path = 'models/saved_models/emotion_resnet_resnet18_best.pth'
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建实时检测器
    detector = RealTimeEmotionDetector(model_path, config_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    print("按 'q' 键退出摄像头预览")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测和预测
        results = detector.detect_and_predict(frame)
        
        # 绘制结果
        frame = detector.draw_emotion_on_frame(frame, results)
        
        # 显示结果
        cv2.imshow('情绪识别', frame)
        
        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='情绪识别推理')
    parser.add_argument('--mode', choices=['image', 'webcam'], default='image',
                       help='运行模式: image (单张图像) 或 webcam (摄像头)')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        demo_single_image()
    elif args.mode == 'webcam':
        demo_webcam()
