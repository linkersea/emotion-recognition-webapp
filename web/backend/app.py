"""
Emotion Recognition Web Application Backend API
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import yaml

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch

# Import custom modules
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.inference.predictor import EmotionPredictor
from src.visualization.emotion_visualizer import EmotionVisualizer
from src.text_generation.emotion_responses import EmotionTextGenerator


class EmotionRecognitionAPI:
    """Emotion Recognition API Service"""
    
    def __init__(self, config_path, model_path):
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize models and components
        self.predictor = EmotionPredictor(model_path, config_path)
        self.visualizer = EmotionVisualizer(self.config)
        self.text_generator = EmotionTextGenerator(self.config)
        
        # Create Flask application
        self.app = Flask(__name__, 
                        template_folder='../frontend',
                        static_folder='../static')
        CORS(self.app)
        
        # Register routes
        self._register_routes()
        
        print("Emotion Recognition API service initialized successfully")
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.route('/')
        def index():
            """Home page"""
            return render_template('index.html')
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict_emotion():
            """Emotion prediction API"""
            try:
                # Get uploaded image
                if 'image' not in request.files:
                    return jsonify({'error': 'No image uploaded'}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # 读取图像
                image = Image.open(file.stream)
                
                # 预测情绪
                result = self.predictor.predict(image)
                
                # 获取用户补充说明（如果有）
                user_input = request.form.get('user_input', '').strip()
                context = {'user_input': user_input} if user_input else None
                
                # 使用AI API进行专业情绪抚慰
                confidence = result.get('confidence', 0.0)
                text_response = self.text_generator.generate_response(
                    emotion=result['predicted_emotion'],
                    confidence=confidence,
                    context=context
                )
                
                # 添加文本回应到结果
                result['text_response'] = text_response
                
                # Add conversation summary (shows emotion trend)
                result['conversation_summary'] = self.text_generator.get_conversation_summary()
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        @self.app.route('/api/predict_base64', methods=['POST'])
        def predict_emotion_base64():
            """Base64 image emotion prediction API"""
            try:
                data = request.get_json()
                
                if 'image' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Parse base64 image
                image_data = data['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Predict emotion
                result = self.predictor.predict(image)
                
                # Use new intelligent text generation - users just need to take photos, system handles automatically
                confidence = result.get('confidence', 0.0)
                text_response = self.text_generator.generate_response(
                    emotion=result['predicted_emotion'],
                    confidence=confidence
                )
                
                result['text_response'] = text_response
                
                # Add conversation summary (shows emotion trend)  
                result['conversation_summary'] = self.text_generator.get_conversation_summary()
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        @self.app.route('/api/visualize', methods=['POST'])
        def create_visualization():
            """Create visualization chart API"""
            try:
                data = request.get_json()
                
                if 'probabilities' not in data:
                    return jsonify({'error': 'No probability data provided'}), 400
                
                probabilities = data['probabilities']
                chart_type = data.get('chart_type', 'bar')
                title = data.get('title', 'Emotion Distribution')
                
                # Create visualization
                if chart_type == 'bar':
                    fig = self.visualizer.create_interactive_bar_chart(probabilities, title)
                elif chart_type == 'radar':
                    fig = self.visualizer.create_radar_chart(probabilities, title)
                elif chart_type == 'pie':
                    fig = self.visualizer.create_emotion_pie_chart(probabilities, title)
                else:
                    return jsonify({'error': 'Unsupported chart type'}), 400
                
                # Convert to HTML
                chart_html = self.visualizer.plotly_to_html(fig)
                
                return jsonify({'chart_html': chart_html})
                
            except Exception as e:
                return jsonify({'error': f'Visualization creation failed: {str(e)}'}), 500
        
        @self.app.route('/api/generate_text', methods=['POST'])
        def generate_text_response():
            """Generate text response API"""
            try:
                data = request.get_json()
                
                if 'emotion' not in data:
                    return jsonify({'error': 'No emotion information provided'}), 400
                
                emotion = data['emotion']
                confidence = data.get('confidence', 0.8)  # Default confidence
                
                # Use new intelligent text generation
                text_response = self.text_generator.generate_response(
                    emotion=emotion,
                    confidence=confidence
                )
                
                return jsonify({
                    'text_response': text_response,
                    'conversation_summary': self.text_generator.get_conversation_summary()
                })
                
            except Exception as e:
                return jsonify({'error': f'Text generation failed: {str(e)}'}), 500
        
        @self.app.route('/api/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction API"""
            try:
                # Get uploaded multiple files
                files = request.files.getlist('images')
                
                if not files:
                    return jsonify({'error': 'No images uploaded'}), 400
                
                results = []
                for file in files:
                    if file.filename != '':
                        image = Image.open(file.stream)
                        result = self.predictor.predict(image)
                        result['filename'] = file.filename
                        results.append(result)
                
                return jsonify({'results': results})
                
            except Exception as e:
                return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500
        
        @self.app.route('/api/model_info', methods=['GET'])
        def get_model_info():
            """Get model information API"""
            try:
                info = {
                    'model_name': self.config['model']['name'],
                    'num_classes': self.config['model']['num_classes'],
                    'emotion_labels': self.config['emotion_labels'],
                    'emotion_emojis': self.config['emotion_emojis'],
                    'device': str(self.predictor.device)
                }
                return jsonify(info)
                
            except Exception as e:
                return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check API"""
            return jsonify({'status': 'healthy', 'message': 'Service is running normally'})
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Start service"""
        print(f"Starting Emotion Recognition API service...")
        print(f"Access URL: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app():
    """Create Flask application"""
    
    # Configuration file path
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    model_path = os.path.join(os.path.dirname(__file__), '../../models/saved_models/emotion_resnet_resnet18_best.pth')
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create API instance
    api = EmotionRecognitionAPI(config_path, model_path)
    
    return api.app


def main():
    """Main function"""
    try:
        # Configuration file path
        config_path = 'configs/config.yaml'
        
        # Get port from environment variable (for cloud deployment)
        # HF Spaces uses 7860, others use 5000
        port = int(os.environ.get('PORT', 7860 if 'SPACE_ID' in os.environ else 5000))
        
        # Dynamically find the latest best model file
        import glob
        models_dir = "models/saved_models"
        model_files = glob.glob(f"{models_dir}/*_best.pth")
        
        if not model_files:
            print(f"❌ Model file not found: {models_dir}")
            print("Please train the model first: python setup.py --train")
            return
        
        # Use the latest model file
        model_path = max(model_files, key=os.path.getctime)
        print(f"✅ Auto-selected model: {os.path.basename(model_path)}")
        
        # Check configuration file
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            print("Please create configuration file first")
            return
        
        # Create API service
        api = EmotionRecognitionAPI(config_path, model_path)
        
        # Start service
        print(f"🚀 Starting Web service...")
        print(f"📱 Access URL: http://0.0.0.0:{port}")
        
        # 显示局域网访问信息
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"🏠 LAN Access: http://{local_ip}:{port}")
            print(f"💡 Other devices on the same WiFi can access via the LAN IP")
        except Exception:
            print(f"💡 To find your LAN IP, run: ipconfig (Windows) or ifconfig (Mac/Linux)")
        
        # For cloud deployment, use 0.0.0.0 and environment port
        api.run(
            host='0.0.0.0',
            port=port,
            debug=False  # Disable debug in production
        )
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
