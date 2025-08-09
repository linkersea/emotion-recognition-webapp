"""
基于多API的智能情绪响应文本生成器 - 支持DeepSeek等多种API
"""

import os
import requests
import json
import random
import time
from typing import Dict, List, Optional
from datetime import datetime


class EmotionTextGenerator:
    """智能情绪文本生成器 - 支持多种API，自动切换"""
    
    def __init__(self, config: Dict):
        """
        初始化文本生成器
        
        Args:
            config: 配置字典，包含API配置等信息
        """
        self.config = config
        self.api_config = config.get('api', {})
        
        # 确定使用的API提供商
        self.provider = self.api_config.get('provider', 'deepseek')
        self.current_api_config = self.api_config.get(self.provider, {})
        
        self.api_key = self.current_api_config.get('api_key', '')
        self.base_url = self.current_api_config.get('base_url', '')
        self.model = self.current_api_config.get('model', 'deepseek-chat')
        
        # User emotion history for context awareness
        self.emotion_history = []
        self.max_history = 5  # Record up to 5 emotions
        
        # Intelligent offline response library (natural friend-like responses)
        self.fallback_responses = {
            'angry': [
                "I can see you're angry. Want to vent? I'm here to listen.",
                "Having a rough time? Take a deep breath, let's talk it through.",
                "Anger is normal. Don't hold it in, expressing it helps.",
                "I can feel your frustration. What's got you so upset?",
                "Remember to take care of yourself when you're angry. I'm here with you."
            ],
            'disgust': [
                "Looks like something really bothered you. Want to change the topic?",
                "I understand that feeling. Sometimes we encounter truly repulsive things.",
                "Uncomfortable feelings will pass. Want to hear something light?",
                "Your instincts are telling you to avoid bad things. That's normal.",
                "I understand this feeling. Need a distraction?"
            ],
            'fear': [
                "Feeling scared is normal. Remember, I'm always here with you.",
                "Fear makes us more careful. It's actually protecting us.",
                "Courage isn't about not being afraid, it's moving forward despite fear.",
                "Want to talk about what's worrying you? It might help to share.",
                "Take a deep breath and tell yourself: I can handle this."
            ],
            'happy': [
                "Your smile is so contagious! What good news made you so happy?",
                "Seeing you this happy makes me happy too! Care to share?",
                "Happiness should spread to more people. Keep it up!",
                "That smile is so healing! Smiling is good for your health!",
                "Happy moments are worth treasuring. May you always be this joyful!"
            ],
            'neutral': [
                "A calm state is wonderful too. Inner peace is a blessing.",
                "You seem to be in a good mood today. How have things been lately?",
                "Sometimes we need quiet moments like this. Enjoy the present.",
                "Your peaceful expression is quite charming. Want to share how you're feeling today?",
                "Staying calm amid busyness is a remarkable ability."
            ],
            'sad': [
                "Seeing you sad makes me want to give you a warm hug.",
                "Sadness is like a rainy day - it will pass, and the sun will shine again.",
                "It's okay to cry. Releasing emotions is good for mental health.",
                "You're not bearing this alone. Is there anything you'd like to talk about?",
                "When you're sad, remember there are always people who care about you."
            ],
            'surprise': [
                "Wow! What surprised you so much? It looks interesting!",
                "That expression is so cute! Must have encountered some unexpected surprise?",
                "Keep that curiosity - it makes life more exciting and interesting!",
                "Surprises are always memorable. Care to share?",
                "Unexpected joys are the most precious. Your expression tells me it must be wonderful!"
            ]
        }
        
        print(f"✅ Intelligent Emotion Response Generator initialized successfully")
        print(f"🔧 API Provider: {self.provider}")
        if self.api_key:
            print(f"🔑 API configured - Using {self.provider} for AI emotional support")
        else:
            print(f"🧠 Using offline intelligent mode - Carefully designed emotion-aware responses")
    
    def generate_response(self, emotion: str, confidence: float = 0.0, context: Dict = None) -> str:
        """
        Intelligently generate emotion-related response text - Prioritize AI API for professional emotional support
        
        Args:
            emotion: Detected emotion type
            confidence: Confidence level (0.0-1.0)
            context: Additional context information (optional)
            
        Returns:
            Emotion-related response text
        """
        # 记录当前情绪到历史中
        self._add_to_history(emotion, confidence)
        
        # 分析情绪模式
        emotion_pattern = self._analyze_emotion_pattern()
        
        try:
            # 尝试使用AI API生成个性化回应
            ai_response = self._generate_ai_response(emotion, confidence, context, emotion_pattern)
            if ai_response:
                return ai_response
        except Exception as e:
            print(f"AI回应生成失败，使用智能离线模式: {e}")
        
        # 使用智能离线回应
        return self._generate_offline_response(emotion, confidence, emotion_pattern)
    
    def _add_to_history(self, emotion: str, confidence: float):
        """添加情绪记录到历史中"""
        timestamp = datetime.now()
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # 保持历史记录在合理长度
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
    
    def _analyze_emotion_pattern(self) -> str:
        """分析近期情绪变化模式"""
        if len(self.emotion_history) < 2:
            return "stable"
        
        recent_emotions = [record['emotion'] for record in self.emotion_history[-3:]]
        
        # 检查情绪改善模式
        positive_emotions = ['happy', 'surprise', 'neutral']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        if len(recent_emotions) >= 2:
            # 从负面转向正面 = 改善
            if (recent_emotions[-2] in negative_emotions and 
                recent_emotions[-1] in positive_emotions):
                return "improving"
            
            # 从正面转向负面 = 下降  
            elif (recent_emotions[-2] in positive_emotions and 
                  recent_emotions[-1] in negative_emotions):
                return "declining"
            
            # 持续负面情绪 = 需要特别关怀
            elif all(e in negative_emotions for e in recent_emotions[-2:]):
                return "concerning"
            
            # 持续正面情绪 = 状态良好
            elif all(e in positive_emotions for e in recent_emotions[-2:]):
                return "positive"
        
        return "stable"
    
    def _generate_ai_response(self, emotion: str, confidence: float, context: Dict, emotion_pattern: str) -> Optional[str]:
        self._record_emotion(emotion, confidence)
        
        # 优先使用AI API来提供真正的情绪抚慰
        if self.api_key:
            try:
                return self._generate_with_api(emotion, confidence, context)
            except Exception as e:
                # 如果是402错误（配额用完）或其他API错误，静默使用备用回应
                error_msg = str(e).lower()
                if '402' in error_msg or 'payment' in error_msg or 'quota' in error_msg:
                    print(f"💡 API配额已用完，切换到离线智能模式")
                elif 'not found' in error_msg or '404' in error_msg:
                    print(f"🔧 API端点错误，切换到离线智能模式")
                else:
                    print(f"🔄 API暂时不可用，使用离线智能模式: {e}")
                
                return self._get_smart_fallback_response(emotion, confidence)
        else:
            return self._get_smart_fallback_response(emotion, confidence)
    
    def _record_emotion(self, emotion: str, confidence: float):
        """记录情绪历史用于上下文感知"""
        current_record = {
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        self.emotion_history.append(current_record)
        
        # 保持历史记录在合理范围内
        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history:]
    
    def _generate_with_api(self, emotion: str, confidence: float, context: Dict = None) -> str:
        """使用API生成专业情绪抚慰回应"""
        
        # 智能构建提示词
        prompt = self._build_intelligent_prompt(emotion, confidence, context)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': self.current_api_config.get('max_tokens', 200),
            'temperature': self.current_api_config.get('temperature', 0.7),
            'stream': False
        }
        
        print(f"🌐 正在调用{self.provider} API...")
        print(f"📡 URL: {self.base_url}")
        
        # 添加重试机制和更长的超时时间
        for attempt in range(2):  # 最多重试2次
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=30  # 增加超时时间到30秒
                )
                break
            except requests.exceptions.Timeout as e:
                if attempt == 0:
                    print(f"⏰ 第{attempt+1}次超时，重试中...")
                    time.sleep(2)  # 等待2秒后重试
                else:
                    raise e
        
        print(f"📊 API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            
            # 智能后处理：确保回应自然友好
            return self._post_process_response(generated_text, emotion)
        else:
            # 输出详细的错误信息用于调试
            try:
                error_detail = response.json()
                print(f"❌ API错误详情: {error_detail}")
            except:
                print(f"❌ API响应内容: {response.text}")
            
            raise Exception(f"API请求失败: {response.status_code}")
    
    def _build_intelligent_prompt(self, emotion: str, confidence: float, context: Dict = None) -> str:
        """Intelligently build API request prompts - Focus on emotional comfort"""
        
        emotion_descriptions = {
            'angry': 'angry/furious',
            'disgust': 'disgusted/repulsed', 
            'fear': 'fearful/scared',
            'happy': 'happy/joyful',
            'neutral': 'calm/neutral',
            'sad': 'sad/melancholy',
            'surprise': 'surprised/amazed'
        }
        
        # Intelligent confidence analysis
        if confidence > 0.8:
            confidence_level = "very confident"
        elif confidence > 0.6:
            confidence_level = "fairly confident"
        else:
            confidence_level = "not very confident"
        
        emotion_desc = emotion_descriptions.get(emotion, emotion)
        
        # Get user's additional context
        user_context = ""
        if context and context.get('user_input'):
            user_context = f"User's additional note: \"{context['user_input']}\""
        
        # Analyze emotion history patterns
        recent_emotions = [h['emotion'] for h in self.emotion_history[-3:]]
        emotion_pattern = self._analyze_emotion_pattern()
        
        # Design specific prompts for different emotions to make responses more natural and warm
        if emotion in ['happy', 'surprise']:
            # Positive emotion handling - naturally share joy
            context_hint = "User's emotion is positive, " if emotion_pattern != "improving" else "User's emotion is improving, "
                
            prompt = f"""
You are a warm and caring friend. The user just showed "{emotion_desc}" emotion through facial recognition, and you can feel their happiness.
{user_context}

Please respond like a genuine friend who truly cares about them:
- Naturally express that you're also happy for them
- If there's additional context, respond sincerely to the specific content
- Use a light and cheerful tone, can include some humor
- Avoid overly formal or preachy language
- The response should be warm, like a real conversation between friends

Please provide a natural response directly, no more than 40 words.
"""
        elif emotion in ['sad', 'fear']:
            # Negative emotion handling - sincere companionship and comfort
            context_hint = "User's emotion is low, " if emotion_pattern != "declining" else "User's emotion continues to be low, "
                
            prompt = f"""
You are an understanding and caring close friend. The user shows "{emotion_desc}" emotion through facial recognition, and you can feel their unease or sadness.
{user_context}

Please comfort them like a friend who truly cares:
- Use the most sincere language to express your understanding and care
- If there's additional context, provide thoughtful responses to specific situations
- The tone should be warm rather than cold advice
- Let them feel genuine companionship and support
- You can gently inquire but don't pressure them

Please provide a warm response directly, no more than 40 words.
"""
        elif emotion == 'angry':
            # Anger emotion handling - understanding and support
            prompt = f"""
You are an understanding and tolerant good friend. The user shows "{emotion_desc}" emotion through facial expression, and you can feel their anger or frustration.
{user_context}

Please respond like a friend who truly understands them:
- First express that you can understand their feelings
- If there's additional context, show understanding for the specific situation
- Use a calm but supportive tone
- Avoid lecturing, instead stand by their side
- You can casually suggest ways to vent

Please provide an understanding and supportive response directly, no more than 40 words.
"""
        elif emotion == 'disgust':
            # Disgust emotion handling - empathetic understanding
            prompt = f"""
You are an empathetic friend. The user shows "{emotion_desc}" emotion through facial expression, possibly encountering something repulsive.
{user_context}

Please respond like a friend who truly understands them:
- Express that you can understand this feeling
- If there's additional context, show agreement with the specific situation
- Use a light tone to help redirect attention
- Avoid dwelling on details, instead give understanding and support

Please provide an understanding and tolerant response directly, no more than 40 words.
"""
        else:  # neutral or others
            # Neutral emotion handling - natural companionship
            prompt = f"""
You are a gentle friend. The user shows "{emotion_desc}" state through facial expression, appearing quite calm.
{user_context}

Please interact with them like a natural friend:
- Acknowledge their current state
- If there's additional context, respond naturally
- Use a relaxed and friendly tone
- You can chat casually, don't be too formal
- Create a relaxed atmosphere

Please provide a natural and friendly response directly, no more than 40 words.
"""
        
        return prompt
    
    def _generate_offline_response(self, emotion: str, confidence: float, emotion_pattern: str) -> str:
        """生成智能离线回应，考虑情绪模式和历史"""
        # 获取基础回应
        responses = self.fallback_responses.get(emotion, [
            "我能感受到你现在的情绪，想要聊聊吗？"
        ])
        
        # 根据情绪模式和上下文调整回应策略
        if emotion_pattern == "improving" and emotion in ['happy', 'neutral']:
            # 情绪在好转，给予积极反馈
            positive_responses = {
                'happy': "太棒了！看到你情绪越来越好，我也替你开心！",
                'neutral': "感觉你心情在慢慢好转，这很不错呢～"
            }
            return positive_responses.get(emotion, responses[0])
        
        elif emotion_pattern == "declining" and emotion in ['sad', 'angry', 'fear']:
            # 情绪在下降，提供更多温暖关怀
            supportive_responses = {
                'sad': "最近情绪有些低落呢，要不要和我聊聊？我会一直陪着你。",
                'angry': "看起来最近有些事情让你很烦心，深呼吸一下，咱们一起面对。",
                'fear': "感受到你内心的不安，别怕，有什么事我们一起想办法。"
            }
            return supportive_responses.get(emotion, responses[-1])
        
        elif emotion_pattern == "concerning":
            # 持续负面情绪，给予特别关怀
            caring_responses = {
                'sad': "注意到你最近情绪不太好，记住你并不孤单，我一直在这里。",
                'angry': "看起来有很多事情让你困扰，要不要先休息一下，慢慢来？",
                'fear': "感觉你一直很担心，这样很累吧？要不要说说让你担心的事？"
            }
            return caring_responses.get(emotion, "最近看起来很辛苦，要记得照顾好自己。")
        
        # 根据置信度选择合适的回应风格
        if confidence > 0.85:
            # High confidence: more direct, confident responses
            return responses[0] if responses else "I can clearly see your emotions from your expression!"
        elif confidence > 0.65:
            # Medium confidence: balanced natural responses
            mid_idx = len(responses) // 2
            return responses[mid_idx] if responses else "I can sense your current mood."
        else:
            # Low confidence: gentler, exploratory responses
            gentle_responses = {
                'happy': "You seem to be in a good mood? Either way, seeing you makes me happy~",
                'sad': "Feeling a bit down perhaps? If you want to talk, I'm always here.",
                'angry': "Seems like there might be some emotional turbulence. Want to talk about what's bothering you?",
                'neutral': "You look quite calm. How was your day?",
                'fear': "Feeling a bit uneasy? Don't worry, we can work through whatever it is together.",
                'disgust': "Looks like you might have encountered something unpleasant?",
                'surprise': "Something unexpected happen?"
            }
            return gentle_responses.get(emotion, "Whatever you're feeling right now, I'm here with you.")
    
    def _post_process_response(self, generated_text: str, emotion: str) -> str:
        """Intelligently post-process generated responses"""
        # Remove unnecessary punctuation and formatting
        text = generated_text.strip()
        
        # Remove possible quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Ensure appropriate length
        if len(text) > 40:
            # If too long, try to find a suitable truncation point
            sentences = text.split('。')
            if len(sentences) > 1 and len(sentences[0]) <= 35:
                text = sentences[0] + '。'
            else:
                text = text[:35] + '...'
        
        # 确保以适当的标点结尾
        if not text.endswith(('。', '！', '？', '~', '.', '!', '?')):
            if emotion in ['happy', 'surprise']:
                text += '！'
            elif emotion in ['sad', 'fear']:
                text += '。'
            else:
                text += '~'
        
        return text
    
    def _get_smart_fallback_response(self, emotion: str, confidence: float) -> str:
        """获取智能离线回应 - 根据置信度、历史和上下文智能调整"""
        responses = self.fallback_responses.get(emotion, [
            "I understand how you're feeling right now. Is there anything you'd like to share?"
        ])
        
        # Analyze emotion history patterns for context-aware responses
        recent_emotions = [h['emotion'] for h in self.emotion_history[-3:]]
        emotion_pattern = self._analyze_emotion_pattern()
        
        # Adjust response strategy based on emotion patterns and context
        if emotion_pattern == "improving" and emotion in ['happy', 'neutral']:
            # Emotions are improving, provide positive feedback
            positive_responses = {
                'happy': "That's wonderful! Seeing your mood getting better makes me happy too!",
                'neutral': "It feels like your mood is slowly improving. That's really nice~"
            }
            return positive_responses.get(emotion, responses[0])
        
        elif emotion_pattern == "declining" and emotion in ['sad', 'angry', 'fear']:
            # Emotions are declining, provide more warm care
            supportive_responses = {
                'sad': "Your mood seems a bit low lately. Want to talk with me? I'll always be here with you.",
                'angry': "Looks like some things have been bothering you lately. Take a deep breath, let's face it together.",
                'fear': "I can sense your inner unease. Don't be afraid, we can figure out whatever it is together."
            }
            return supportive_responses.get(emotion, responses[-1])
        
        # Choose appropriate response style based on confidence level
        if confidence > 0.85:
            # 高置信度：更直接、确定的回应
            return responses[0] if responses else "从你的表情能看出很明确的情绪呢！"
        elif confidence > 0.65:
            # 中等置信度：平衡自然的回应
            mid_idx = len(responses) // 2
            return responses[mid_idx] if responses else "能感受到你现在的心情变化。"
        else:
            # 低置信度：更温和、探索性的回应
            gentle_responses = {
                'happy': "似乎心情不错？不管怎样，看到你我就很开心～",
                'sad': "感觉可能有点不开心？如果想聊聊，我随时在这里。",
                'angry': "似乎有些情绪波动，要不要说说是什么事？",
                'neutral': "看起来还挺平静的，今天过得怎么样？",
                'fear': "感觉有点不安？别担心，有什么事我们一起解决。",
                'disgust': "看起来可能遇到了不太愉快的事？",
                'surprise': "好像有什么让你觉得意外的事？"
            }
            return gentle_responses.get(emotion, "不管你现在什么感受，我都在这里陪着你。")
    
    def generate_batch_responses(self, emotions: List[str]) -> Dict[str, str]:
        """批量生成多个情绪的回应"""
        responses = {}
        for emotion in emotions:
            responses[emotion] = self.generate_response(emotion)
        return responses
    
    def get_conversation_summary(self) -> str:
        """获取对话历史摘要"""
        if not self.emotion_history:
            return "还没有情绪记录"
        
        recent = self.emotion_history[-3:]
        emotions = [h['emotion'] for h in recent]
        
        if len(set(emotions)) == 1:
            return f"最近情绪比较稳定，主要是{emotions[0]}"
        else:
            return f"情绪有变化：{' → '.join(emotions[-3:])}"


# 为了向后兼容，保留原有的函数接口
def generate_emotion_response(emotion: str, config: Dict = None) -> str:
    """
    生成情绪回应（向后兼容函数）
    
    Args:
        emotion: 情绪类型
        config: 配置信息
        
    Returns:
        回应文本
    """
    if config is None:
        # 默认配置
        config = {
            'api': {
                'provider': 'deepseek',
                'deepseek': {
                    'api_key': '',
                    'base_url': 'https://api.deepseek.com/chat/completions',
                    'model': 'deepseek-chat',
                    'max_tokens': 200,
                    'temperature': 0.7
                }
            }
        }
    
    generator = EmotionTextGenerator(config)
    return generator.generate_response(emotion)


if __name__ == "__main__":
    # 测试硅基流动API调用
    test_config = {
        'api': {
            'provider': 'siliconflow',
            'siliconflow': {
                'api_key': 'sk-uxsiqasafikwfctaebknfpsfrvbnyyaxclygouotdjddnbyq',
                'base_url': 'https://api.siliconflow.cn/v1/chat/completions',
                'model': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                'max_tokens': 200,
                'temperature': 0.7
            }
        }
    }
    
    print("🧪 测试硅基流动API调用 (免费DeepSeek模型):")
    print("=" * 50)
    
    generator = EmotionTextGenerator(test_config)
    
    # 测试单个情绪
    test_emotion = 'happy'
    test_confidence = 0.9
    test_context = {'user_input': '今天工作很顺利，心情特别好！'}
    
    print(f"📸 测试情绪: {test_emotion}")
    print(f"🎯 置信度: {test_confidence}")
    print(f"💬 用户说明: {test_context['user_input']}")
    print()
    
    response = generator.generate_response(test_emotion, test_confidence, test_context)
    print(f"🤖 AI回应: {response}")
    print()
    
    # 测试不同情绪
    emotions_to_test = ['sad', 'angry']
    for emotion in emotions_to_test:
        response = generator.generate_response(emotion, 0.8)
        print(f"🎭 {emotion}: {response}")
