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
        
        # 用户情绪历史记录（用于上下文感知）
        self.emotion_history = []
        self.max_history = 5  # 最多记录5次情绪
        
        # 智能离线回应库（更自然的朋友式回应）
        self.fallback_responses = {
            'angry': [
                "看得出你很生气，想发泄一下吗？我在这里听着。",
                "遇到糟心事了？深呼吸一下，咱们慢慢聊。",
                "愤怒是正常的，别憋着，说出来会好些。",
                "感受到你的火气了，什么事让你这么upset？",
                "生气的时候记得照顾好自己，我陪着你。"
            ],
            'disgust': [
                "看起来遇到什么恶心的事了？要不咱换个话题？",
                "这种感觉我懂，有时候确实会遇到让人反感的事。",
                "不舒服的感觉总会过去的，想听点轻松的吗？",
                "直觉告诉你远离不好的东西，这很正常。",
                "我理解这种感受，需要转移一下注意力吗？"
            ],
            'fear': [
                "感到害怕很正常，记住我一直在这里陪着你。",
                "恐惧让我们更小心，这其实是在保护自己。",
                "勇敢不是不害怕，而是害怕了还能继续前行。",
                "想聊聊让你担心的事吗？说出来可能会轻松些。",
                "深呼吸，告诉自己：这些我都能应对。"
            ],
            'happy': [
                "你的笑容真有感染力！什么好事让你这么开心？",
                "看到你这么高兴我也很开心！分享一下呗～",
                "快乐就是要传染给更多人，继续保持！",
                "这个笑容太治愈了，多笑笑对身体好！",
                "开心的时光值得好好珍惜，愿你一直这么快乐！"
            ],
            'neutral': [
                "平静的状态也很棒，内心安宁是种福气。",
                "看起来今天心情不错，想聊聊最近怎么样吗？",
                "有时候就需要这样安静的时刻，享受当下。",
                "平和的表情很有魅力，想分享下今天的心情吗？",
                "在忙碌中保持冷静，这是很了不起的能力。"
            ],
            'sad': [
                "看到你难过，真想给你一个温暖的拥抱。",
                "悲伤就像雨天，总会过去的，阳光还会再来。",
                "想哭就哭吧，释放情感对心理健康有好处。",
                "你不是一个人在承受，有什么想说的吗？",
                "难过的时候记住，总有人在默默关心着你。"
            ],
            'surprise': [
                "哇！什么事让你这么惊讶？看起来很有趣！",
                "这个表情太可爱了！一定遇到什么意外惊喜了？",
                "保持这份好奇心，生活会更加精彩有趣！",
                "惊喜总是让人印象深刻，愿意分享一下吗？",
                "意外之喜最珍贵，你的表情告诉我一定很棒！"
            ]
        }
        
        print(f"✅ 智能情绪响应生成器初始化完成")
        print(f"🔧 API提供商: {self.provider}")
        if self.api_key:
            print(f"🔑 API已配置 - 使用{self.provider}进行AI情绪抚慰")
        else:
            print(f"🧠 使用离线智能模式 - 精心设计的情绪感知回应")
    
    def generate_response(self, emotion: str, confidence: float = 0.0, context: Dict = None) -> str:
        """
        智能生成情绪相关的回应文本 - 优先使用AI API进行专业情绪抚慰
        
        Args:
            emotion: 检测到的情绪类型
            confidence: 置信度 (0.0-1.0)
            context: 附加上下文信息（可选）
            
        Returns:
            情绪相关的回应文本
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
        """智能构建API请求的提示词 - 专注于情绪抚慰"""
        
        emotion_descriptions = {
            'angry': '愤怒/生气',
            'disgust': '厌恶/恶心', 
            'fear': '恐惧/害怕',
            'happy': '开心/快乐',
            'neutral': '平静/中性',
            'sad': '悲伤/难过',
            'surprise': '惊讶/意外'
        }
        
        # 智能置信度分析
        if confidence > 0.8:
            confidence_level = "非常确定"
        elif confidence > 0.6:
            confidence_level = "比较确定"
        else:
            confidence_level = "不太确定"
        
        emotion_cn = emotion_descriptions.get(emotion, emotion)
        
        # 获取用户补充说明
        user_context = ""
        if context and context.get('user_input'):
            user_context = f"用户补充说明：「{context['user_input']}」"
        
        # 分析情绪历史模式
        recent_emotions = [h['emotion'] for h in self.emotion_history[-3:]]
        emotion_pattern = self._analyze_emotion_pattern()
        
        # 根据不同情绪专门设计提示词，让回应更自然有温度
        if emotion in ['happy', 'surprise']:
            # 积极情绪处理 - 自然分享喜悦
            context_hint = "用户情绪积极，" if emotion_pattern != "improving" else "用户情绪在好转，"
                
            prompt = f"""
你是一个温暖贴心的朋友。用户刚刚通过表情识别显示出"{emotion_cn}"的情绪，你能感受到他们的开心。
{user_context}

请像一个真正关心他们的朋友一样回应：
- 自然地表达你也为他们感到高兴
- 如果有补充说明，要真诚地回应具体内容
- 用轻松愉快的语气，可以带点幽默感
- 避免过于正式或教条式的语言
- 回应要有温度，像朋友间的真实对话

请直接给出自然的回应，不超过40字。
"""
        elif emotion in ['sad', 'fear']:
            # 负面情绪处理 - 真诚陪伴安慰
            context_hint = "用户情绪低落，" if emotion_pattern != "declining" else "用户情绪持续低落，"
                
            prompt = f"""
你是一个善解人意的知心朋友。用户通过表情识别显示出"{emotion_cn}"的情绪，你能感受到他们内心的不安或难过。
{user_context}

请像一个真正在乎他们的朋友一样给予安慰：
- 用最真诚的语言表达你的理解和关心
- 如果有补充说明，要针对具体情况给予贴心回应
- 语气要温暖而不是冷冰冰的建议
- 让他们感受到真正的陪伴和支持
- 可以轻柔地询问但不要给压力

请直接给出温暖的回应，不超过40字。
"""
        elif emotion == 'angry':
            # 愤怒情绪处理 - 理解和支持
            prompt = f"""
你是一个理解包容的好朋友。用户通过表情显示出"{emotion_cn}"的情绪，你能感受到他们的愤怒或挫败。
{user_context}

请像一个真正理解他们的朋友一样回应：
- 首先表达你能理解他们的感受
- 如果有补充说明，要针对具体情况表示理解
- 用平静而有力的语气给予支持
- 避免说教，而是站在他们一边
- 可以轻松地建议发泄方式

请直接给出理解支持的回应，不超过40字。
"""
        elif emotion == 'disgust':
            # 厌恶情绪处理 - 共情理解
            prompt = f"""
你是一个善于共情的朋友。用户通过表情显示出"{emotion_cn}"的情绪，可能遇到了让人反感的事情。
{user_context}

请像一个真正理解他们的朋友一样回应：
- 表达你能理解这种感受
- 如果有补充说明，要针对具体情况表示认同
- 用轻松的语气帮助转移注意力
- 避免深究细节，而是给予理解和支持

请直接给出理解包容的回应，不超过40字。
"""
        else:  # neutral 或其他
            # 中性情绪处理 - 自然陪伴
            prompt = f"""
你是一个温和的朋友。用户通过表情显示出"{emotion_cn}"的状态，看起来比较平静。
{user_context}

请像一个自然的朋友一样与他们交流：
- 认可他们当前的状态
- 如果有补充说明，要自然地回应
- 用轻松友好的语气
- 可以随意聊聊，不要太正式
- 创造轻松的氛围

请直接给出自然友好的回应，不超过40字。
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
    
    def _post_process_response(self, generated_text: str, emotion: str) -> str:
        """智能后处理生成的回应"""
        # 移除不必要的标点和格式
        text = generated_text.strip()
        
        # 移除可能的引号
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # 确保长度合适
        if len(text) > 40:
            # 如果太长，尝试找到合适的截断点
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
            "我理解你现在的感受，有什么想分享的吗？"
        ])
        
        # 分析情绪历史模式，提供上下文感知的回应
        recent_emotions = [h['emotion'] for h in self.emotion_history[-3:]]
        emotion_pattern = self._analyze_emotion_pattern()
        
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
        
        # 根据置信度选择合适的回应风格
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
