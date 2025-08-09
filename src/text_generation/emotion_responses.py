"""
Multi-API intelligent emotion response text generator - supports DeepSeek and other APIs
"""

import os
import requests
import json
import random
import time
from typing import Dict, List, Optional
from datetime import datetime


class EmotionTextGenerator:
    """Intelligent emotion text generator - supports multiple APIs with automatic switching"""
    
    def __init__(self, config: Dict):
        """
        Initialize text generator
        
        Args:
            config: Configuration dictionary containing API settings and other information
        """
        self.config = config
        self.api_config = config.get('api', {})
        
        # Determine which API provider to use
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
        # Record current emotion to history
        self._add_to_history(emotion, confidence)
        
        # Analyze emotion patterns
        emotion_pattern = self._analyze_emotion_pattern()
        
        try:
            # Try to use AI API to generate personalized responses
            ai_response = self._generate_ai_response(emotion, confidence, context, emotion_pattern)
            if ai_response:
                return ai_response
        except Exception as e:
            print(f"AI response generation failed, using intelligent offline mode: {e}")
        
        # Use intelligent offline response
        return self._generate_offline_response(emotion, confidence, emotion_pattern)
    
    def _add_to_history(self, emotion: str, confidence: float):
        """Add emotion record to history"""
        timestamp = datetime.now()
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # Keep history at reasonable length
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
    
    def _analyze_emotion_pattern(self) -> str:
        """Analyze recent emotion change patterns"""
        if len(self.emotion_history) < 2:
            return "stable"
        
        recent_emotions = [record['emotion'] for record in self.emotion_history[-3:]]
        
        # Check emotion improvement patterns
        positive_emotions = ['happy', 'surprise', 'neutral']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        if len(recent_emotions) >= 2:
            # From negative to positive = improvement
            if (recent_emotions[-2] in negative_emotions and 
                recent_emotions[-1] in positive_emotions):
                return "improving"
            
            # From positive to negative = decline  
            elif (recent_emotions[-2] in positive_emotions and 
                  recent_emotions[-1] in negative_emotions):
                return "declining"
            
            # Persistent negative emotions = need special care
            elif all(e in negative_emotions for e in recent_emotions[-2:]):
                return "concerning"
            
            # Persistent positive emotions = good state
            elif all(e in positive_emotions for e in recent_emotions[-2:]):
                return "positive"
        
        return "stable"
    
    def _generate_ai_response(self, emotion: str, confidence: float, context: Dict, emotion_pattern: str) -> Optional[str]:
        self._record_emotion(emotion, confidence)
        
        # Prioritize using AI API to provide genuine emotional comfort
        if self.api_key:
            try:
                return self._generate_with_api(emotion, confidence, context)
            except Exception as e:
                # If 402 error (quota exhausted) or other API errors, silently use backup responses
                error_msg = str(e).lower()
                if '402' in error_msg or 'payment' in error_msg or 'quota' in error_msg:
                    print(f"💡 API quota exhausted, switching to offline intelligent mode")
                elif 'not found' in error_msg or '404' in error_msg:
                    print(f"🔧 API endpoint error, switching to offline intelligent mode")
                else:
                    print(f"🔄 API temporarily unavailable, using offline intelligent mode: {e}")
                
                return self._get_smart_fallback_response(emotion, confidence)
        else:
            return self._get_smart_fallback_response(emotion, confidence)
    
    def _record_emotion(self, emotion: str, confidence: float):
        """Record emotion history for context awareness"""
        current_record = {
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        self.emotion_history.append(current_record)
        
        # Keep history within reasonable range
        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history:]
    
    def _generate_with_api(self, emotion: str, confidence: float, context: Dict = None) -> str:
        """Use API to generate professional emotional comfort responses"""
        
        # Intelligently build prompts
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
        
        print(f"🌐 Calling {self.provider} API...")
        print(f"📡 URL: {self.base_url}")
        
        # Add retry mechanism and longer timeout
        for attempt in range(2):  # Retry up to 2 times
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=30  # Increase timeout to 30 seconds
                )
                break
            except requests.exceptions.Timeout as e:
                if attempt == 0:
                    print(f"⏰ Attempt {attempt+1} timed out, retrying...")
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    raise e
        
        print(f"📊 API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            
            # Intelligent post-processing: ensure responses are natural and friendly
            return self._post_process_response(generated_text, emotion)
        else:
            # Output detailed error information for debugging
            try:
                error_detail = response.json()
                print(f"❌ API error details: {error_detail}")
            except:
                print(f"❌ API response content: {response.text}")
            
            raise Exception(f"API request failed: {response.status_code}")
    
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
        """Generate intelligent offline responses, considering emotion patterns and history"""
        # Get basic responses
        responses = self.fallback_responses.get(emotion, [
            "I can sense your current emotions. Would you like to talk about it?"
        ])
        
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
        
        elif emotion_pattern == "concerning":
            # Persistent negative emotions, provide special care
            caring_responses = {
                'sad': "I've noticed your mood hasn't been great lately. Remember you're not alone, I'm always here.",
                'angry': "Looks like many things are troubling you. How about taking a break and going slowly?",
                'fear': "I feel you've been worried. That must be exhausting, right? Want to talk about what's worrying you?"
            }
            return caring_responses.get(emotion, "It looks like things have been tough lately. Remember to take care of yourself.")
        
        # Choose appropriate response style based on confidence level
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
        
        # Ensure appropriate punctuation at the end
        if not text.endswith(('。', '！', '？', '~', '.', '!', '?')):
            if emotion in ['happy', 'surprise']:
                text += '!'
            elif emotion in ['sad', 'fear']:
                text += '.'
            else:
                text += '~'
        
        return text
    
    def _get_smart_fallback_response(self, emotion: str, confidence: float) -> str:
        """Get intelligent offline responses - intelligently adjust based on confidence, history and context"""
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
            # High confidence: more direct, confident responses
            return responses[0] if responses else "I can clearly see your emotions from your expression!"
        elif confidence > 0.65:
            # Medium confidence: balanced natural responses
            mid_idx = len(responses) // 2
            return responses[mid_idx] if responses else "I can sense your current mood changes."
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
    
    def generate_batch_responses(self, emotions: List[str]) -> Dict[str, str]:
        """Generate responses for multiple emotions in batch"""
        responses = {}
        for emotion in emotions:
            responses[emotion] = self.generate_response(emotion)
        return responses
    
    def get_conversation_summary(self) -> str:
        """Get conversation history summary"""
        if not self.emotion_history:
            return "No emotion records yet"
        
        recent = self.emotion_history[-3:]
        emotions = [h['emotion'] for h in recent]
        
        if len(set(emotions)) == 1:
            return f"Recent emotions are quite stable, mainly {emotions[0]}"
        else:
            return f"Emotions have changed: {' → '.join(emotions[-3:])}"


# For backward compatibility, keep the original function interface
def generate_emotion_response(emotion: str, config: Dict = None) -> str:
    """
    Generate emotion response (backward compatibility function)
    
    Args:
        emotion: emotion type
        config: configuration information
        
    Returns:
        response text
    """
    if config is None:
        # Default configuration
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
    # Test SiliconFlow API call
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
    
    print("🧪 Testing SiliconFlow API call (Free DeepSeek model):")
    print("=" * 50)
    
    generator = EmotionTextGenerator(test_config)
    
    # Test single emotion
    test_emotion = 'happy'
    test_confidence = 0.9
    test_context = {'user_input': 'Work went really well today, feeling great!'}
    
    print(f"📸 Test emotion: {test_emotion}")
    print(f"🎯 Confidence: {test_confidence}")
    print(f"💬 User input: {test_context['user_input']}")
    print()
    
    response = generator.generate_response(test_emotion, test_confidence, test_context)
    print(f"🤖 AI Response: {response}")
    print()
    
    # Test different emotions
    emotions_to_test = ['sad', 'angry']
    for emotion in emotions_to_test:
        response = generator.generate_response(emotion, 0.8)
        print(f"🎭 {emotion}: {response}")
