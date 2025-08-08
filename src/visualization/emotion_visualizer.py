"""
情绪可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO


class EmotionVisualizer:
    """情绪可视化器"""
    
    def __init__(self, config):
        self.config = config
        self.emotion_labels = list(config['emotion_labels'].values())
        self.emotion_colors = config['emotion_colors']
        self.emotion_emojis = config['emotion_emojis']
        
        # 设置matplotlib英文字体
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_probability_bar_chart(self, probabilities, title="Emotion Probability Distribution"):
        """Create probability bar chart"""
        emotions = list(probabilities.keys())
        probs = list(probabilities.values())
        colors = [self.emotion_colors[emotion] for emotion in emotions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(emotions, probs, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Emotion Category', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_ylim(0, 1)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_radar_chart(self, probabilities, title="Emotion Radar Chart"):
        """Create emotion radar chart"""
        emotions = list(probabilities.keys())
        values = list(probabilities.values())
        
        # 使用plotly创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions,
            fill='toself',
            name='Emotion Probability',
            line_color='rgb(255, 99, 132)',
            fillcolor='rgba(255, 99, 132, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title=title,
            font=dict(size=14)
        )
        
        return fig
    
    def create_emotion_pie_chart(self, probabilities, title="情绪分布饼图"):
        """创建情绪饼图"""
        emotions = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [self.emotion_colors[emotion] for emotion in emotions]
        
        fig = go.Figure(data=[go.Pie(
            labels=emotions,
            values=values,
            hole=0.3,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def create_interactive_bar_chart(self, probabilities, title="交互式概率分布"):
        """创建交互式柱状图"""
        emotions = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [self.emotion_colors[emotion] for emotion in emotions]
        emojis = [self.emotion_emojis[emotion] for emotion in emotions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=emotions,
                y=values,
                marker_color=colors,
                text=[f'{emoji}<br>{val:.3f}' for emoji, val in zip(emojis, values)],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             '概率: %{y:.3f}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="情绪类别",
            yaxis_title="概率",
            yaxis=dict(range=[0, 1]),
            font=dict(size=14),
            showlegend=False
        )
        
        return fig
    
    def create_emotion_timeline(self, emotion_history, title="情绪时间轴"):
        """创建情绪变化时间轴"""
        if not emotion_history:
            return None
        
        timestamps = [record['timestamp'] for record in emotion_history]
        emotions = [record['emotion'] for record in emotion_history]
        confidences = [record['confidence'] for record in emotion_history]
        
        # 为每种情绪分配数值
        emotion_values = {emotion: i for i, emotion in enumerate(self.emotion_labels)}
        y_values = [emotion_values[emotion] for emotion in emotions]
        
        fig = go.Figure()
        
        # 添加散点图
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_values,
            mode='markers+lines',
            marker=dict(
                size=[conf * 20 for conf in confidences],  # 置信度影响点大小
                color=confidences,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="置信度")
            ),
            line=dict(width=2),
            text=[f'{emotion}<br>置信度: {conf:.3f}' 
                  for emotion, conf in zip(emotions, confidences)],
            hovertemplate='<b>%{text}</b><br>' +
                         '时间: %{x}<br>' +
                         '<extra></extra>',
            name='情绪变化'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="时间",
            yaxis_title="情绪类别",
            yaxis=dict(
                tickmode='array',
                tickvals=list(emotion_values.values()),
                ticktext=list(emotion_values.keys())
            ),
            font=dict(size=14),
            height=500
        )
        
        return fig
    
    def create_comparison_chart(self, multiple_results, title="多图像情绪对比"):
        """创建多图像情绪对比图"""
        if not multiple_results:
            return None
        
        emotions = self.emotion_labels
        n_images = len(multiple_results)
        
        fig = make_subplots(
            rows=1, cols=n_images,
            subplot_titles=[f"图像 {i+1}" for i in range(n_images)],
            specs=[[{"type": "polar"}] * n_images]
        )
        
        for i, result in enumerate(multiple_results):
            values = [result['all_probabilities'].get(emotion, 0) for emotion in emotions]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=emotions,
                    fill='toself',
                    name=f'图像 {i+1}',
                    line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def fig_to_base64(self, fig, format='png'):
        """将matplotlib图形转换为base64字符串"""
        buffer = BytesIO()
        fig.savefig(buffer, format=format, bbox_inches='tight', dpi=300)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return f"data:image/{format};base64,{image_base64}"
    
    def plotly_to_html(self, fig):
        """将plotly图形转换为HTML字符串"""
        return fig.to_html(include_plotlyjs='cdn', div_id=f"plot_{np.random.randint(1000, 9999)}")
    
    def create_emotion_summary_dashboard(self, result):
        """创建情绪分析仪表板"""
        probabilities = result['all_probabilities']
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["概率分布", "雷达图", "置信度", "Top-3情绪"],
            specs=[
                [{"type": "bar"}, {"type": "polar"}],
                [{"type": "indicator"}, {"type": "bar"}]
            ]
        )
        
        # 1. 概率分布柱状图
        emotions = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [self.emotion_colors[emotion] for emotion in emotions]
        
        fig.add_trace(
            go.Bar(x=emotions, y=values, marker_color=colors, showlegend=False),
            row=1, col=1
        )
        
        # 2. 雷达图
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=emotions,
                fill='toself',
                name='概率分布',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. 置信度指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=result['confidence'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "置信度"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=1
        )
        
        # 4. Top-3情绪
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_emotions = [item[0] for item in sorted_emotions]
        top3_values = [item[1] for item in sorted_emotions]
        top3_colors = [self.emotion_colors[emotion] for emotion in top3_emotions]
        
        fig.add_trace(
            go.Bar(
                x=top3_emotions,
                y=top3_values,
                marker_color=top3_colors,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=f"情绪分析仪表板 - {result['predicted_emotion']} {result['emoji']}",
            height=800,
            font=dict(size=12)
        )
        
        return fig


def demo_visualization():
    """可视化演示"""
    import yaml
    
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建可视化器
    visualizer = EmotionVisualizer(config)
    
    # 模拟预测结果
    mock_result = {
        'predicted_emotion': 'happy',
        'confidence': 0.85,
        'emoji': '😊',
        'all_probabilities': {
            'angry': 0.02,
            'disgust': 0.01,
            'fear': 0.03,
            'happy': 0.85,
            'neutral': 0.05,
            'sad': 0.02,
            'surprise': 0.02
        }
    }
    
    # 创建各种可视化
    print("创建概率柱状图...")
    bar_fig = visualizer.create_probability_bar_chart(mock_result['all_probabilities'])
    bar_fig.savefig('emotion_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("创建雷达图...")
    radar_fig = visualizer.create_radar_chart(mock_result['all_probabilities'])
    radar_fig.write_html('emotion_radar_chart.html')
    
    print("创建饼图...")
    pie_fig = visualizer.create_emotion_pie_chart(mock_result['all_probabilities'])
    pie_fig.write_html('emotion_pie_chart.html')
    
    print("创建仪表板...")
    dashboard_fig = visualizer.create_emotion_summary_dashboard(mock_result)
    dashboard_fig.write_html('emotion_dashboard.html')
    
    print("可视化演示完成！")


if __name__ == "__main__":
    demo_visualization()
