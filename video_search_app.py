import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime
import base64
from io import BytesIO
import subprocess
import shutil

# 安全导入 CLIP 模型
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    try:
        # 备选：直接导入处理器
        from transformers.models.clip import CLIPProcessor, CLIPModel
    except ImportError:
        st.error("❌ 无法导入 CLIP 模型。请运行: pip install --upgrade transformers")
        st.stop()

# 检查ffmpeg是否可用
if shutil.which("ffmpeg") is None:
    st.error("⚠️ 系统未安装ffmpeg，某些功能可能不可用")

# ================= 页面配置 =================
st.set_page_config(
    page_title="🎬 视频搜索引擎 - Video Search Engine",
    page_icon="🎬",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        height: 50px;
        font-size: 1.1em;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    h3 {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ================= 模型加载 (缓存) =================
@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"📱 正在使用 {device.upper()} 设备加载模型...")
        
        # 加载处理器 - 带重试和备选方案
        st.info("加载处理器 (Processor)...")
        try:
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                trust_remote_code=True,
                timeout=30
            )
        except Exception as e:
            st.warning(f"⚠️ 处理器加载失败，尝试备选方案: {str(e)}")
            # 备选：使用自动模型类
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        st.success("✓ 处理器加载成功")
        
        # 加载模型 - 带重试和备选方案
        st.info("加载CLIP模型... (首次加载需要几分钟)")
        try:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                trust_remote_code=True,
                timeout=30
            )
        except Exception as e:
            st.warning(f"⚠️ 模型加载失败，尝试备选方案: {str(e)}")
            # 备选：使用自动模型类
            from transformers import AutoModel
            model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        st.success("✓ 模型加载成功")
        
        # 移至设备
        st.info(f"将模型移至 {device.upper()}...")
        model = model.to(device)
        st.success(f"✓ 模型已移至 {device.upper()}")
        
        # 验证模型
        st.info("验证模型...")
        if hasattr(model, 'vision_model') and hasattr(model, 'text_model'):
            st.success("✓ 模型结构正确")
        else:
            st.warning("⚠️ 模型结构不标准")
        
        return model, processor, device
    
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.error(f"详细错误信息:\n{str(e)}")
        import traceback
        st.error(f"追踪:\n{traceback.format_exc()}")
        return None, None, None

# ================= 视频处理函数 =================
def extract_frames(video_file, interval=1):
    """从视频文件中提取帧（使用ffmpeg）"""
    # 保存上传的文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    # 创建临时输出目录
    output_dir = tempfile.mkdtemp()
    
    try:
        # 使用ffmpeg提取帧
        # fps=1/interval 表示每隔interval秒提取一帧
        cmd = [
            'ffmpeg',
            '-i', tmp_path,
            '-vf', f'fps=1/{interval}',
            '-q:v', '2',
            os.path.join(output_dir, 'frame_%04d.jpg'),
            '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"❌ 视频处理失败: {result.stderr}")
            return None, None
        
        # 读取提取的帧
        frames = []
        timestamps = []
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(output_dir, frame_file)
            try:
                img = Image.open(frame_path)
                frames.append(img)
                timestamps.append(idx * interval)
            except Exception as e:
                st.warning(f"⚠️ 无法读取帧 {frame_file}")
                continue
        
        if not frames:
            st.error("❌ 无法从视频中提取任何帧")
            return None, None
        
        return frames, timestamps
    
    except FileNotFoundError:
        st.error("❌ ffmpeg 未安装。请安装 ffmpeg 或使用云端部署版本。")
        return None, None
    
    except Exception as e:
        st.error(f"❌ 处理视频时出错: {str(e)}")
        return None, None
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
        except:
            pass

def search_frames(model, processor, search_text, frames, timestamps, device):
    """搜索最匹配的帧"""
    try:
        # 验证输入
        if not frames or len(frames) == 0:
            st.error("❌ 没有提取到任何帧")
            return []
        
        if not search_text or search_text.strip() == "":
            st.error("❌ 搜索词不能为空")
            return []
        
        # 处理输入
        try:
            inputs = processor(
                text=[search_text],
                images=frames,
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            st.error(f"❌ 处理器错误: {str(e)}")
            return []
        
        # 移至设备
        try:
            inputs = inputs.to(device)
        except Exception as e:
            st.error(f"❌ 设备转移失败: {str(e)}")
            return []
        
        # 推理
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            st.error(f"❌ 模型推理失败: {str(e)}")
            return []
        
        # 检查输出
        if outputs is None or outputs.logits_per_image is None:
            st.error("❌ 模型没有返回有效的输出")
            return []
        
        # logits_per_image shape: [num_images, 1]
        logits_per_image = outputs.logits_per_image
        
        # 调试信息
        st.info(f"📊 Debug: logits_per_image 形状 = {logits_per_image.shape}, 值 = {logits_per_image.squeeze().tolist()[:3]}...")
        
        # 检查是否所有logits都相等
        unique_logits = torch.unique(logits_per_image)
        if len(unique_logits) == 1:
            st.warning("⚠️ 警告: 所有logits相等，模型可能未正确学习")
        
        # 挤压维度
        try:
            logits_per_image = logits_per_image.squeeze(-1)  # [num_images, 1] → [num_images]
        except Exception as e:
            st.error(f"❌ squeeze失败: {str(e)}")
            return []
        
        # 应用softmax
        try:
            import torch.nn.functional as F
            probs = F.softmax(logits_per_image, dim=0)
        except Exception as e:
            st.error(f"❌ softmax失败: {str(e)}")
            return []
        
        # 验证概率
        prob_sum = probs.sum().item()
        if abs(prob_sum - 1.0) > 0.01:
            st.warning(f"⚠️ 概率和 = {prob_sum:.4f}（应该≈1.0）")
        
        # 获取Top-5结果
        k = min(5, len(frames))
        if k == 1:
            # 如果只有1张图，直接返回
            top5_probs = probs.unsqueeze(0)
            top5_indices = torch.tensor([0]).to(device)
        else:
            try:
                top5_probs, top5_indices = torch.topk(probs, k=k)
            except Exception as e:
                st.error(f"❌ topk失败: {str(e)}")
                return []
        
        # 构建结果
        results = []
        for prob, idx in zip(top5_probs, top5_indices):
            results.append({
                'frame': frames[idx.item()],
                'timestamp': timestamps[idx.item()],
                'score': prob.item()
            })
        
        return results
    
    except Exception as e:
        st.error(f"❌ 搜索函数出错: {str(e)}")
        import traceback
        st.error(f"详细错误:\n{traceback.format_exc()}")
        return []

def format_time(seconds):
    """格式化时间"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{mins}:{secs:02d}"

# ================= 主应用 =================
st.title("🎬 智能视频搜索引擎")
st.markdown("### 上传视频，用自然语言描述找到你想要的片段")

# 加载模型
model, processor, device = load_model()

# 检查模型是否成功加载
if model is None or processor is None:
    st.error("❌ 模型加载失败，应用无法继续")
    st.stop()

st.success(f"✅ 模型已加载 (运行在 {device.upper()})")

# 快速验证模型
try:
    st.info("🔍 验证模型可用性...")
    from PIL import Image
    test_img = Image.new('RGB', (224, 224), (100, 100, 100))
    test_inputs = processor(text=["test"], images=[test_img], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        test_outputs = model(**test_inputs)
    if test_outputs and test_outputs.logits_per_image is not None:
        st.success("✓ 模型验证通过")
    else:
        st.error("❌ 模型验证失败：输出无效")
        st.stop()
except Exception as e:
    st.error(f"❌ 模型验证失败: {str(e)}")
    st.stop()

# 侧边栏设置
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ 设置")
    
    interval = st.slider(
        "帧提取间隔 (秒)",
        min_value=1,
        max_value=10,
        value=2,
        help="每隔多少秒提取一帧。较小的值更精确但速度更慢"
    )
    
    st.markdown("---")
    st.markdown("### 💡 使用提示")
    st.info(
        """
        1. 上传MP4格式的视频文件
        2. 输入要搜索的内容描述（英文效果更好）
        3. 点击搜索，获取Top-5匹配结果
        4. 每个结果显示时间点和置信度
        """
    )

# 主应用区域
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件 (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="视频文件大小限制根据服务器而定"
    )

with col2:
    st.markdown("### 🔍 搜索描述")
    search_text = st.text_input(
        "输入你要搜索的内容",
        placeholder="例如: 'A cat sleeping' 或 'Ball entering goal'",
        help="使用英文描述效果最佳"
    )

# 处理上传和搜索
if uploaded_file and search_text:
    st.markdown("---")
    
    # 提取视频帧
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📹 正在提取视频帧...")
    frames, timestamps = extract_frames(uploaded_file, interval=interval)
    progress_bar.progress(30)
    
    if frames is None:
        st.stop()
    
    status_text.text(f"✅ 成功提取 {len(frames)} 帧")
    progress_bar.progress(60)
    
    # 搜索
    status_text.text("🔎 正在搜索匹配的帧...")
    results = search_frames(model, processor, search_text, frames, timestamps, device)
    progress_bar.progress(100)
    
    status_text.text("✅ 搜索完成！")
    
    # 显示结果
    st.markdown("---")
    st.markdown(f"## 🎯 搜索结果")
    st.markdown(f"**搜索词:** \"{search_text}\" | **提取帧数:** {len(frames)} | **处理间隔:** {interval}秒")
    
    # 显示Top-5结果
    for idx, result in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result['frame'], use_container_width=True)
            
            with col2:
                st.markdown(f"### #{idx} 结果")
                st.markdown(f"**⏱️ 时间:** {format_time(result['timestamp'])}")
                
                # 置信度指标
                confidence = result['score']
                st.markdown(f"**📊 置信度:** {confidence*100:.1f}%")
                st.progress(confidence)
                
                # 下载按钮
                img_bytes = BytesIO()
                result['frame'].save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                st.download_button(
                    label=f"⬇️ 下载第{idx}个结果",
                    data=img_bytes,
                    file_name=f"search_result_{idx}_at_{format_time(result['timestamp'])}.png",
                    mime="image/png"
                )
            
            st.markdown("---")
    
    # 生成HTML报告
    st.markdown("## 📊 生成报告")
    
    if st.button("📄 生成HTML报告"):
        # 创建HTML内容
        results_html = ""
        for idx, result in enumerate(results, 1):
            img_bytes = BytesIO()
            result['frame'].save(img_bytes, format='PNG')
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            
            results_html += f"""
            <div style="background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 4px solid #667eea;">
                <h3>结果 #{idx}</h3>
                <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; border-radius: 8px; margin: 10px 0;">
                <p><strong>⏱️ 时间:</strong> {format_time(result['timestamp'])}</p>
                <p><strong>📊 置信度:</strong> {result['score']*100:.2f}%</p>
                <div style="background: #e0e0e0; height: 8px; border-radius: 4px; margin: 10px 0; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {result['score']*100}%;"></div>
                </div>
            </div>
            """
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频搜索结果报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 40px;
            max-width: 1000px;
            margin: 0 auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #667eea;
            margin: 0;
        }}
        .header p {{
            color: #999;
            margin: 10px 0 0 0;
        }}
        .search-info {{
            background: #f0f4ff;
            border-left: 5px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 视频搜索结果报告</h1>
            <p>生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="search-info">
            <strong>🔍 搜索词:</strong> "{search_text}"<br>
            <strong>📊 提取帧数:</strong> {len(frames)}<br>
            <strong>⚙️ 处理间隔:</strong> {interval}秒
        </div>
        
        <h2>📋 Top-5 匹配结果</h2>
        {results_html}
        
        <div class="footer">
            <p>由 CLIP 视频搜索引擎生成</p>
        </div>
    </div>
</body>
</html>"""
        
        # 下载报告
        st.download_button(
            label="📥 下载HTML报告",
            data=html_content,
            file_name=f"video_search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
        st.success("✅ 报告已准备好下载！")

else:
    # 欢迎界面
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📹 功能特点
        - 🎥 支持多种视频格式
        - ⚡ 秒级响应速度
        - 🤖 基于CLIP深度学习
        - 📊 多结果排名显示
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 如何使用
        1. 上传视频文件
        2. 输入搜索描述
        3. 点击搜索
        4. 查看Top-5结果
        5. 下载报告
        """)
    
    with col3:
        st.markdown("""
        ### 💡 搜索建议
        - 使用英文效果最佳
        - 简洁清晰的描述
        - 具体的视觉特征
        - 例: "soccer goal"
        """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 20px;">
    <p>🎬 智能视频搜索引擎 | 由 OpenAI CLIP 提供支持</p>
    <p>可部署到 Streamlit Cloud / Hugging Face Spaces / 云服务器</p>
</div>
""", unsafe_allow_html=True)
