import os
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
from music21 import *
import matplotlib.pyplot as plt
from spleeter.separator import Separator
import speech_recognition as sr
from pydub import AudioSegment
import jieba
from pypinyin import pinyin, Style
import json
import wave
import struct
from pathlib import Path

# 设置 ffmpeg 路径
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + r'C:\Users\Jimmie\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin'

class MusicProcessor:
    def __init__(self):
        # 检查并下载必需的模型
        self._setup_models()
        
        # 使用2stems模型代替5stems，因为它更小更常用
        model_name = 'spleeter:2stems'
        
        # 检查本地模型路径
        local_model_path = os.path.join(os.getcwd(), 'pretrained_models')
        
        # 检查模型是否存在
        if os.path.exists(local_model_path):
            print(f"使用本地模型路径: {local_model_path}")
            # 确保模型目录存在于环境变量中
            os.environ['SPLEETER_MODELS_PATH'] = local_model_path
        else:
            print("本地模型路径不存在，将使用默认缓存路径")
            # 使用默认缓存路径
            os.environ.pop('SPLEETER_MODELS_PATH', None)  # 移除环境变量，使用默认路径
        
        try:
            # 初始化分离器
            self.separator = Separator(model_name, multiprocess=False)
            print(f"成功初始化分离器: {model_name}")
        except Exception as e:
            print(f"初始化分离器失败: {str(e)}")
            print("尝试下载模型...")
            # 如果初始化失败，可能是模型不存在，尝试下载
            from spleeter.commands.download_pretrained_models import download
            download(model_name)
            # 重新初始化
            self.separator = Separator(model_name, multiprocess=False)
    
    def _setup_models(self):
        """
        检查并下载必需的模型
        """
        print("🔍 检查模型状态...")
        
        # 检查并导入模型管理器
        try:
            from model_manager import ModelManager
            manager = ModelManager()
            
            # 检查模型状态
            status = manager.get_model_status()
            
            missing_models = []
            for model_name, info in status.items():
                if not info["exists"]:
                    missing_models.append(model_name)
                else:
                    print(f"✅ {model_name}: 已安装")
            
            if missing_models:
                print(f"⚠️ 发现 {len(missing_models)} 个缺失的模型:")
                for model in missing_models:
                    print(f"   - {model}")
                
                print("\n📥 开始自动下载缺失的模型...")
                
                # 自动下载所有缺失的模型
                for model_name in missing_models:
                    print(f"\n📥 下载 {model_name}...")
                    if manager.download_model(model_name):
                        print(f"✅ {model_name} 下载完成")
                    else:
                        print(f"❌ {model_name} 下载失败")
                        
                print("\n🎉 模型设置完成！")
            else:
                print("✅ 所有必需的模型都已安装")
                
        except ImportError:
            print("⚠️ 模型管理器不可用，跳过模型检查")
        except Exception as e:
            print(f"⚠️ 模型检查失败: {str(e)}")
        
    def separate_vocals(self, input_file, output_dir):
        """
        使用Spleeter分离人声和伴奏
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用绝对路径
        input_file = os.path.abspath(input_file)
        output_dir = os.path.abspath(output_dir)
        
        try:
            print(f"开始分离音轨: {input_file}")
            self.separator.separate_to_file(input_file, output_dir)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_path = os.path.join(output_dir, base_name)
            
            print(f"音轨分离完成，输出路径: {output_path}")
            
            # 2stems模型只分离人声和伴奏，我们将伴奏作为其他乐器
            vocals_path = os.path.join(output_path, 'vocals.wav')
            accompaniment_path = os.path.join(output_path, 'accompaniment.wav')
            
            # 检查文件是否存在
            if not os.path.exists(vocals_path):
                raise Exception(f"人声文件不存在: {vocals_path}")
            if not os.path.exists(accompaniment_path):
                raise Exception(f"伴奏文件不存在: {accompaniment_path}")
                
            # 只返回人声和伴奏两个轨道
            return {
                'vocals': vocals_path,
                'accompaniment': accompaniment_path
            }
        except Exception as e:
            print(f"音轨分离失败详细错误: {str(e)}")
            raise Exception(f"音轨分离失败: {str(e)}")
        
    def extract_pitch(self, audio_file, sr=None):
        """
        使用librosa提取音高和节拍信息
        """
        # 设置较低的采样率以加快处理速度
        target_sr = 22050  # 降低到22.05kHz（原始通常是44.1kHz）
        
        # 加载音频
        y, sr = librosa.load(audio_file, sr=target_sr)
        
        # 使用librosa进行音高检测
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # 获取每帧最强音高
        pitches_with_confidence = []
        times = []
        
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            magnitude = magnitudes[index, i]
            
            # 只保留有足够响度的音高
            if magnitude > np.mean(magnitudes) * 0.5:
                pitches_with_confidence.append(pitch)
                times.append(librosa.frames_to_time(i, sr=sr))
        
        time = np.array(times)
        frequency = np.array(pitches_with_confidence)
        
        # 使用librosa进行节拍检测
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        return time, frequency, beat_times, tempo
    
    def extract_lyrics_with_whisper(self, vocals_file):
        """
        使用本地Whisper模型提取歌词，不依赖网络连接
        需要安装 openai-whisper: pip install openai-whisper
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(vocals_file):
                print(f"❌ 错误: 人声文件不存在: {vocals_file}")
                return None
                
            # 检查文件大小
            file_size_mb = os.path.getsize(vocals_file) / (1024 * 1024)
            print(f"📊 人声文件大小: {file_size_mb:.2f} MB")
            
            # 预处理音频以提高识别率
            print("🔧 预处理音频以提高识别率...")
            try:
                # 加载音频
                audio = AudioSegment.from_file(vocals_file)
                
                # 1. 转换为单声道
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                    print("✅ 已转换为单声道")
                
                # 2. 调整采样率为16kHz (Whisper推荐)
                audio = audio.set_frame_rate(16000)
                
                # 3. 音量归一化 (增强信号)，针对中文语音优化
                audio = audio.normalize()
                
                # 4. 应用高通滤波器，减少低频噪音（针对中文语音优化）
                audio = audio.high_pass_filter(100)  # 提高截止频率，更好保留中文语音特征
                
                # 5. 应用低通滤波器，减少高频噪音（针对中文语音优化）
                audio = audio.low_pass_filter(6000)  # 降低截止频率，减少高频噪音干扰
                
                # 6. 增加动态范围压缩，提高语音清晰度
                try:
                    audio = audio.compress()
                    print("✅ 已应用动态范围压缩")
                except:
                    print("⚠️ 动态范围压缩不可用，跳过")
                
                # 保存预处理后的音频
                temp_file = "temp_processed_audio.wav"
                audio.export(temp_file, format="wav")
                print(f"✅ 音频预处理完成: {temp_file}")
            except Exception as preprocess_error:
                print(f"⚠️ 音频预处理失败: {str(preprocess_error)}")
                # 如果预处理失败，使用原始文件
                temp_file = vocals_file
            
            # 检查是否安装了whisper
            try:
                import whisper
                print("✅ Whisper模块已成功导入")
            except ImportError as e:
                print(f"❌ 警告: 未安装whisper模块，无法使用高级语音识别: {str(e)}")
                print("请运行: pip install openai-whisper")
                # 清理临时文件
                if temp_file != vocals_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return None
                
            print(f"📝 使用本地Whisper模型提取歌词...")
            
            # 设置环境变量，确保模型在本地下载和使用
            whisper_home = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            os.makedirs(whisper_home, exist_ok=True)
            os.environ["WHISPER_HOME"] = whisper_home
            print(f"📂 Whisper模型缓存目录: {whisper_home}")
            
            # 设置离线模式 - 确保使用本地模型
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            # 尝试多种模型大小，优先使用更准确的模型
            model_sizes = ["medium", "small", "base"]  # 修改模型加载顺序，优先使用medium模型
            model = None
            
            for model_size in model_sizes:
                try:
                    print(f"🔄 尝试加载Whisper {model_size}模型...")
                    model = whisper.load_model(model_size)
                    print(f"✅ Whisper {model_size}模型加载成功")
                    break
                except Exception as model_error:
                    print(f"⚠️ 加载{model_size}模型失败: {str(model_error)}")
            
            if model is None:
                print("❌ 所有模型加载均失败")
                # 清理临时文件
                if temp_file != vocals_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return None
            
            # 使用whisper识别
            print("🔍 开始使用Whisper识别歌词...")
            try:
                # 设置转录选项，针对中文优化
                options = {
                    "fp16": False,  # 使用FP32以提高兼容性
                    "task": "transcribe",  # 转录任务
                    "beam_size": 10,  # 增加beam size，提高识别准确性
                    "best_of": 5,
                    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8],  # 使用温度采样，增加多样性
                    "patience": 2.0,  # 增加patience值，提高准确性
                    "length_penalty": 1.0,  # 使用默认长度惩罚
                    "suppress_blank": True,
                    "suppress_tokens": [-1],  # 抑制空白标记
                    "condition_on_previous_text": True,  # 基于先前文本的条件生成，提高连贯性
                    "compression_ratio_threshold": 2.4,  # 降低压缩比阈值，接受更多变化
                    "logprob_threshold": -0.8,  # 提高对数概率阈值，过滤低质量结果
                    "no_speech_threshold": 0.4,  # 降低无语音阈值，提高灵敏度
                    "word_timestamps": True  # 启用词级时间戳
                }
                
                # 尝试多种语言设置和任务类型，优先中文
                language_combinations = [
                    ("zh", "transcribe"),  # 中文转录
                    ("zh", "translate"),  # 中文翻译
                    (None, "transcribe"),  # 自动检测语言，转录
                    ("en", "transcribe"),  # 英文转录
                    ("ja", "transcribe"),  # 日文转录
                    ("ko", "transcribe")  # 韩文转录
                ]
                
                best_result = None
                best_score = 0
                
                for lang, task in language_combinations:
                    if lang is not None:
                        options["language"] = lang
                    else:
                        options.pop("language", None)  # 移除语言设置，使用自动检测
                    
                    options["task"] = task
                    print(f"🔄 尝试使用 {task} 任务, 语言: {lang or 'auto'}")
                    
                    # 执行转录
                    result = model.transcribe(temp_file, **options)
                    
                    if result and "text" in result and result["text"].strip():
                        recognized_text = result["text"].strip()
                        print(f"✅ Whisper识别成功 ({task}, {lang or 'auto'}): {recognized_text}")
                        
                        # 对中文文本进行特殊处理
                        if lang == "zh" or (lang is None and any('\u4e00' <= char <= '\u9fff' for char in recognized_text[:10])):
                            # 如果是中文或检测到中文，清理文本格式
                            # 不再移除空格，保留原始格式
                            # 使用中文文本清理方法
                            cleaned_text = self._clean_chinese_text(recognized_text)
                            if cleaned_text:
                                recognized_text = cleaned_text
                                print(f"🔤 中文文本清理后: {recognized_text}")
                            else:
                                print("⚠️ 清理后文本为空，跳过此结果")
                                continue
                        
                        # 计算文本质量分数
                        quality_score = self._calculate_text_quality_score(recognized_text)
                        print(f"📊 文本质量分数: {quality_score:.2f}")
                        
                        # 如果质量分数高于当前最佳结果，更新最佳结果
                        if quality_score > best_score:
                            best_result = recognized_text
                            best_score = quality_score
                            print(f"🏆 更新最佳结果，当前最佳分数: {best_score:.2f}")
                        
                        # 如果质量分数很高，直接返回
                        if quality_score > 0.8:
                            print(f"🎯 高质量结果，直接返回")
                            # 清理临时文件
                            if temp_file != vocals_file and os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except:
                                    pass
                            return recognized_text
                    else:
                        print(f"⚠️ 使用 {task}, {lang or 'auto'} 设置未能识别出文本")
                
                # 返回最佳结果（如果有）
                if best_result and best_score > 0.3:
                    print(f"🏆 返回最佳识别结果，质量分数: {best_score:.2f}")
                    # 清理临时文件
                    if temp_file != vocals_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    return best_result
                
                print("❌ 所有语言和任务设置均未能识别出文本")
                
                # 尝试分析音频节拍，生成占位符歌词
                try:
                    print("🎵 尝试分析音频节拍...")
                    y, sr = librosa.load(temp_file, sr=16000)
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                    beat_times = librosa.frames_to_time(beats, sr=sr)
                    
                    if len(beat_times) > 0:
                        print(f"✅ 检测到 {len(beat_times)} 个节拍点，生成占位符歌词")
                        # 每4个节拍生成一个"♪"
                        placeholder_lyrics = "♪ " * (len(beat_times) // 4 + 1)
                        # 清理临时文件
                        if temp_file != vocals_file and os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                        return f"（检测到旋律，无法识别具体歌词）\n{placeholder_lyrics.strip()}"
                except Exception as beat_error:
                    print(f"⚠️ 节拍分析失败: {str(beat_error)}")
                
                # 清理临时文件
                if temp_file != vocals_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return None
            except Exception as transcribe_error:
                print(f"❌ Whisper转录失败: {str(transcribe_error)}")
                # 清理临时文件
                if temp_file != vocals_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return None
                
        except Exception as e:
            print(f"❌ 使用whisper提取歌词失败: {str(e)}")
            import traceback
            print(f"📋 详细错误信息: {traceback.format_exc()}")
            # 清理临时文件
            if 'temp_file' in locals() and temp_file != vocals_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return None
            
    def _validate_text_quality(self, text):
        """
        验证识别文本的质量，过滤低质量结果
        """
        if not text or len(text.strip()) < 2:
            return False
            
        text = text.strip()
        
        # 检查是否包含太多无意义字符
        meaningless_chars = ["[unk]", "[UNK]", "[unkown]"]
        meaningless_count = sum(text.count(char) for char in meaningless_chars)
        
        # 如果无意义字符占比超过50%，认为是低质量文本
        if meaningless_count / len(text) > 0.5:
            return False
            
        # 检查是否有太多重复字符
        import re
        repeated_chars = re.findall(r'(.)\1{3,}', text)  # 改为4次以上重复才过滤
        if len(repeated_chars) > len(text) * 0.3:  # 如果重复字符超过30%
            return False
            
        # 检查是否包含足够的中文字符（如果是中文文本）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if chinese_chars and len(chinese_chars) / len(text) < 0.1:  # 降低到10%中文字符要求
            return False
            
        # 检查文本长度是否合理
        if len(text) < 2 or len(text) > 1500:  # 放宽长度限制
            return False
            
        return True
    
    def _calculate_text_quality_score(self, text):
        """
        计算文本质量分数（0-1之间）
        """
        if not text or len(text.strip()) < 2:
            return 0.0
            
        text = text.strip()
        score = 1.0
        
        import re
        
        # 1. 长度分数（理想长度50-500字符）
        length = len(text)
        if length < 10:
            length_score = length / 10.0
        elif length > 500:
            length_score = max(0.5, 500.0 / length)
        else:
            length_score = 1.0
        score *= length_score
        
        # 2. 中文字符比例分数
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        chinese_ratio = len(chinese_chars) / len(text) if chinese_chars else 0
        chinese_score = min(1.0, chinese_ratio * 3)  # 中文字符越多越好
        score *= chinese_score
        
        # 3. 无意义字符惩罚
        meaningless_chars = ["[unk]", "[UNK]", "[unkown]", "[UNKNOWN]"]
        meaningless_count = sum(text.count(char) for char in meaningless_chars)
        meaningless_ratio = meaningless_count / len(text)
        meaningless_penalty = max(0.1, 1.0 - meaningless_ratio * 2)  # 无意义字符越多惩罚越大
        score *= meaningless_penalty
        
        # 4. 重复字符惩罚
        repeated_chars = re.findall(r'(.)\1{3,}', text)  # 4次以上重复
        repeated_ratio = len(repeated_chars) / len(text)
        repeated_penalty = max(0.3, 1.0 - repeated_ratio * 3)  # 重复字符越多惩罚越大
        score *= repeated_penalty
        
        # 5. 词汇多样性分数（基于唯一字符比例）
        unique_chars = len(set(text))
        diversity_ratio = unique_chars / len(text)
        diversity_score = min(1.0, diversity_ratio * 2)  # 多样性越高越好
        score *= diversity_score
        
        return max(0.0, min(1.0, score))
    
    def _clean_chinese_text(self, text):
        """
        清理中文识别文本，并优化分词
        """
        if not text:
            return ""
            
        # 移除[unk]等标记
        text = text.replace("[unk]", "").replace("[UNK]", "").replace("[unkown]", "").replace("[UNKNOWN]", "")
        
        # 移除连续出现的无意义词汇（哈哈、呵呵等叠词）
        import re
        meaningless_words = ["哈哈", "呵呵", "嘿嘿", "嘻嘻", "嗯嗯", "啊啊", "哦哦", "呜呜", "啦啦"]
        for word in meaningless_words:
            # 匹配连续出现2次以上的叠词
            pattern = f"({re.escape(word)})\\1+"
            text = re.sub(pattern, word, text)
        
        # 移除重复字符（超过3次重复的字符，保留2个）
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # 中文分词优化：在特定字符后添加空格
        # 1. 在标点符号后添加空格
        text = re.sub(r'([。！？，；：])', r'\1 ', text)
        # 2. 在常见连接词前后添加空格
        text = re.sub(r'(\w)([你我他她它这那])(\w)', r'\1 \2 \3', text)
        # 3. 移除多余空格，但保留换行符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 如果文本太短或为空，返回None
        if len(text) < 2:
            return None
            
        return text
    
    def extract_lyrics_with_vosk(self, vocals_file):
        """
        使用Vosk模型进行本地语音识别，提取歌词
        Vosk是一个轻量级的离线语音识别工具包，支持多种语言
        """
        try:
            # 检查是否安装了vosk
            try:
                from vosk import Model, KaldiRecognizer, SetLogLevel
                print("✅ Vosk模块已成功导入")
                # 设置Vosk日志级别为警告，减少输出
                SetLogLevel(-1)
            except ImportError as e:
                print(f"❌ 警告: 未安装vosk模块，无法使用Vosk语音识别: {str(e)}")
                print("请运行: pip install vosk")
                return None
                
            print(f"📝 使用Vosk模型提取歌词: {vocals_file}")
            
            # 检查文件是否存在
            if not os.path.exists(vocals_file):
                print(f"❌ 错误: 人声文件不存在: {vocals_file}")
                return None
            
            # 预处理音频文件
            print("🔧 预处理音频以适应Vosk模型...")
            temp_file = "temp_vosk_input.wav"
            
            try:
                # 加载音频
                audio = AudioSegment.from_file(vocals_file)
                
                # Vosk要求16kHz采样率，单声道，16位PCM
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio = audio.set_sample_width(2)  # 16位
                
                # 音量归一化
                audio = audio.normalize()
                
                # 针对中文语音的额外处理
                # 增加音量，确保语音清晰
                if audio.dBFS < -20:  # 如果音量太小
                    audio = audio + 10  # 增加10dB
                
                # 应用高通滤波器，去除低频噪声
                try:
                    from pydub.effects import high_pass_filter
                    audio = high_pass_filter(audio, 80)  # 80Hz高通滤波
                except:
                    pass  # 如果滤波器不可用，忽略错误
                
                # 保存临时文件
                audio.export(temp_file, format="wav")
                print(f"✅ 音频预处理完成: {temp_file}")
            except Exception as preprocess_error:
                print(f"⚠️ 音频预处理失败: {str(preprocess_error)}")
                return None
                
            # 检查模型目录
            model_path = os.path.join(os.getcwd(), "vosk-model")
            model_paths = {
                "zh": os.path.join(model_path, "vosk-model-small-cn-0.22"),
                "en": os.path.join(model_path, "vosk-model-small-en-us-0.15")
            }
            
            # 检查模型是否已下载
            models_available = []
            for lang, path in model_paths.items():
                if os.path.exists(path) and os.path.isdir(path):
                    models_available.append(lang)
                    
            if not models_available:
                print("⚠️ 未找到Vosk模型，将跳过Vosk识别")
                print("如果您想使用Vosk进行更准确的离线识别，请手动下载模型：")
                print("1. 从 https://alphacephei.com/vosk/models 或 https://huggingface.co/alphacep 下载模型")
                print("2. 解压到项目目录下的 vosk-model 文件夹中")
                print("3. 确保文件夹结构为 vosk-model/vosk-model-small-cn-0.22/ 或 vosk-model/vosk-model-small-en-us-0.15/")
                
                # 创建模型目录
                os.makedirs(model_path, exist_ok=True)
                
                # 返回None，让系统尝试其他识别方法
                return None
            
            # 使用可用的模型进行识别
            print(f"📊 可用的Vosk模型: {', '.join(models_available)}")
            
            results = []
            
            # 打开预处理后的音频文件
            wf = wave.open(temp_file, "rb")
            
            # 对每个可用模型进行识别
            for lang in models_available:
                try:
                    print(f"🔍 使用 {lang} 模型进行识别...")
                    model = Model(model_paths[lang])
                    
                    # 创建识别器，针对中文优化参数
                    if lang == "zh":
                        # 对于中文模型，使用更详细的配置
                        rec = KaldiRecognizer(model, wf.getframerate())
                        # 启用词级别时间戳和部分结果
                        rec.SetWords(True)
                        rec.SetPartialWords(True)
                        # 设置更精确的识别参数
                        try:
                            rec.SetMaxAlternatives(5)  # 获取更多候选结果
                        except:
                            try:
                                rec.SetMaxAlternatives(3)  # 如果5不支持，尝试3
                            except:
                                pass  # 如果不支持，忽略错误
                    else:
                        rec = KaldiRecognizer(model, wf.getframerate())
                        rec.SetWords(True)  # 启用词级别时间戳
                    
                    # 重置文件指针
                    wf.rewind()
                    
                    # 分块读取音频并识别
                    chunk_size = 8000  # 增加每次读取的样本数，提高识别准确性
                    text_parts = []
                    alternative_results = []
                    
                    while True:
                        data = wf.readframes(chunk_size)
                        if len(data) == 0:
                            break
                            
                        if rec.AcceptWaveform(data):
                            part_result = json.loads(rec.Result())
                            
                            # 处理主要结果
                            if "text" in part_result and part_result["text"].strip():
                                # 对中文文本进行清理，移除多余空格
                                cleaned_text = part_result["text"].strip()
                                if lang == "zh":
                                    # 中文文本不需要空格分隔
                                    cleaned_text = cleaned_text.replace(" ", "")
                                    # 过滤无意义词汇和[unk]标记
                                    cleaned_text = self._clean_chinese_text(cleaned_text)
                                if cleaned_text:  # 只添加非空文本
                                    text_parts.append(cleaned_text)
                            
                            # 处理候选结果（仅对中文）
                            if lang == "zh" and "alternatives" in part_result:
                                for alt in part_result["alternatives"]:
                                    if "text" in alt and alt["text"].strip():
                                        alt_text = alt["text"].strip().replace(" ", "")
                                        alt_text = self._clean_chinese_text(alt_text)
                                        if alt_text:
                                            alternative_results.append(alt_text)
                                
                    # 获取最终结果
                    final_result = json.loads(rec.FinalResult())
                    
                    # 处理主要最终结果
                    if "text" in final_result and final_result["text"].strip():
                        # 对中文文本进行清理，移除多余空格
                        cleaned_text = final_result["text"].strip()
                        if lang == "zh":
                            # 中文文本不需要空格分隔
                            cleaned_text = cleaned_text.replace(" ", "")
                            # 过滤无意义词汇和[unk]标记
                            cleaned_text = self._clean_chinese_text(cleaned_text)
                        if cleaned_text:  # 只添加非空文本
                            text_parts.append(cleaned_text)
                    
                    # 处理候选最终结果（仅对中文）
                    if lang == "zh" and "alternatives" in final_result:
                        for alt in final_result["alternatives"]:
                            if "text" in alt and alt["text"].strip():
                                alt_text = alt["text"].strip().replace(" ", "")
                                alt_text = self._clean_chinese_text(alt_text)
                                if alt_text:
                                    alternative_results.append(alt_text)
                        
                    # 合并识别结果
                    if text_parts:
                        if lang == "zh":
                            # 中文文本不需要空格分隔
                            full_text = "".join(text_parts)
                            
                            # 如果有候选结果且主要结果较短，尝试使用候选结果
                            if alternative_results and len(full_text) < 10:
                                # 选择最长的候选结果
                                best_alternative = max(alternative_results, key=len)
                                if len(best_alternative) > len(full_text):
                                    print(f"🔄 使用候选结果替代: {best_alternative}")
                                    full_text = best_alternative
                        else:
                            full_text = " ".join(text_parts)
                        print(f"✅ {lang} 模型识别结果: {full_text}")
                        results.append((lang, full_text))
                    else:
                        print(f"⚠️ {lang} 模型未能识别出文本")
                        
                except Exception as model_error:
                    print(f"❌ 使用 {lang} 模型识别失败: {str(model_error)}")
            
            # 关闭音频文件
            wf.close()
            
            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 已清理临时文件: {temp_file}")
                except:
                    pass
            
            # 返回最佳结果
            if results:
                # 优先选择中文结果，如果没有则选择英文结果
                zh_results = [text for lang, text in results if lang == "zh"]
                if zh_results:
                    return zh_results[0]
                return results[0][1]
            else:
                print("❌ 所有模型均未能识别出文本")
                return None
                
        except Exception as e:
            print(f"❌ 使用Vosk提取歌词失败: {str(e)}")
            import traceback
            print(f"📋 详细错误信息: {traceback.format_exc()}")
            # 清理临时文件
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return None
    
    def recognize_lyrics(self, vocals_file):
        """
        使用多种本地模型从人声中识别歌词，不依赖网络连接
        """
        try:
            print(f"🎤 开始从人声文件识别歌词: {vocals_file}")
            
            # 检查文件是否存在
            if not os.path.exists(vocals_file):
                print(f"❌ 错误: 人声文件不存在: {vocals_file}")
                return "（无法识别歌词：文件不存在）"
                
            # 检查文件格式
            if not vocals_file.lower().endswith(('.wav', '.mp3', '.flac')):
                print(f"⚠️ 警告: 不支持的音频格式: {vocals_file}")
                # 继续尝试，但可能会失败
            
            # 1. 首先尝试使用Vosk模型识别歌词
            print("🔍 尝试使用Vosk模型识别歌词...")
            vosk_lyrics = self.extract_lyrics_with_vosk(vocals_file)
            if vosk_lyrics:
                print("✅ 使用Vosk成功识别歌词")
                return vosk_lyrics
            
            # 2. 如果Vosk失败，尝试使用Whisper模型
            print("🔍 尝试使用Whisper模型识别歌词...")
            whisper_lyrics = self.extract_lyrics_with_whisper(vocals_file)
            if whisper_lyrics:
                print("✅ 使用Whisper成功识别歌词")
                return whisper_lyrics
            
            # 3. 如果两种模型都失败，尝试使用音频特征分析
            print("🔄 语音识别失败，尝试分析音频特征...")
            
            # 加载音频文件
            print("📂 加载音频文件进行特征分析...")
            try:
                # 使用librosa加载音频
                y, sr = librosa.load(vocals_file, sr=22050)
                
                # 计算音频特征
                print("🔍 进行更全面的音频特征分析...")
                
                # 1. 计算RMS能量
                rms = librosa.feature.rms(y=y)[0]
                mean_rms = np.mean(rms)
                max_rms = np.max(rms)
                
                # 2. 计算过零率
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                mean_zcr = np.mean(zcr)
                
                # 3. 计算频谱质心
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                mean_spectral_centroid = np.mean(spectral_centroid)
                
                # 4. 计算梅尔频率倒谱系数(MFCC)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                mfcc_vars = np.var(mfccs, axis=1)
                
                # 5. 计算谐波与噪声比
                # 使用谱质心的变化作为谐波性的简单估计
                harmonic_ratio = np.std(spectral_centroid) / (mean_spectral_centroid + 1e-10)
                
                print(f"📊 音频特征分析:")
                print(f"  - 平均RMS能量: {mean_rms:.6f}, 最大RMS能量: {max_rms:.6f}")
                print(f"  - 平均过零率: {mean_zcr:.6f}")
                print(f"  - 平均频谱质心: {mean_spectral_centroid:.2f}Hz")
                print(f"  - 谐波比估计: {harmonic_ratio:.6f}")
                print(f"  - MFCC[1-5]: {', '.join([f'{x:.4f}' for x in mfcc_means[1:6]])}")
                
                # 使用更宽松的阈值判断是否有人声
                # 1. 能量阈值降低，考虑最大值
                energy_check = mean_rms > 0.001 or max_rms > 0.01
                
                # 2. 人声通常在300-3400Hz范围内，但我们放宽范围
                freq_check = 200 < mean_spectral_centroid < 4000
                
                # 3. 人声的MFCC特征有特定模式
                mfcc_check = np.abs(mfcc_means[1]) > 10 or np.abs(mfcc_means[2]) > 15
                
                # 4. 谐波比检查
                harmonic_check = harmonic_ratio > 0.001
                
                # 综合判断
                has_vocals = energy_check and (freq_check or mfcc_check or harmonic_check)
                
                # 详细输出判断结果
                print(f"📋 人声检测结果:")
                print(f"  - 能量检查: {'通过' if energy_check else '未通过'}")
                print(f"  - 频率检查: {'通过' if freq_check else '未通过'}")
                print(f"  - MFCC检查: {'通过' if mfcc_check else '未通过'}")
                print(f"  - 谐波检查: {'通过' if harmonic_check else '未通过'}")
                print(f"  - 最终结果: {'检测到人声' if has_vocals else '未检测到人声'}")
                
                if has_vocals:
                    print("✅ 音频特征分析显示包含人声，但无法识别具体歌词")
                    # 尝试从音频特征推断歌词的韵律
                    # 检测节拍
                    try:
                        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                        beat_times = librosa.frames_to_time(beats, sr=sr)
                        
                        # 根据节拍数量生成占位符歌词
                        if len(beat_times) > 0:
                            print(f"🎵 检测到 {len(beat_times)} 个节拍点，生成占位符歌词")
                            # 每4个节拍生成一个"♪"
                            placeholder_lyrics = "♪ " * (len(beat_times) // 4 + 1)
                            return f"（检测到人声旋律，无法识别具体歌词）\n{placeholder_lyrics.strip()}"
                    except Exception as beat_error:
                        print(f"⚠️ 节拍检测失败: {str(beat_error)}")
                    
                    return "（检测到人声，但无法识别具体歌词）"
                else:
                    print("⚠️ 音频特征分析显示可能不包含人声或人声信号较弱")
                    # 由于我们已经知道这是人声轨道，所以即使分析结果为否，也返回一个更友好的消息
                    return "（人声信号较弱或特征不明显，建议检查音频质量）"
                    
            except Exception as analysis_error:
                print(f"❌ 音频特征分析失败: {str(analysis_error)}")
                return "（无法识别歌词：音频分析失败）"
                
        except Exception as e:
            print(f"❌ 歌词识别总体失败: {str(e)}")
            import traceback
            print(f"📋 详细错误信息: {traceback.format_exc()}")
            return "（无法识别歌词：处理过程出错）"
    
    def frequency_to_midi_note(self, frequency):
        """
        将频率转换为MIDI音符编号
        """
        return 69 + 12 * np.log2(frequency / 440.0)
    
    def frequency_to_numbered_notation(self, frequency):
        """
        将频率转换为简谱音符
        """
        # 获取MIDI音符编号
        midi_note = self.frequency_to_midi_note(frequency)
        
        # 将MIDI音符转换为简谱
        # 简谱使用1-7表示自然音阶
        midi_note_int = int(round(midi_note))
        octave = (midi_note_int - 60) // 12  # 以C4为中心音
        note_in_octave = (midi_note_int - 60) % 12
        
        # 简谱音符映射 (C4 = 1)
        numbered_mapping = {
            0: '1',  # C
            1: '#1',  # C#
            2: '2',  # D
            3: '#2',  # D#
            4: '3',  # E
            5: '4',  # F
            6: '#4',  # F#
            7: '5',  # G
            8: '#5',  # G#
            9: '6',  # A
            10: '#6',  # A#
            11: '7'   # B
        }
        
        # 获取简谱音符
        note = numbered_mapping[note_in_octave]
        
        # 添加八度标记
        if octave > 0:
            note = note + "'" * octave
        elif octave < 0:
            note = note + "," * abs(octave)
            
        return note
        
    def create_midi(self, time, frequency, beat_times, tempo, output_file):
        """
        创建MIDI文件，包含节拍信息
        """
        try:
            pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program)
            
            # 将频率转换为MIDI音符
            midi_notes = self.frequency_to_midi_note(frequency)
            
            # 创建音符事件
            current_note = None
            note_start_time = None
            
            for i, (t, note_value) in enumerate(zip(time, midi_notes)):
                # 确保我们处理的是数值而不是Note对象
                if isinstance(note_value, pretty_midi.Note):
                    print(f"警告：在索引{i}处发现Note对象而不是数值")
                    continue
                    
                if current_note is None:
                    current_note = note_value
                    note_start_time = t
                # 检查是否是新音符
                elif isinstance(current_note, (int, float)) and isinstance(note_value, (int, float)):
                    if abs(note_value - current_note) > 0.5:  # 新音符
                        if note_start_time is not None:
                            midi_note = pretty_midi.Note(
                                velocity=100,
                                pitch=int(round(current_note)),
                                start=note_start_time,
                                end=t
                            )
                            piano.notes.append(midi_note)
                        current_note = note_value
                        note_start_time = t
                else:
                    print(f"警告：在索引{i}处发现类型不匹配：current_note={type(current_note)}，note_value={type(note_value)}")
                    current_note = note_value if isinstance(note_value, (int, float)) else None
                    note_start_time = t
            
            # 添加最后一个音符
            if current_note is not None and note_start_time is not None and len(time) > 0:
                midi_note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(round(current_note)),
                    start=note_start_time,
                    end=time[-1] + 0.5  # 添加一点额外时间
                )
                piano.notes.append(midi_note)
            
            pm.instruments.append(piano)
            
            # 添加节拍标记
            for beat_time in beat_times:
                ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=beat_time)
                pm.time_signature_changes.append(ts)
            
            pm.write(output_file)
            return True
        except Exception as e:
            print(f"创建MIDI文件失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"创建MIDI文件失败: {str(e)}")
        
    def create_sheet_music(self, midi_file, output_file):
        """
        使用music21生成五线谱
        """
        score = converter.parse(midi_file)
        score.write('musicxml', fp=output_file)
    
    def create_numbered_notation(self, time, frequency, lyrics, output_file):
        """
        创建带歌词的简谱
        """
        # 将频率转换为简谱音符
        notes = [self.frequency_to_numbered_notation(f) for f in frequency]
        
        # 处理歌词
        if lyrics and lyrics != "♪ (无法识别歌词) ♪":
            # 使用结巴分词
            words = list(jieba.cut(lyrics))
            
            # 获取拼音
            pinyin_list = pinyin(lyrics, style=Style.NORMAL)
            pinyin_flat = [p[0] for p in pinyin_list]
            
            # 调整歌词长度以匹配音符长度
            if len(words) < len(notes):
                # 如果歌词少于音符，用空格填充
                words.extend([''] * (len(notes) - len(words)))
            elif len(words) > len(notes):
                # 如果歌词多于音符，截断
                words = words[:len(notes)]
        else:
            # 如果没有歌词，使用占位符
            words = ['♪'] * len(notes)
        
        # 创建简谱文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 简谱 (Numbered Musical Notation)\n\n")
            
            # 写入基本信息
            f.write("## 基本信息\n")
            f.write("- 调号: C调\n")
            f.write("- 拍号: 4/4\n\n")
            
            # 写入简谱和歌词
            f.write("## 旋律与歌词\n\n")
            f.write("```\n")
            
            # 每行最多显示16个音符
            notes_per_line = 16
            for i in range(0, len(notes), notes_per_line):
                # 写入音符行
                line_notes = notes[i:i+notes_per_line]
                f.write(" ".join(line_notes) + "\n")
                
                # 写入歌词行
                line_words = words[i:i+notes_per_line]
                f.write(" ".join(line_words) + "\n\n")
            
            f.write("```\n")
            
            f.write("\n## 说明\n")
            f.write("- 1=C, 2=D, 3=E, 4=F, 5=G, 6=A, 7=B\n")
            f.write("- #表示升号，b表示降号\n")
            f.write("- '表示高八度，,表示低八度\n")
            
        return output_file
    
    def process_instrument_tracks(self, tracks, output_dir):
        """
        处理乐器轨道，提取主旋律
        """
        try:
            results = {}
            
            # 获取伴奏文件
            accompaniment_file = tracks.get('accompaniment')
            if not accompaniment_file or not os.path.exists(accompaniment_file):
                print("未找到伴奏文件，尝试使用原始音频文件")
                # 如果没有分离的伴奏文件，尝试使用原始音频文件
                vocals_file = tracks.get('vocals')
                if vocals_file and os.path.exists(vocals_file):
                    accompaniment_file = vocals_file
                else:
                    print("未找到任何可用的音频文件")
                    return results
            
            # 提取音高和节拍信息
            time, frequency, beat_times, tempo = self.extract_pitch(accompaniment_file)
            if len(time) == 0 or len(frequency) == 0:
                print("无法提取音高信息")
                return results
            
            # 创建MIDI文件
            midi_path = os.path.join(output_dir, 'accompaniment.mid')
            self.create_midi(time, frequency, beat_times, tempo, midi_path)
            
            # 创建五线谱
            sheet_path = os.path.join(output_dir, 'accompaniment_sheet.pdf')
            self.create_sheet_music(midi_path, sheet_path)
            
            # 构建结果
            results['accompaniment'] = {
                'audio': accompaniment_file,
                'midi': midi_path,
                'sheet': sheet_path
            }
            
            return results
        except Exception as e:
            print(f"处理乐器轨道时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}
