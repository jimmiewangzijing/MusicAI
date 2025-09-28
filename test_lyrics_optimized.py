#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
from music_processor import MusicProcessor

def test_lyrics_recognition():
    """测试歌词识别功能"""
    print("=" * 50)
    print("测试歌词识别功能")
    print("=" * 50)
    
    # 初始化音乐处理器
    processor = MusicProcessor()
    
    # 查找最近的音频文件
    uploads_dir = "uploads"
    audio_files = []
    
    # 支持的音频格式
    extensions = ["*.mp3", "*.wav", "*.flac"]
    
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(uploads_dir, ext)))
    
    if not audio_files:
        print("❌ 未找到音频文件")
        return
    
    # 按修改时间排序，获取最新的文件
    audio_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    test_file = audio_files[0]
    
    print(f"📁 使用测试文件: {test_file}")
    
    # 测试Vosk识别
    print("\n1. 测试Vosk识别...")
    start_time = time.time()
    vosk_result = processor.extract_lyrics_with_vosk(test_file)
    vosk_time = time.time() - start_time
    
    if vosk_result:
        print(f"✅ Vosk识别结果 ({vosk_time:.2f}秒): {vosk_result}")
    else:
        print(f"❌ Vosk识别失败 ({vosk_time:.2f}秒)")
    
    # 测试Whisper识别
    print("\n2. 测试Whisper识别...")
    start_time = time.time()
    whisper_result = processor.extract_lyrics_with_whisper(test_file)
    whisper_time = time.time() - start_time
    
    if whisper_result:
        print(f"✅ Whisper识别结果 ({whisper_time:.2f}秒): {whisper_result}")
    else:
        print(f"❌ Whisper识别失败 ({whisper_time:.2f}秒)")
    
    # 测试完整识别流程
    print("\n3. 测试完整识别流程...")
    start_time = time.time()
    full_result = processor.recognize_lyrics(test_file)
    full_time = time.time() - start_time
    
    if full_result:
        print(f"✅ 完整识别结果 ({full_time:.2f}秒): {full_result}")
    else:
        print(f"❌ 完整识别失败 ({full_time:.2f}秒)")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_lyrics_recognition()