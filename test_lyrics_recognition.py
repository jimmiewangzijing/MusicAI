import os
import sys
from music_processor import MusicProcessor

def test_lyrics_recognition():
    """测试歌词识别功能"""
    print("🎵 测试歌词识别功能...")
    
    # 检查是否有测试音频文件
    test_files = []
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                test_files.append(os.path.join("uploads", file))
    
    if os.path.exists("output"):
        for root, dirs, files in os.walk("output"):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    test_files.append(os.path.join(root, file))
    
    if not test_files:
        print("⚠️ 未找到测试音频文件，请先上传或放置音频文件到uploads或output目录")
        return
    
    print(f"📁 找到 {len(test_files)} 个测试文件:")
    for i, file in enumerate(test_files):
        print(f"  {i+1}. {file}")
    
    # 选择第一个文件进行测试
    test_file = test_files[0]
    print(f"\n🔍 选择文件进行测试: {test_file}")
    
    try:
        # 初始化处理器
        processor = MusicProcessor()
        
        # 测试Vosk识别
        print("\n=== 测试Vosk识别 ===")
        vosk_result = processor.extract_lyrics_with_vosk(test_file)
        print(f"Vosk识别结果: {vosk_result}")
        
        # 测试Whisper识别
        print("\n=== 测试Whisper识别 ===")
        whisper_result = processor.extract_lyrics_with_whisper(test_file)
        print(f"Whisper识别结果: {whisper_result}")
        
        # 测试完整识别流程
        print("\n=== 测试完整识别流程 ===")
        final_result = processor.recognize_lyrics(test_file)
        print(f"最终识别结果: {final_result}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lyrics_recognition()