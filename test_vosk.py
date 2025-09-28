import os
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel

def test_vosk_chinese():
    """测试Vosk中文模型是否能正常工作"""
    print("🔍 测试Vosk中文模型...")
    
    # 检查模型路径
    model_path = os.path.join(os.getcwd(), "vosk-model", "vosk-model-small-cn-0.22")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ 中文模型目录不存在")
        return False
    
    try:
        # 设置Vosk日志级别
        SetLogLevel(-1)
        
        # 加载模型
        print("🔄 加载Vosk中文模型...")
        model = Model(model_path)
        print("✅ Vosk中文模型加载成功")
        
        # 创建识别器
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        
        # 如果有测试音频文件，可以在这里进行测试
        print("✅ Vosk中文模型配置正确")
        return True
        
    except Exception as e:
        print(f"❌ Vosk中文模型测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_vosk_chinese()