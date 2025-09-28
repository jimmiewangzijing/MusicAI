#!/usr/bin/env python3
"""
MusicAI 项目安装脚本
自动安装依赖并下载必需的预训练模型
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 需要 Python 3.8 或更高版本")
        sys.exit(1)
    print("✅ Python 版本检查通过")

def install_requirements():
    """安装依赖包"""
    print("\n📦 安装项目依赖...")
    
    try:
        # 使用pip安装requirements.txt中的包
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        sys.exit(1)

def setup_models():
    """设置预训练模型"""
    print("\n🤖 设置预训练模型...")
    
    try:
        # 导入模型管理器
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
            print(f"\n📥 发现 {len(missing_models)} 个缺失的模型，开始下载...")
            
            # 下载所有缺失的模型
            success_count = 0
            for model_name in missing_models:
                print(f"\n📥 下载 {model_name}...")
                if manager.download_model(model_name):
                    print(f"✅ {model_name} 下载完成")
                    success_count += 1
                else:
                    print(f"❌ {model_name} 下载失败")
            
            if success_count == len(missing_models):
                print(f"\n🎉 所有模型下载完成！")
            else:
                print(f"\n⚠️ 部分模型下载失败，请检查网络连接后重试")
        else:
            print("✅ 所有必需的模型都已安装")
            
    except ImportError as e:
        print(f"❌ 无法导入模型管理器: {e}")
        print("请确保已安装所有依赖包")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 模型设置失败: {e}")
        sys.exit(1)

def test_installation():
    """测试安装是否成功"""
    print("\n🧪 测试安装...")
    
    # 测试关键模块是否可导入
    modules_to_test = [
        "librosa", "soundfile", "pretty_midi", "music21",
        "spleeter", "speech_recognition", "pydub",
        "whisper"
    ]
    
    success_count = 0
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: 导入成功")
            success_count += 1
        except ImportError:
            print(f"❌ {module_name}: 导入失败")
    
    if success_count == len(modules_to_test):
        print("✅ 所有关键模块导入成功")
    else:
        print(f"⚠️ 部分模块导入失败 ({success_count}/{len(modules_to_test)})")

def main():
    """主安装流程"""
    print("🎵 MusicAI 项目安装程序")
    print("=" * 50)
    
    # 检查Python版本
    check_python_version()
    
    # 安装依赖
    install_requirements()
    
    # 设置模型
    setup_models()
    
    # 测试安装
    test_installation()
    
    print("\n" + "=" * 50)
    print("🎉 安装完成！")
    print("\n📖 使用说明:")
    print("1. 运行 Web 应用: python app.py")
    print("2. 测试功能: python test_lyrics_optimized.py")
    print("3. 查看 README.md 获取详细文档")
    print("\n💡 提示: 首次运行时可能需要下载额外的模型文件")

if __name__ == "__main__":
    main()