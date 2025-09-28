import os
import sys
import requests
from tqdm import tqdm
import tarfile
import shutil
import json

class ModelManager:
    """
    模型管理器：负责下载和管理预训练模型
    """
    
    def __init__(self):
        self.models_config = {
            "spleeter_2stems": {
                "url": "https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz",
                "filename": "2stems.tar.gz",
                "target_dir": "pretrained_models/2stems",
                "required_files": ["checkpoint", "model.data-00000-of-00001", "model.index", "model.meta"]
            },
            "spleeter_5stems": {
                "url": "https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz",
                "filename": "5stems.tar.gz",
                "target_dir": "pretrained_models/5stems",
                "required_files": ["checkpoint", "model.data-00000-of-00001", "model.index", "model.meta"]
            },
            "vosk_cn": {
                "url": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
                "filename": "vosk-model-small-cn-0.22.zip",
                "target_dir": "vosk-model/vosk-model-small-cn-0.22",
                "required_files": ["am/final.mdl", "graph/HCLG.fst", "ivector/final.ie"]
            },
            "vosk_en": {
                "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
                "filename": "vosk-model-small-en-us-0.15.zip",
                "target_dir": "vosk-model/vosk-model-small-en-us-0.15",
                "required_files": ["am/final.mdl", "graph/HCLG.fst"]
            }
        }
        
        # 创建必要的目录
        os.makedirs("pretrained_models", exist_ok=True)
        os.makedirs("vosk-model", exist_ok=True)
    
    def download_file(self, url, filename):
        """
        下载文件并显示进度条
        """
        try:
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            return True
        except Exception as e:
            print(f"下载文件失败: {str(e)}")
            return False
    
    def extract_archive(self, filename, target_dir):
        """
        解压文件
        """
        try:
            if filename.endswith('.tar.gz'):
                with tarfile.open(filename, 'r:gz') as tar:
                    tar.extractall(target_dir)
            elif filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
            else:
                print(f"不支持的文件格式: {filename}")
                return False
            
            return True
        except Exception as e:
            print(f"解压文件失败: {str(e)}")
            return False
    
    def check_model_exists(self, model_name):
        """
        检查模型是否已存在
        """
        if model_name not in self.models_config:
            return False
        
        config = self.models_config[model_name]
        target_dir = config["target_dir"]
        required_files = config["required_files"]
        
        # 检查目标目录是否存在
        if not os.path.exists(target_dir):
            return False
        
        # 检查所有必需文件是否存在
        for file_path in required_files:
            full_path = os.path.join(target_dir, file_path)
            if not os.path.exists(full_path):
                return False
        
        return True
    
    def download_model(self, model_name):
        """
        下载指定模型
        """
        if model_name not in self.models_config:
            print(f"未知的模型: {model_name}")
            return False
        
        config = self.models_config[model_name]
        
        # 检查模型是否已存在
        if self.check_model_exists(model_name):
            print(f"模型 {model_name} 已存在，跳过下载")
            return True
        
        print(f"开始下载 {model_name} 模型...")
        
        # 下载模型文件
        if not self.download_file(config["url"], config["filename"]):
            return False
        
        # 解压文件
        print(f"解压 {model_name} 模型...")
        if not self.extract_archive(config["filename"], config["target_dir"]):
            return False
        
        # 清理临时文件
        try:
            os.remove(config["filename"])
        except:
            pass
        
        print(f"{model_name} 模型下载完成！")
        return True
    
    def download_all_models(self):
        """
        下载所有模型
        """
        print("开始下载所有必需的预训练模型...")
        
        success_count = 0
        for model_name in self.models_config:
            if self.download_model(model_name):
                success_count += 1
        
        print(f"模型下载完成！成功下载 {success_count}/{len(self.models_config)} 个模型")
        return success_count == len(self.models_config)
    
    def get_model_status(self):
        """
        获取所有模型的状态
        """
        status = {}
        for model_name in self.models_config:
            status[model_name] = {
                "exists": self.check_model_exists(model_name),
                "config": self.models_config[model_name]
            }
        return status

def setup_models():
    """
    设置所有必需的模型
    """
    manager = ModelManager()
    
    print("检查模型状态...")
    status = manager.get_model_status()
    
    # 显示模型状态
    for model_name, info in status.items():
        status_text = "已安装" if info["exists"] else "未安装"
        print(f"{model_name}: {status_text}")
    
    # 询问用户是否下载缺失的模型
    missing_models = [name for name, info in status.items() if not info["exists"]]
    
    if not missing_models:
        print("所有模型都已安装！")
        return True
    
    print(f"\n发现 {len(missing_models)} 个缺失的模型:")
    for model in missing_models:
        print(f"  - {model}")
    
    response = input("\n是否下载缺失的模型？(y/n): ").strip().lower()
    if response in ['y', 'yes', '是']:
        return manager.download_all_models()
    else:
        print("跳过模型下载")
        return False

if __name__ == "__main__":
    setup_models()