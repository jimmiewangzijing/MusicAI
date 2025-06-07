import os
import sys
import requests
from tqdm import tqdm
import tarfile
import shutil

def download_file(url, filename):
    """
    使用 requests 下载文件，显示进度条
    """
    # 在requests.get中添加verify=False参数
    response = requests.get(url, stream=True, verify=False)
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

def setup_spleeter_model():
    """
    设置 Spleeter 模型
    """
    # 创建必要的目录
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, '.cache', 'spleeter')
    pretrained_dir = os.path.join(cache_dir, 'pretrained_models')
    
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # 模型文件 URL（使用国内镜像）
    # 将第XX行的下载链接改为原始地址
    MODEL_URL = 'https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz'
    model_file = "2stems.tar.gz"
    
    print("开始下载 Spleeter 模型...")
    try:
        # 下载模型文件
        download_file(model_url, model_file)
        
        print("解压模型文件...")
        # 解压文件
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(pretrained_dir)
        
        # 清理临时文件
        os.remove(model_file)
        
        print("模型设置完成！")
        print(f"模型文件保存在: {pretrained_dir}")
        
    except Exception as e:
        print(f"下载或解压过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_spleeter_model()