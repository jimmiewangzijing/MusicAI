import os
import sys
from spleeter.separator import Separator

def test_spleeter():
    print("测试 Spleeter 模型加载...")
    
    # 设置模型路径
    model_path = os.path.join(os.getcwd(), 'pretrained_models')
    os.environ['SPLEETER_MODELS_PATH'] = model_path
    
    print(f"模型路径: {model_path}")
    print(f"环境变量 SPLEETER_MODELS_PATH: {os.environ.get('SPLEETER_MODELS_PATH')}")
    
    # 检查2stems模型目录
    stems2_path = os.path.join(model_path, '2stems')
    if os.path.exists(stems2_path):
        print(f"2stems模型目录存在: {stems2_path}")
        files = os.listdir(stems2_path)
        print(f"目录内容: {files}")
    else:
        print(f"2stems模型目录不存在: {stems2_path}")
    
    try:
        # 尝试初始化分离器
        print("初始化Separator...")
        separator = Separator('spleeter:2stems')
        print("成功初始化Separator!")
        
        # 如果有音频文件，尝试分离
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            if os.path.exists(input_file):
                print(f"尝试分离音频文件: {input_file}")
                output_dir = "test_output"
                os.makedirs(output_dir, exist_ok=True)
                separator.separate_to_file(input_file, output_dir)
                print(f"分离完成，输出保存到: {output_dir}")
            else:
                print(f"文件不存在: {input_file}")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_spleeter() 