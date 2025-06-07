from music_processor import MusicProcessor
import os

def main():
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化处理器
    processor = MusicProcessor()
    
    # 输入音频文件路径
    input_file = input("请输入音频文件路径（支持MP3/WAV格式）：")
    
    try:
        # 1. 分离人声
        print("正在分离人声...")
        vocals_file = processor.separate_vocals(input_file, os.path.join(output_dir, "separated"))
        
        # 2. 提取音高和节拍
        print("正在提取音高和节拍信息...")
        time, frequency, beat_times, tempo = processor.extract_pitch(vocals_file)
    
        # 3. 创建MIDI文件
        print("正在生成MIDI文件...")
        midi_file = os.path.join(output_dir, "output.mid")
        processor.create_midi(time, frequency, beat_times, tempo, midi_file)
        
        # 4. 生成五线谱
        print("正在生成五线谱...")
        sheet_music_file = os.path.join(output_dir, "sheet_music.musicxml")
        processor.create_sheet_music(midi_file, sheet_music_file)
        
        print("\n处理完成！")
        print(f"输出文件位置：\n"
              f"- 分离的人声：{vocals_file}\n"
              f"- MIDI文件：{midi_file}\n"
              f"- 五线谱：{sheet_music_file}")
              
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main() 