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
    input_file = input("请输入音频文件路径（支持MP3/WAV/FLAC格式）：")
    
    try:
        # 1. 分离音轨（人声和各种乐器）
        print("正在分离音轨...")
        tracks = processor.separate_vocals(input_file, os.path.join(output_dir, "separated"))
        
        # 2. 处理人声轨道
        print("\n处理人声轨道...")
        vocals_file = tracks['vocals']
        
        # 2.1 提取音高和节拍
        print("正在提取音高和节拍信息...")
        time, frequency, beat_times, tempo = processor.extract_pitch(vocals_file)
        
        # 2.2 识别歌词
        print("正在识别歌词...")
        lyrics = processor.recognize_lyrics(vocals_file)
        print(f"识别到的歌词: {lyrics[:100]}..." if len(lyrics) > 100 else f"识别到的歌词: {lyrics}")
        
        # 2.3 创建MIDI文件
        print("正在生成MIDI文件...")
        midi_file = os.path.join(output_dir, "vocals.mid")
        processor.create_midi(time, frequency, beat_times, tempo, midi_file)
        
        # 2.4 生成五线谱
        print("正在生成五线谱...")
        sheet_music_file = os.path.join(output_dir, "vocals.musicxml")
        processor.create_sheet_music(midi_file, sheet_music_file)
        
        # 2.5 生成带歌词的简谱
        print("正在生成带歌词的简谱...")
        numbered_notation_file = os.path.join(output_dir, "vocals_numbered.txt")
        processor.create_numbered_notation(time, frequency, lyrics, numbered_notation_file)
        
        # 3. 处理乐器轨道
        print("\n处理乐器轨道...")
        instrument_results = processor.process_instrument_tracks(tracks, output_dir)
        
        # 4. 输出结果
        print("\n处理完成！")
        print(f"输出文件位置：\n")
        print("人声轨道：")
        print(f"- 分离的人声：{vocals_file}")
        print(f"- MIDI文件：{midi_file}")
        print(f"- 五线谱：{sheet_music_file}")
        print(f"- 带歌词简谱：{numbered_notation_file}")
        
        print("\n乐器轨道：")
        for name, files in instrument_results.items():
            print(f"- {name}:")
            print(f"  - 音频文件：{tracks[name]}")
            print(f"  - MIDI文件：{files['midi']}")
            print(f"  - 五线谱：{files['sheet']}")
              
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()