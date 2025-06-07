# 音乐处理工具

这是一个功能强大的音乐处理工具，支持人声分离、音高检测和五线谱生成功能。

## 功能特点

- 人声分离：使用 Spleeter 将音乐分离为人声和伴奏
- 音高检测：使用 librosa 进行音高检测
- 五线谱生成：将音频转换为 MIDI 和五线谱

## 安装说明

1. 安装 Python 3.8 或更高版本

2. 安装 FFmpeg
   - Windows: 工具会自动下载和配置 FFmpeg
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

3. 安装依赖包
   ```bash
   pip install -r requirements.txt
   ```

4. 下载 Spleeter 模型
   ```bash
   python download_model.py
   ```

## 使用方法

1. 导入 MusicProcessor 类：
   ```python
   from music_processor import MusicProcessor
   processor = MusicProcessor()
   ```

2. 人声分离：
   ```python
   vocals_path = processor.separate_vocals('input.mp3', 'output_dir')
   ```

3. 音高检测：
   ```python
   time, frequency, beat_times, tempo = processor.extract_pitch('input.mp3')
   ```

4. 生成 MIDI 文件：
   ```python
   processor.create_midi(time, frequency, beat_times, tempo, 'output.mid')
   ```

5. 生成五线谱：
   ```python
   processor.create_sheet_music('output.mid', 'sheet_music.xml')
   ```

## 故障排除

1. 如果遇到 Spleeter 模型下载问题：
   - 运行 `download_model.py` 脚本
   - 脚本会自动使用国内镜像源下载模型

2. 如果遇到 FFmpeg 相关错误：
   - 确保 FFmpeg 已正确安装
   - 检查环境变量是否正确设置

3. 如果遇到内存不足错误：
   - 尝试处理较短的音频片段
   - 确保系统有足够的可用内存

## 注意事项

- 支持的输入音频格式：MP3, WAV, FLAC
- 生成的五线谱格式为 MusicXML，可以用 MuseScore 等软件打开
- 处理大文件时可能需要较长时间，请耐心等待

## 许可证

MIT License 