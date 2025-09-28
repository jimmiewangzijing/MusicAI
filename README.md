# 音乐处理工具 Web 版

这是一个基于 Web 的多功能音乐处理工具，旨在自动化音乐分析和转录的核心流程。用户可以通过一个简洁的网页界面上传音频文件，系统将自动执行音轨分离、旋律提取、歌词识别，并生成多种格式的乐谱文件。

## ✨ 核心功能

- **🎵 音轨分离**: 采用 Spleeter 的 `2stems` 模型，将上传的音频文件精准分离为人声（vocals）和伴奏（accompaniment）两个独立的音轨。
- **🎤 人声分析**: 
  - **旋律提取**: 对分离出的人声轨道进行深度分析，使用 `librosa` 库提取其详细的音高（pitch）和节拍（beat）信息。
  - **多重歌词识别**:
    - **Vosk 本地识别**: 使用轻量级的 Vosk 模型进行完全离线的歌词识别，支持中文和英文。
    - **Whisper 本地识别**: 集成 OpenAI Whisper 模型进行高精度的本地歌词识别，无需网络连接。
    - **音频特征分析**: 当无法识别具体歌词时，分析音频特征和节拍，生成韵律占位符。
- **🎸 伴奏主旋律分析**: 从伴奏轨道中提取主旋律，捕捉关键的音乐主题和旋律线。
- **🎼 乐谱生成**:
  - **MIDI 文件**: 为人声和伴奏主旋律分别生成标准的 MIDI 文件，可用于音乐制作和编辑。
  - **五线谱 (MusicXML)**: 将 MIDI 文件转换为 MusicXML 格式，方便导入到专业打谱软件中。
  - **带歌词的简谱**: 为人声轨道生成包含音符和对应歌词的文本简谱，便于演唱和学习。
- **🌐 Web 用户界面**:
  - **文件上传**: 支持 `.mp3`, `.wav`, `.flac` 等多种主流音频格式。
  - **异步处理**: 文件上传后，后端进行异步处理，避免长时间的页面等待。
  - **实时进度反馈**: 通过进度条和状态信息，实时向用户展示处理进度。
  - **结果可视化**: 处理完成后，在专门的结果页面展示所有可用的文件，包括音频预览、歌词文本以及各种乐谱的下载链接。

## 🛠️ 技术栈

- **后端**: Python 3.8, Flask
- **音轨分离**: Spleeter (优化使用2stems模型)
- **音频分析**: Librosa, NumPy (优化使用22.05kHz采样率)
- **乐谱处理**: PrettyMIDI, Music21
- **歌词识别**: 
  - **Vosk**: 轻量级离线语音识别，支持多语言
  - **OpenAI Whisper**: 高精度本地语音识别 (tiny/base模型)
  - **音频特征分析**: 使用MFCC、频谱质心等特征进行人声检测
- **前端**: HTML, CSS, JavaScript, Bootstrap
- **核心依赖**: FFmpeg

## 📂 项目结构

```
.
├── app.py                  # Flask Web应用主文件
├── music_processor.py      # 核心音频处理逻辑模块
├── download_model.py       # Spleeter 模型下载脚本
├── download_vosk_models.py # Vosk 语音识别模型下载脚本
├── requirements.txt        # Python 依赖列表
├── templates/
│   ├── index.html          # 应用首页 (文件上传)
│   └── visualize.html      # 结果展示页面
├── static/
│   ├── css/style.css       # 自定义样式
│   └── js/
│       ├── main.js         # 首页的交互逻辑
│       └── visualize.js    # 结果页面的交互逻辑
├── uploads/                # 存放用户上传的音频文件
├── output/                 # 存放所有处理后的结果文件
├── pretrained_models/      # 存放 Spleeter 预训练模型
└── vosk-model/             # 存放 Vosk 语音识别模型
    ├── vosk-model-small-zh-cn-0.22/  # 中文识别模型
    └── vosk-model-small-en-us-0.15/  # 英文识别模型
```

## 🚀 安装与启动

### 功能与性能优化说明

本项目经过精简和优化，专注于核心功能并提高处理速度：

1. **简化音轨处理**：只分离和分析人声与伴奏主旋律，不再按不同乐器类型进行细分，减少了不必要的处理时间。
2. **使用 Spleeter 的 2stems 模型**：相比 5stems 模型，内存占用更少，处理速度更快。
3. **降低音频采样率**：从标准的 44.1kHz 降至 22.05kHz，显著减少处理数据量。
4. **多种本地歌词识别模型**：
   - **Vosk 模型**：轻量级，启动快，占用资源少，适合简单识别场景
   - **Whisper 模型**：精度高，但资源占用较大，作为备选方案

这些优化和简化可以将处理时间减少约 50-70%，同时保持良好的输出质量，专注于用户最需要的功能。

### Vosk 模型安装

为了使用 Vosk 进行本地歌词识别，您需要下载相应的语音模型：

1. 运行提供的下载脚本：
   ```bash
   python download_vosk_models.py
   ```

2. 默认情况下，脚本会下载中文和英文两种语言模型。如果只需要其中一种，可以指定：
   ```bash
   # 只下载中文模型
   python download_vosk_models.py --lang zh
   
   # 只下载英文模型
   python download_vosk_models.py --lang en
   ```

3. 模型会被下载到项目目录下的 `vosk-model` 文件夹中，大小约为每个模型 40MB。

4. 下载完成后，系统会自动检测并使用这些模型进行歌词识别。

### 1. 环境准备

- **Python**: 确保您已安装 Python 3.8 或更高版本。
- **FFmpeg**: 这是一个核心依赖，必须提前安装。
  - **自动安装**: 项目包含了一个便捷的下载器。在项目根目录下运行以下命令即可自动下载和配置：
    ```bash
    python -m ffmpeg_downloader install
    ```
  - **手动安装**: 您也可以从 [FFmpeg 官网](https://ffmpeg.org/download.html) 自行下载并将其 `bin` 目录添加到系统环境变量 `PATH` 中。

### 2. 项目设置

1. **克隆或下载项目**
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. **创建并激活虚拟环境** (推荐)
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **安装 Python 依赖**
   建议使用国内镜像源以加快下载速度。
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   *注意：`openai-whisper` 可能需要一些时间来安装。*

4. **下载 Spleeter 模型**
   运行 `download_model.py` 脚本以下载 `2stems` 模型。脚本会自动将模型下载到 `pretrained_models` 目录中。
   ```bash
   python download_model.py
   ```

### 3. 启动应用

一切准备就绪后，运行以下命令启动 Flask Web 服务器：
```bash
python app.py
```

服务器启动后，您会看到类似以下的输出：
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

## 🖥️ 如何使用

1. **访问主页**: 打开浏览器，访问 `http://127.0.0.1:5000`。
2. **上传文件**: 点击 "选择文件" 按钮，选择一个支持的音频文件，然后点击 "上传并处理"。
3. **查看进度**: 上传成功后，页面下方会显示处理状态和进度条。请耐心等待处理完成。
4. **查看结果**: 处理完成后，"查看处理结果" 按钮会变为可用。点击该按钮，页面将跳转到结果详情页。
5. **下载与预览**: 在结果页面，您可以：
   - 在线播放分离出的人声音轨和伴奏。
   - 查看识别出的歌词。
   - 点击链接下载所有生成的乐谱文件（.mid, .musicxml, .txt）。

---
希望这份文档能帮助您更好地理解和使用这个项目！ 