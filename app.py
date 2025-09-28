from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from music_processor import MusicProcessor
import uuid
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 存储任务状态
tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    allowed_extensions = {'mp3', 'wav', 'flac'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': '不支持的文件类型'}), 400
    
    # 生成唯一文件名
    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 创建任务ID并初始化状态
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'pending',
        'message': '任务已创建',
        'progress': 0,
        'file_path': file_path,
        'results': {}
    }
    
    # 启动后台处理线程
    thread = threading.Thread(target=process_audio, args=(task_id, file_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

def process_audio(task_id, file_path):
    """后台处理音频文件"""
    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['message'] = '正在处理音频...'
        
        # 创建输出目录
        task_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 初始化处理器
        processor = MusicProcessor()
        
        # 1. 分离音轨
        tasks[task_id]['message'] = '正在分离音轨...'
        tasks[task_id]['progress'] = 10
        tracks = processor.separate_vocals(file_path, os.path.join(task_output_dir, "separated"))
        
        # 2. 处理人声轨道
        tasks[task_id]['message'] = '正在处理人声轨道...'
        tasks[task_id]['progress'] = 30
        vocals_file = tracks['vocals']
        
        # 2.1 提取音高和节拍
        tasks[task_id]['message'] = '正在提取音高和节拍信息...'
        tasks[task_id]['progress'] = 40
        time_array, frequency, beat_times, tempo = processor.extract_pitch(vocals_file)
        
        # 2.2 识别歌词
        tasks[task_id]['message'] = '正在识别歌词...'
        tasks[task_id]['progress'] = 50
        lyrics = processor.recognize_lyrics(vocals_file)
        
        # 2.3 创建MIDI文件
        tasks[task_id]['message'] = '正在生成MIDI文件...'
        tasks[task_id]['progress'] = 60
        midi_file = os.path.join(task_output_dir, "vocals.mid")
        processor.create_midi(time_array, frequency, beat_times, tempo, midi_file)
        
        # 2.4 生成五线谱
        tasks[task_id]['message'] = '正在生成五线谱...'
        tasks[task_id]['progress'] = 70
        sheet_music_file = os.path.join(task_output_dir, "vocals.musicxml")
        processor.create_sheet_music(midi_file, sheet_music_file)
        
        # 2.5 生成带歌词的简谱
        tasks[task_id]['message'] = '正在生成带歌词的简谱...'
        tasks[task_id]['progress'] = 80
        numbered_notation_file = os.path.join(task_output_dir, "vocals_numbered.txt")
        processor.create_numbered_notation(time_array, frequency, lyrics, numbered_notation_file)
        
        # 3. 处理伴奏主旋律
        tasks[task_id]['message'] = '正在处理伴奏主旋律...'
        tasks[task_id]['progress'] = 90
        accompaniment_results = processor.process_instrument_tracks(tracks, task_output_dir)
        
        # 4. 收集结果
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = '处理完成'
        tasks[task_id]['progress'] = 100
        
        # 准备结果数据
        tasks[task_id]['results'] = {
            'vocals': {
                'audio': os.path.relpath(vocals_file, start=app.config['OUTPUT_FOLDER']),
                'midi': os.path.relpath(midi_file, start=app.config['OUTPUT_FOLDER']),
                'sheet': os.path.relpath(sheet_music_file, start=app.config['OUTPUT_FOLDER']),
                'numbered': os.path.relpath(numbered_notation_file, start=app.config['OUTPUT_FOLDER']),
                'lyrics': lyrics
            },
            'accompaniment': {}
        }
        
        # 添加伴奏结果
        # 确保accompaniment_results不为None且包含'accompaniment'键
        if accompaniment_results:
            if 'accompaniment' in accompaniment_results and accompaniment_results['accompaniment']:
                accompaniment_file = tracks.get('accompaniment')
                if accompaniment_file and os.path.exists(accompaniment_file):
                    tasks[task_id]['results']['accompaniment'] = {
                        'audio': os.path.relpath(accompaniment_file, start=app.config['OUTPUT_FOLDER']),
                        'midi': os.path.relpath(accompaniment_results['accompaniment']['midi'], start=app.config['OUTPUT_FOLDER']),
                        'sheet': os.path.relpath(accompaniment_results['accompaniment']['sheet'], start=app.config['OUTPUT_FOLDER'])
                    }
            # 如果accompaniment_results是一个字典且不包含'accompaniment'键，但包含其他轨道信息
            elif isinstance(accompaniment_results, dict) and accompaniment_results:
                # 取第一个轨道作为主旋律伴奏轨道
                first_track_key = next(iter(accompaniment_results))
                first_track = accompaniment_results[first_track_key]
                if isinstance(first_track, dict) and 'midi' in first_track:
                    tasks[task_id]['results']['accompaniment'] = {
                        'audio': '',  # 没有明确的音频文件路径
                        'midi': os.path.relpath(first_track['midi'], start=app.config['OUTPUT_FOLDER']),
                        'sheet': os.path.relpath(first_track.get('sheet', ''), start=app.config['OUTPUT_FOLDER'])
                    }
        else:
            # 如果没有成功处理伴奏，添加空对象避免前端报错
            tasks[task_id]['results']['accompaniment'] = {}
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'处理失败: {str(e)}'
        tasks[task_id]['progress'] = 0
        print(f"处理错误: {str(e)}")
        print(f"详细错误信息: {error_trace}")

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(tasks[task_id])

@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/visualize/<task_id>', methods=['GET'])
def visualize_results(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return jsonify({'error': '任务不存在或未完成'}), 404
    
    return render_template('visualize.html', task_id=task_id)

if __name__ == '__main__':
    app.run(debug=True)
