import os
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
from music21 import *
import matplotlib.pyplot as plt
from spleeter.separator import Separator
import whisper
import json

# 设置 ffmpeg 路径
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + r'C:\Users\Jimmie\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin'

class MusicProcessor:
    def __init__(self):
        # 设置模型路径
        model_path = os.path.expanduser('~/.cache/spleeter/models')
        pretrained_path = os.path.join(model_path, 'pretrained_models')
        
        # 确保模型目录存在
        if not os.path.exists(pretrained_path):
            os.makedirs(pretrained_path)
        
        # 初始化分离器，指定模型路径
        self.separator = Separator('spleeter:2stems', multiprocess=False)
        
        # 初始化语音识别模型
        self.whisper_model = whisper.load_model("base")
        
        # 简谱音符映射
        self.numbered_notes = {
            60: "1",  # C4 (中央C)
            62: "2",  # D4
            64: "3",  # E4
            65: "4",  # F4
            67: "5",  # G4
            69: "6",  # A4
            71: "7",  # B4
        }
        
    def recognize_lyrics(self, audio_file):
        """
        使用whisper识别音频中的歌词
        """
        print("正在识别歌词...")
        result = self.whisper_model.transcribe(audio_file, language='Chinese')
        
        # 获取带时间戳的歌词
        lyrics_with_timestamps = []
        for segment in result['segments']:
            lyrics_with_timestamps.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
            
        return lyrics_with_timestamps
        
    def midi_note_to_numbered(self, midi_note):
        """
        将MIDI音符转换为简谱记号
        """
        # 获取音高在一个八度内的值
        note_in_octave = midi_note % 12
        octave = midi_note // 12 - 5  # MIDI中60是中央C，对应第4个八度
        
        # 映射到简谱音符
        base_notes = {
            0: "1",   # C
            2: "2",   # D
            4: "3",   # E
            5: "4",   # F
            7: "5",   # G
            9: "6",   # A
            11: "7",  # B
        }
        
        if note_in_octave in [1, 3, 6, 8, 10]:  # 半音
            base_note = base_notes[note_in_octave - 1]
            return f"#{base_note}"
        
        note = base_notes.get(note_in_octave, "")
        
        # 添加八度标记
        if octave > 0:
            note = note + "'" * octave
        elif octave < 0:
            note = note + "," * abs(octave)
            
        return note
        
    def create_numbered_notation(self, midi_file, output_file, output_musicxml=None, lyrics_data=None):
        """
        生成简谱，同时支持文本格式和MusicXML格式，包含歌词
        """
        # 读取MIDI文件
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        
        # 获取所有音符
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'start': note.start,
                    'end': note.end,
                    'pitch': note.pitch,
                    'duration': note.end - note.start
                })
        
        # 按开始时间排序
        notes.sort(key=lambda x: x['start'])
        
        # 生成简谱文本
        notation = []
        current_measure = 1
        beats_per_measure = 4  # 假设是4/4拍
        
        # 为每个音符匹配歌词
        if lyrics_data:
            for note in notes:
                # 找到与音符时间最接近的歌词
                matching_lyric = None
                min_time_diff = float('inf')
                for lyric in lyrics_data:
                    time_diff = abs(note['start'] - lyric['start'])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        matching_lyric = lyric
                if matching_lyric and min_time_diff < 0.5:  # 只在时间差小于0.5秒时匹配
                    note['lyric'] = matching_lyric['text']
                else:
                    note['lyric'] = ''
        
        for note in notes:
            # 添加小节线
            measure = int(note['start'] * midi_data.get_tempo_changes()[0][0] / 60 / beats_per_measure) + 1
            if measure > current_measure:
                notation.append(" | ")
                if measure % 4 == 1:  # 每4小节换行
                    notation.append("\n")
                current_measure = measure
            
            # 转换音符
            numbered_note = self.midi_note_to_numbered(note['pitch'])
            
            # 添加时值记号（简化处理）
            duration = note['duration']
            if duration >= 2.0:
                numbered_note += "-"  # 长音
            elif duration <= 0.25:
                numbered_note += "."  # 短音
            
            # 添加歌词（如果有）
            if lyrics_data and 'lyric' in note and note['lyric']:
                numbered_note += f"({note['lyric']})"
            
            notation.append(numbered_note + " ")
        
        # 写入文本文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("简谱（带歌词）\n")
            f.write("节拍：4/4\n")
            f.write("速度：" + str(int(midi_data.get_tempo_changes()[0][0])) + "\n\n")
            f.write("".join(notation))
            
        # 如果指定了MusicXML输出文件，则生成MusicXML格式的简谱
        if output_musicxml:
            self.create_numbered_musicxml(notes, midi_data.get_tempo_changes()[0][0], output_musicxml)
            
    def create_numbered_musicxml(self, notes, tempo, output_file):
        """
        生成简谱的MusicXML文件，包含歌词
        """
        # 创建乐谱
        score = stream.Score()
        
        # 添加标题
        score.insert(0, metadata.Metadata())
        score.metadata.title = '简谱（带歌词）'
        
        # 创建声部
        main_part = stream.Part()
        
        # 添加拍号和速度标记
        time_signature = meter.TimeSignature('4/4')
        main_part.append(time_signature)
        
        # 创建速度标记
        from music21.tempo import MetronomeMark
        mm = MetronomeMark(number=float(tempo))
        main_part.append(mm)
        
        # 创建小节
        current_measure = stream.Measure()
        current_measure.timeSignature = time_signature
        
        # 处理每个音符
        for note_data in notes:
            # 创建音符
            pitch_obj = pitch.Pitch()
            pitch_obj.midi = note_data['pitch']
            
            # 计算音符时值
            duration_in_quarters = note_data['duration'] * 4  # 转换为四分音符单位
            if duration_in_quarters >= 2.0:
                note_type = 'half'
            elif duration_in_quarters <= 0.25:
                note_type = '16th'
            else:
                note_type = 'quarter'
                
            # 创建音符对象
            n = note.Note(pitch_obj, type=note_type)
            
            # 添加简谱记号作为第一个歌词
            numbered_note = self.midi_note_to_numbered(note_data['pitch'])
            n.addLyric(numbered_note)
            
            # 添加实际歌词作为第二个歌词（如果有）
            if 'lyric' in note_data and note_data['lyric']:
                n.addLyric(note_data['lyric'])
            
            # 将音符添加到小节
            current_measure.append(n)
            
            # 如果小节满了，添加到声部并创建新小节
            if current_measure.duration.quarterLength >= 4.0:
                main_part.append(current_measure)
                current_measure = stream.Measure()
                current_measure.timeSignature = time_signature
        
        # 添加最后一个小节（如果有内容）
        if len(current_measure) > 0:
            main_part.append(current_measure)
        
        # 将声部添加到乐谱
        score.append(main_part)
        
        # 保存为MusicXML文件
        score.write('musicxml', fp=output_file)
        
    def separate_vocals(self, input_file, output_dir):
        """
        使用Spleeter分离人声和伴奏
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用绝对路径
        input_file = os.path.abspath(input_file)
        output_dir = os.path.abspath(output_dir)
        
        try:
            self.separator.separate_to_file(input_file, output_dir)
            vocals_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0], 'vocals.wav')
            return vocals_path
        except Exception as e:
            raise Exception(f"人声分离失败: {str(e)}")
        
    # 修正后的extract_pitch方法
    def extract_pitch(self, audio_file, sr=None):
        """
        使用librosa提取音高和节拍信息
        """
        # 加载音频
        y, sr = librosa.load(audio_file, sr=sr)
        
        # 使用librosa进行音高检测
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # 获取每帧最强音高
        pitches_with_confidence = []
        times = []
        
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            magnitude = magnitudes[index, i]
            
            # 只保留有足够响度的音高
            if magnitude > np.mean(magnitudes) * 0.5:
                pitches_with_confidence.append(pitch)
                times.append(librosa.frames_to_time(i, sr=sr))
        
        time = np.array(times)
        frequency = np.array(pitches_with_confidence)
        
        # 使用librosa进行节拍检测
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        return time, frequency, beat_times, tempo
        
    def frequency_to_midi_note(self, frequency):
        """
        将频率转换为MIDI音符编号
        """
        return 69 + 12 * np.log2(frequency / 440.0)
        
    # 修正后的MIDI生成逻辑
    def create_midi(self, time, frequency, measure_starts, tempo, output_file, min_note_duration=0.3):
        """
        创建MIDI文件，包含节拍信息
        """
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        # 将频率转换为MIDI音符
        midi_notes = self.frequency_to_midi_note(frequency)
        
        # 创建音符事件
        current_note = None
        note_start_time = 0.0  # 显式初始化
        
        for i, (t, note_value) in enumerate(zip(time, midi_notes)):
            if current_note is None:
                current_note = note_value
                note_start_time = t
            else:
                duration = t - note_start_time
                if duration >= min_note_duration:
                    if abs(note_value - current_note) > 2.0:
                        new_note = pretty_midi.Note(
                            velocity=100,
                            pitch=int(round(current_note)),
                            start=note_start_time,
                            end=note_start_time + duration
                        )
                        piano.notes.append(new_note)
                        current_note = note_value
                        note_start_time = t
                
        pm.instruments.append(piano)
        
        # 添加节拍标记
        for beat_time in measure_starts:
            ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=beat_time)
            pm.time_signature_changes.append(ts)
        
        pm.write(output_file)
        
    def create_sheet_music(self, midi_file, output_file):
        """
        使用music21生成五线谱
        """
        score = converter.parse(midi_file)
        score.write('musicxml', fp=output_file)
        # 确保模型路径指向正确位置
        MODEL_PATH = './pretrained_models/2stems'
