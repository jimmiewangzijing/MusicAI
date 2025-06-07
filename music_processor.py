import os
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
from music21 import *
import matplotlib.pyplot as plt
from spleeter.separator import Separator

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
        
    def create_midi(self, time, frequency, beat_times, tempo, output_file):
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
        note_start_time = None
        
        for t, note in zip(time, midi_notes):
            if current_note is None:
                current_note = note
                note_start_time = t
            elif abs(note - current_note) > 0.5:  # 新音符
                if note_start_time is not None:
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(round(current_note)),
                        start=note_start_time,
                        end=t
                    )
                    piano.notes.append(note)
                current_note = note
                note_start_time = t
                
        pm.instruments.append(piano)
        
        # 添加节拍标记
        for beat_time in beat_times:
            ts = pretty_midi.TimeSignature(numerator=4, denominator=4, time=beat_time)
            pm.time_signature_changes.append(ts)
        
        pm.write(output_file)
        
    def create_sheet_music(self, midi_file, output_file):
        """
        使用music21生成五线谱
        """
        score = converter.parse(midi_file)
        score.write('musicxml', fp=output_file) 