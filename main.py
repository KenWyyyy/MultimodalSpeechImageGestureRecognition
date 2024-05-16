import gensim
import scipy.io.wavfile as wav
import cv2
import mediapipe as mp
import math
import tkinter as tk
import numpy as np
import sys
from faster_whisper import WhisperModel
import sounddevice as sd
import noisereduce as nr


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert('end', message)
        self.text_widget.see('end')
# 创建Tkinter窗口
window = tk.Tk()

# 设置字体样式和大小
font_style = ('微软雅黑', 20)

# 设置窗口标题
window.title('消息提示：')

# 创建文本框用于显示输出，并设置字体
output_text = tk.Text(window, font=font_style)
output_text.pack()

# 重定向标准输出到文本框
sys.stdout = StdoutRedirector(output_text)

def yuyin(gesture_str):
    # 定义录音参数
    duration = 2  # 录音时间，单位为秒
    sample_rate = 16000  # 采样率


    def record_audio(duration,sample_rate=16000):
        class AudioRecorder:
            def __init__(self, root, sample_rate):
                self.root = root
                self.sample_rate = sample_rate
                self.recording = False
                self.audio_data = np.array([], dtype='int16')
                self.stream = None

                # 设置录音按钮
                self.start_button = tk.Button(self.root, text='开始录音', command=self.start_recording)
                self.start_button.pack()

                # 设置停止按钮
                self.stop_button = tk.Button(self.root, text='停止录音', command=self.stop_recording)
                self.stop_button.pack()

            def start_recording(self):
                self.recording = True
                print("开始录音...")
                #window.mainloop()
                self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.callback)
                self.stream.start()

            def stop_recording(self):
                self.recording = False
                self.stream.stop()
                print("录音结束。")
                self.audio_data = nr.reduce_noise(y=self.audio_data, sr=self.sample_rate)
                self.root.destroy()
                self.root.quit()  # 结束Tkinter事件循环

            def callback(self, indata, frame_count, time_info, status):
                if self.recording:
                    self.audio_data = np.append(self.audio_data, indata.flatten(), axis=0)

        # 创建Tkinter窗口
        root = tk.Tk()
        root.title('录音机')

        # 实例化AudioRecorder
        recorder = AudioRecorder(root, sample_rate)

        # 运行Tkinter事件循环
        root.mainloop()

        # 窗口关闭后，返回录音数据
        return recorder.audio_data

    def save_audio_to_file(audio, sample_rate, file_path):
        # 将录音保存为WAV文件
        wav.write(file_path, sample_rate, audio)

    # 定义转录音频的函数
    def transcribe_audio(model, audio_path):
        # 使用faster-whisper进行转录
        segments, _ = model.transcribe(audio_path,language="en")

        # 拼接所有转录的文本段落
        transcription = ' '.join(segment.text for segment in segments)
        return transcription


    # 录音并保存到本地文件
    p = False
    while not p:
        audio = record_audio(duration, sample_rate)
        audio_file_path = "local_audio.wav"
        save_audio_to_file(audio, sample_rate, audio_file_path)

        # 翻译音频并输出结果
        input_sentence = transcribe_audio(model, audio_file_path).strip(".").strip()

        if (input_sentence == "1"): input_sentence = "one"
        if (input_sentence == "2"): input_sentence = "two"
        if (input_sentence == "3"): input_sentence = "three"
        if (input_sentence == "5"): input_sentence = "five"
        if (input_sentence == "6"): input_sentence = "six"

        print(f"语音识别结果为:{input_sentence}")

        if input_sentence in model1:

            # input_sentence = "下"
            fixed_words = ["one", "two", "three", "five", "six", "Thumbs_Up", "gun", "love","fist"]  # 9个固定词语
            q= model1.similarity(gesture_str, input_sentence)
            t=0

            for fixed_word in fixed_words:
                if (q < model1.similarity( fixed_word , input_sentence)): t=1
                print(f"{fixed_word} 和 {input_sentence} 的相似度 {model1.similarity( fixed_word , input_sentence)}")
            if (t==0):
                print(f"输入语音单词为：{input_sentence}，视频手势识别为：{gesture_str}，相似度在所有词库单词中匹配度最高为{q}，可以认为手势为{gesture_str}")
                p=True
            else:
                print("匹配失败，马上重新开始录音")
        else:
            print("必须读入一个单词，马上重新开始录音")

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # fist five gun love one six three thumbup yeah
    '''
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "two"
    return gesture_str

def detect():

    # 导入绘图工具和手部解决方案
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # 初始化手部追踪
    hands = mp_hands.Hands(
        static_image_mode=False,  # 非静态图像模式
        max_num_hands=1,  # 最大检测手数为1
        min_detection_confidence=0.75,  # 最小检测置信度
        min_tracking_confidence=0.75)  # 最小追踪置信度

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 初始化手势字符串和计数器
    prev_gesture_str = ""
    count_same_gesture = 0

    # 主循环
    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
        frame = cv2.flip(frame, 1)  # 水平翻转帧

        # 处理帧以检测手势
        results = hands.process(frame)

        # 将帧转回BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 默认手势为'None'
        gesture_str = "None"

        # 如果检测到手势
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 存储局部手部坐标
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))

                # 如果有手部坐标
                if hand_local:
                    angle_list = hand_angle(hand_local)  # 计算角度列表
                    gesture_str = h_gesture(angle_list)  # 获取手势字符串

                    # 在帧上绘制手势字符串
                    cv2.putText(frame, gesture_str, (0, 100), 0, 1.3, (0, 0, 255), 3)

        # 显示处理后的帧
        cv2.imshow('MediaPipe Hands', frame)

        # 检查手势字符串是否不为"None"且与前一个手势字符串相同
        if gesture_str != "None" and gesture_str == prev_gesture_str:
            count_same_gesture += 1
        else:
            count_same_gesture = 0

        # 更新前一个手势字符串
        prev_gesture_str = gesture_str

        # 如果相同手势字符串出现了60次，则调用yuyin函数
        if count_same_gesture == 60:
            yuyin(gesture_str)
            # 重置计数器
            count_same_gesture = 0

        # print(gesture_str)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    # 初始化Whisper模型
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    model1 = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    detect()