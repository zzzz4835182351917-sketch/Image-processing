# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:30:53 2018
 
形態學操作就是改變物體的形狀，如腐蝕使物體"變瘦"，膨脹使物體"變胖"
先腐蝕後膨脹會分離物體，所以叫開運算，常用來去除小區域物體
先膨脹後腐蝕會消除物體內的小洞，所以叫閉運算
img_path = asksaveasfilename(initialdir = file_path, 
              filetypes=[("jpg格式","jpg"), ("png格式","png"), ("bmp格式","bmp")],
              parent = self.root,
              title = '儲存影像')
"""

import tkinter.messagebox as messagebox
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from tkinter.filedialog import askopenfilename, asksaveasfilename  
import cv2
import numpy as np
import glob
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfd
import datetime
import vlc, time
from datetime import timedelta
import skimage
import scipy.special as special
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import skimage.morphology as morphology
# === YOLO 需要的套件 ===
import torch
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont
# === 手勢辨識需要的套件與模型 ===
import mediapipe as mp
import torch.nn as nn
from collections import deque

class GestureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(63, 128)
        encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 14) # 對應 14 個類別 (含彩蛋)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
# ==========================================
file_path = os.path.dirname(__file__)
test_file_path = 'D:/anaconda/images/lena_std_512.jpg'
WIN_WIDTH = 1224
WIN_HEIGHT = 672

class ImageViewer():
    CANVAS_WIDTH = 600
    CANVAS_HEIGHT = 450

    def __init__(self, master):
        self.parent = master
        self.parent.title("ImageViewer")
        self.parent.resizable(width=tk.TRUE, height=tk.TRUE)
        self.parent.bind("<Left>", self.prev)
        self.parent.bind("<Right>", self.next)

        self.init_menubar()

        self.init_imageviewer()

    def init_menubar(self):
        menubar = tk.Menu(self.parent)
        self.parent.configure(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Dir", underline=0, menu=file_menu)
        file_menu.add_command(label="Open", underline=0, command=self.open_dir)


    def init_imageviewer(self):
        self.images = list()
        self.image_maxreso = dict()
        self.image_tk = None
        self.image_dir = None
        self.image_idx = 0
        self.image_cnt = 0

        # main frame
        self.mframe = tk.Frame(self.parent)
        self.mframe.pack(fill=tk.BOTH, expand=1)

        # image frame
        self.iframe = tk.Frame(self.mframe)
        self.iframe.pack()
        self.image_canvas = tk.Canvas(self.iframe, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,cursor='plus')
        self.image_canvas.pack(pady=0, anchor=tk.N)

        # control frame
        self.cframe = tk.Frame(self.mframe)
        self.cframe.pack(side=tk.TOP, padx=5, pady=10)
        self.prev_button = ttk.Button(self.cframe, text='<<', width=10, command=self.prev)
        self.prev_button.pack(side = tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.cframe, text='>>', width=10, command=self.next)
        self.next_button.pack(side = tk.LEFT, padx=5)

        # status frame
        self.sframe = tk.Frame(self.mframe)
        self.sframe.pack(side=tk.TOP, padx=5, pady=10)
        self.status_label = ttk.Label(self.sframe,
                                     text='{:3d}/{:3d}'.format(0,0),
                                     width=10,
                                     anchor=tk.CENTER)
        self.status_label.pack(side = tk.LEFT, padx=5)
        self.imagenum_entry = ttk.Entry(self.sframe, width=5)
        self.imagenum_entry.insert(tk.END, "")
        self.imagenum_entry.pack(side=tk.LEFT, padx=5)
        self.skip_button = ttk.Button(self.sframe, text="SKIP", width=5, command=self.skip)
        self.skip_button.pack(side = tk.LEFT, padx=5)

    def delete(self):
        # delete str in entrybox.
        self.dir_entry.delete(0, tk.END)

    def prev(self, event=None):
        if 0 < self.image_idx:
            self.image_idx -= 1
            self.show_image(self.image_idx)

    def next(self, event=None):
        if self.image_idx < (self.image_cnt-1):
            self.image_idx += 1
            self.show_image(self.image_idx)

    def skip(self, event=None):
        img_num = self.imagenum_entry.get()
        if img_num.isdecimal():
            img_idx = int(img_num) - 1
            if 0 <= img_idx and img_idx <= (self.image_cnt-1):
                self.image_idx = img_idx
                self.show_image(self.image_idx)

    def show_image(self, idx):
        DISP_X = 0
        DISP_Y = 0

        if idx < 0 or idx >= self.image_cnt:
            raise ValueError("imageidx invalid")

        # update cnavas size
        new_canvas_widht = max(self.image_maxreso["width"], self.CANVAS_WIDTH)
        new_canvas_height = max(self.image_maxreso["height"], self.CANVAS_HEIGHT)
        self.image_canvas.config(width=new_canvas_widht, height=new_canvas_height)

        # update cnavas image
        self.image_tk = ImageTk.PhotoImage(self.images[idx])
        self.image_canvas.create_image(DISP_X, DISP_Y, image=self.image_tk, anchor=tk.NW)

        # update status label
        self.status_label.configure(text='{:3d}/{:3d}'.format(self.image_idx+1,self.image_cnt))

    def open_dir(self):
        self.image_dir = tkfd.askdirectory()

        if self.image_dir == "":
            return

        if not os.path.exists(self.image_dir):
            tkmsg.showwarning("Warning", message="{} doesn't exist.".format(self.image_dir))
            return

        if not os.path.isdir(self.image_dir):
            tkmsg.showwarning("Warning", message="{} isn't dir.".format(self.image_dir))
            return

        image_paths = list()
        accepted_ext = (".jpeg", '.jpg', '.png')
        for ext in accepted_ext:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, "*"+ext)))

        self.images = list()
        self.image_maxreso["height"] = 0
        self.image_maxreso["width"] = 0
        for image_path in image_paths:
            self.images.append(Image.open(image_path))
            height = self.images[-1].height
            width = self.images[-1].width
            if self.image_maxreso["height"] < height and self.image_maxreso["width"] < width:
                self.image_maxreso["height"] = height
                self.image_maxreso["width"] = width

        image_cnt = len(self.images)
        if image_cnt == 0:
            tkmsg.showwarning("Warning", message="image doesn't exist.")
            return

        self.image_idx = 0
        self.image_cnt = image_cnt

        self.show_image(self.image_idx)
        
class VideoProgressBar(tk.Scale):
    def __init__(self, master, command, **kwargs):
        kwargs["showvalue"] = False
        super().__init__(master, from_=0, to=100, orient=tk.HORIZONTAL, length=800, command=command, **kwargs)
        self.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.cget("state") == tk.NORMAL:
            value = (event.x / self.winfo_width()) * 100
            self.set(value)

class MediaPlayerApp(tk.Frame): # 改成繼承 Frame
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.configure(bg="#f0f0f0")
        self.pack(fill=tk.BOTH, expand=True) # 填滿老師給的 current_root
        self.initialize_player()

    def initialize_player(self):
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.current_file = None
        self.playing_video = False
        self.video_paused = False
        self.create_widgets()

    def create_widgets(self):
        self.media_canvas = tk.Canvas(self, bg="black", width=1280, height=500)
        self.media_canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        self.select_file_button = tk.Button(self, text="選擇檔案(Select File)", font=("Arial", 12, "bold"), command=self.select_file)
        self.select_file_button.pack(pady=5)
        self.time_label = tk.Label(self, text="00:00:00 / 00:00:00", font=("Arial", 12, "bold"), fg="#555555", bg="#f0f0f0")
        self.time_label.pack(pady=5)
        
        self.control_buttons_frame = tk.Frame(self, bg="#f0f0f0")
        self.control_buttons_frame.pack(pady=5)
        self.play_button = tk.Button(self.control_buttons_frame, text="播放(Play)", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button = tk.Button(self.control_buttons_frame, text="暫停(Pause)", font=("Arial", 12, "bold"), bg="#FF9800", fg="white", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.stop_button = tk.Button(self.control_buttons_frame, text="停止(Stop)", font=("Arial", 12, "bold"), bg="#F44336", fg="white", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, pady=5)
        self.fast_forward_button = tk.Button(self.control_buttons_frame, text="快轉(Fast Forward)", font=("Arial", 12, "bold"), bg="#2196F3", fg="white", command=self.fast_forward)
        self.fast_forward_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.rewind_button = tk.Button(self.control_buttons_frame, text="<<回帶(Rewind)<<", font=("Arial", 12, "bold"), bg="#2196F3", fg="white", command=self.rewind)
        self.rewind_button.pack(side=tk.LEFT, pady=5)
        
        self.volumeScale = tk.Scale(self.control_buttons_frame, label='聲音(Volume)', font=("Arial", 10, "bold"), from_=0, to=100, orient=tk.HORIZONTAL, command=self.volume_setting, sliderlength=30, bd=0, showvalue=0, length=200)
        self.volumeScale.pack(side=tk.LEFT, padx=10, pady=5)
        self.volumeScale.set(50)
        
        self.progress_bar = VideoProgressBar(self, self.set_video_position, label='播放進度(Progress Bar)', font=("Arial", 10, "bold"), bg="#e0e0e0", highlightthickness=0)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def volume_setting(self, setValue):
        volume = self.volumeScale.get()
        try:
            self.media_player.audio_set_volume(volume)
        except AttributeError:
            pass

    def select_file(self):
        file_path = tkfd.askopenfilename(initialdir='D:/anaconda/images', filetypes=[("Media Files", "*.mp4 *.avi")])
        if file_path:
            self.current_file = file_path
            self.time_label.config(text="00:00:00 / " + self.get_duration_str())
            self.play_video()

    def get_duration_str(self):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            return str(timedelta(milliseconds=total_duration))[:-3]
        return "00:00:00"

    def play_video(self):
        if not self.playing_video:
            media = self.instance.media_new(self.current_file)
            self.media_player.set_media(media)
            self.media_player.set_hwnd(self.media_canvas.winfo_id())
            self.media_player.play()
            time.sleep(1)
            self.playing_video = True

    def fast_forward(self):
        if self.playing_video:
            current_time = self.media_player.get_time() + 10000
            self.media_player.set_time(current_time)

    def rewind(self):
        if self.playing_video:
            current_time = self.media_player.get_time() - 10000
            self.media_player.set_time(current_time)

    def pause_video(self):
        if self.playing_video:
            if self.video_paused:
                self.media_player.play()
                self.video_paused = False
                self.pause_button.config(text="暫停(Pause)")
            else:
                self.media_player.pause()
                self.video_paused = True
                self.pause_button.config(text="恢復(Resume)")

    def stop(self):
        if self.playing_video:
            self.media_player.stop()
            self.playing_video = False
        self.time_label.config(text="00:00:00 / " + self.get_duration_str())

    def set_video_position(self, value):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            position = int((float(value) / 100) * total_duration)
            self.media_player.set_time(position)

    def update_video_progress(self):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            current_time = self.media_player.get_time()
            progress_percentage = (current_time / total_duration) * 100
            self.progress_bar.set(progress_percentage)
            current_time_str = str(timedelta(milliseconds=current_time))[:-3]
            total_duration_str = str(timedelta(milliseconds=total_duration))[:-3]
            self.time_label.config(text=f"{current_time_str} / {total_duration_str}")
        self.after(10, self.update_video_progress)

class Image_sys():
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('1224x672+80+80')
        self.root.title('數位影像處理-2026')#設置視窗標題
       # self.root.iconbitmap('./icon.ico')# 設置視窗圖示
        #scnWidth, scnHeight = self.root.maxsize()
        # 螢幕中心居中
        #center = '%dx%d %d %d' % (WIN_WIDTH, WIN_HEIGHT, (scnWidth - WIN_WIDTH) / 2, (scnHeight - WIN_HEIGHT) / 2)
        #print(center)
        # 設置視窗的大小寬x高 偏移量
        #self.root.geometry(center)
        # 調用方法會禁止根表單改變大小
        self.root.resizable(False, False)
                 
        menubar = tk.Menu(self.root)# 創建功能表列 (Menu)
        self.root.config(menu = menubar)
         
        # 創建文件下拉式功能表
        # 檔菜單下 tearoff=0 表示有沒有分隔符號，默認為有分隔符號
        file_menu = tk.Menu(menubar, tearoff = 0)
        #為頂級功能表實例添加功能表，並級聯相應的子功能表實例
        menubar.add_cascade(label = "影像", menu = file_menu)
        file_menu.add_command(label = "打開測試影像", command = self.open_test_file)
        file_menu.add_command(label = "打開自訂影像", command = self.open_file)
        file_menu.add_command(label = "儲存目標影像", command = self.save_file)
        file_menu.add_command(label = "開啟圖片資料夾", command = self.open_viewer_dir)
        file_menu.add_command(label = "打開影像檢視器", command = self.open_image_viewer)
        file_menu.add_command(label = "復原", command = self.recover)
        file_menu.add_command(label = "清除", command = self.clear)
        file_menu.add_command(label = "退出", command = self.exit_sys)
        
        # 創建視訊下拉式功能表
        video_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "影片/視訊", menu = video_menu)
        video_menu.add_command(label = "開啟 cam", command = self.open_webcam)
        video_menu.add_command(label = "cam照相", command = self.snapshot_webcam)
        video_menu.add_command(label = "cam攝影", command = self.start_recording)
        video_menu.add_command(label = "停止攝影", command = self.stop_recording)
        video_menu.add_command(label = "視訊檔播放", command = self.video_player)
        video_menu.add_command(label = "關閉 cam", command = self.close_webcam)
        video_menu.add_command(label = "VLC 影片播放器 (MP03)", command = self.Media_Player03)
        video_menu.add_command(label = "VLC 影片播放器 (MP04)", command = self.Media_Player04)
        video_menu.add_command(label = "關閉原視窗播放 (恢復影像介面)", command = self.close_Media_Player04)
        
        # 強化
        enhance_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "強化", menu = enhance_menu)
        enhance_menu.add_command(label = "影像負片", command = self.image_negative)
        enhance_menu.add_command(label = "灰階直方圖", command = self.gray_histogram)
        enhance_menu.add_command(label = "色平面直方圖", command = self.color_histogram)
        enhance_menu.add_command(label = "曝光&對比強化:直方圖等化", command = self.histogram_equalization)
        enhance_menu.add_command(label = "曝光加強:對數(gamma02)轉換", command = self.log_transformation)
        # 利用 lambda 共用 Gamma 轉換
        enhance_menu.add_command(label = "曝光加強:gamma01轉換", command = lambda: self.gamma_transformation(0.1))
        enhance_menu.add_command(label = "曝光加強:gamma05轉換", command = lambda: self.gamma_transformation(0.5))
        enhance_menu.add_command(label = "曝光減弱:gamma12轉換", command = lambda: self.gamma_transformation(1.2))
        enhance_menu.add_command(label = "曝光減弱:gamma22轉換", command = lambda: self.gamma_transformation(2.2))
        # 利用 lambda 共用 Beta 轉換
        enhance_menu.add_command(label = "對比柔化:beta0505轉換", command = lambda: self.beta_transformation(0.5, 0.5))
        enhance_menu.add_command(label = "對比銳化:beta2020轉換", command = lambda: self.beta_transformation(2.0, 2.0))
        
        # 創建翻轉下拉式功能表
        turn_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "翻轉", menu = turn_menu)
        turn_menu.add_command(label = "水平", command = self.flip_horizontal)
        turn_menu.add_command(label = "垂直", command = self.flip_vertical)
        turn_menu.add_command(label = "水平&垂直", command = self.flip_hor_ver)
 
        # 形態學
        morph_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "形態學", menu = morph_menu)
        morph_menu.add_command(label = "腐蝕", command = self.mor_expand)
        morph_menu.add_command(label = "膨脹", command = self.mor_corrosion)
        morph_menu.add_command(label = "開運算", command = self.mor_open_operation)
        morph_menu.add_command(label = "閉運算", command = self.mor_close_operation)
        morph_menu.add_command(label = "型態梯度", command = self.mor_gradient)
        morph_menu.add_command(label = "頂帽", command = self.mor_top_hat)
        morph_menu.add_command(label = "黑帽", command = self.mor_black_hat)
        morph_menu.add_command(label = "取骨架", command = self.mor_skelton)
        morph_menu.add_command(label = "細線化", command = self.mor_thinning)
        
        # 雜訊 
        noise_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "雜訊", menu = noise_menu)
        noise_menu.add_command(label = "高斯(gaussian)雜訊", command = self.noise_gaussian)
        noise_menu.add_command(label = "區域變異數(localvar)雜訊", command = self.noise_localvar)
        noise_menu.add_command(label = "波義森(poisson)雜訊", command = self.noise_poisson)
        noise_menu.add_command(label = "鹽(salt)雜訊", command = self.noise_salt)
        noise_menu.add_command(label = "胡椒(pepper)雜訊", command = self.noise_pepper)
        noise_menu.add_command(label = "胡椒鹽(salt & pepper)雜訊", command = self.noise_sp)
        noise_menu.add_command(label = "史培克(speckle)雜訊", command = self.noise_speckle)
        
        # 濾波-平滑模糊
        blur_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "濾波-平滑模糊", menu = blur_menu)
        
        # 利用 lambda 匿名函式，直接在按鈕上傳遞參數給共用核心！
        blur_menu.add_command(label = "均值3x3", command = lambda: self.mean_filter(self.path, (3, 3)))
        blur_menu.add_command(label = "均值5x5", command = lambda: self.mean_filter(self.path, (5, 5)))
        blur_menu.add_command(label = "均值7x7", command = lambda: self.mean_filter(self.path, (7, 7)))
        blur_menu.add_command(label = "均值9x9", command = lambda: self.mean_filter(self.path, (9, 9)))
        blur_menu.add_command(label = "均值11x11", command = lambda: self.mean_filter(self.path, (11, 11)))
        blur_menu.add_command(label = "均值15x15", command = lambda: self.mean_filter(self.path, (15, 15)))
        blur_menu.add_command(label = "均值21x21", command = lambda: self.mean_filter(self.path, (21, 21)))

        blur_menu.add_command(label = "方框3x3nf", command = lambda: self.box_filter(self.path, (3, 3), False))
        blur_menu.add_command(label = "方框3x3nt", command = lambda: self.box_filter(self.path, (3, 3), True))

        blur_menu.add_command(label = "高斯3x3", command = lambda: self.gauss_filter(self.path, (3, 3)))
        blur_menu.add_command(label = "高斯5x5", command = lambda: self.gauss_filter(self.path, (5, 5)))
        blur_menu.add_command(label = "高斯7x7", command = lambda: self.gauss_filter(self.path, (7, 7)))
        blur_menu.add_command(label = "高斯9x9", command = lambda: self.gauss_filter(self.path, (9, 9)))
        blur_menu.add_command(label = "高斯11x11", command = lambda: self.gauss_filter(self.path, (11, 11)))
        blur_menu.add_command(label = "高斯15x15", command = lambda: self.gauss_filter(self.path, (15, 15)))
        blur_menu.add_command(label = "高斯21x21", command = lambda: self.gauss_filter(self.path, (21, 21)))

        blur_menu.add_command(label = "中值3", command = lambda: self.mid_filter(self.path, 3))
        blur_menu.add_command(label = "中值5", command = lambda: self.mid_filter(self.path, 5))
        blur_menu.add_command(label = "中值7", command = lambda: self.mid_filter(self.path, 7))
        blur_menu.add_command(label = "中值9", command = lambda: self.mid_filter(self.path, 9))
        blur_menu.add_command(label = "中值11", command = lambda: self.mid_filter(self.path, 11))
        blur_menu.add_command(label = "中值15", command = lambda: self.mid_filter(self.path, 15))
        blur_menu.add_command(label = "中值21", command = lambda: self.mid_filter(self.path, 21))

        blur_menu.add_command(label = "雙邊37575", command = lambda: self.bilateral_filter(self.path, 3, 75, 75))
        blur_menu.add_command(label = "雙邊57575", command = lambda: self.bilateral_filter(self.path, 5, 75, 75))
        blur_menu.add_command(label = "雙邊77575", command = lambda: self.bilateral_filter(self.path, 7, 75, 75))
        blur_menu.add_command(label = "雙邊97575", command = lambda: self.bilateral_filter(self.path, 9, 75, 75))
        blur_menu.add_command(label = "雙邊117575", command = lambda: self.bilateral_filter(self.path, 11, 75, 75))
        blur_menu.add_command(label = "雙邊157575", command = lambda: self.bilateral_filter(self.path, 15, 75, 75))
        blur_menu.add_command(label = "雙邊217575", command = lambda: self.bilateral_filter(self.path, 21, 75, 75))
        
        # 濾波-梯度邊緣銳化
        sharp_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "濾波-梯度邊緣銳化", menu = sharp_menu)

        sharp_menu.add_command(label = "SobelGx", command = lambda: self.edge_sharpen_filter("SobelGx"))
        sharp_menu.add_command(label = "SobelGy", command = lambda: self.edge_sharpen_filter("SobelGy"))
        sharp_menu.add_command(label = "PrewittGx", command = lambda: self.edge_sharpen_filter("PrewittGx"))
        sharp_menu.add_command(label = "PrewittGy", command = lambda: self.edge_sharpen_filter("PrewittGy"))
        sharp_menu.add_command(label = "FreiChenGx", command = lambda: self.edge_sharpen_filter("FreiChenGx"))
        sharp_menu.add_command(label = "FreiChenGy", command = lambda: self.edge_sharpen_filter("FreiChenGy"))
        sharp_menu.add_command(label = "ScharrGx", command = lambda: self.edge_sharpen_filter("ScharrGx"))
        sharp_menu.add_command(label = "ScharrGy", command = lambda: self.edge_sharpen_filter("ScharrGy"))
        sharp_menu.add_command(label = "Laplacian3x3-01", command = lambda: self.edge_sharpen_filter("Laplacian3x3-01"))
        sharp_menu.add_command(label = "Laplacian3x3-02", command = lambda: self.edge_sharpen_filter("Laplacian3x3-02"))
        sharp_menu.add_command(label = "Laplacian3x3-03", command = lambda: self.edge_sharpen_filter("Laplacian3x3-03"))
        sharp_menu.add_command(label = "Laplacian5x5", command = lambda: self.edge_sharpen_filter("Laplacian5x5"))
        sharp_menu.add_command(label = "Laplacian7x7", command = lambda: self.edge_sharpen_filter("Laplacian7x7"))
        sharp_menu.add_command(label = "Laplacian9x9", command = lambda: self.edge_sharpen_filter("Laplacian9x9"))
        sharp_menu.add_command(label = "Canny", command = lambda: self.edge_sharpen_filter("Canny"))
        sharp_menu.add_command(label = "EdgeSharp3x3-01", command = lambda: self.edge_sharpen_filter("EdgeSharp3x3-01"))
        sharp_menu.add_command(label = "EdgeSharp3x3-02", command = lambda: self.edge_sharpen_filter("EdgeSharp3x3-02"))
        sharp_menu.add_command(label = "EdgeSharp3x3-03", command = lambda: self.edge_sharpen_filter("EdgeSharp3x3-03"))
        sharp_menu.add_command(label = "EdgeSharp5x5", command = lambda: self.edge_sharpen_filter("EdgeSharp5x5"))
        sharp_menu.add_command(label = "EdgeSharp7x7", command = lambda: self.edge_sharpen_filter("EdgeSharp7x7"))
        
        # 色彩檢測
        color_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "色彩檢測", menu = color_menu)
        color_menu.add_command(label = "RYGB色彩檢測", command = self.color_detection)
        color_menu.add_command(label = "即時紅藍綠黃色球(瓶蓋)檢測", command = self.redball_tracking)
        
        # 縮放
        scale_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "縮放", menu = scale_menu)
        scale_menu.add_command(label = "放大PyrUp", command = self.scale_pyrup)
        scale_menu.add_command(label = "縮小PyrDown", command = self.scale_pyrdown)
        scale_menu.add_command(label = "放大Resize", command = self.scale_zoom_in)
        scale_menu.add_command(label = "縮小Resize", command = self.scale_zoom_out)
 
        # 旋轉
        rotate_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "旋轉", menu = rotate_menu)
        rotate_menu.add_command(label = "平移", command = self.rotate_offset)
        rotate_menu.add_command(label = "仿射", command = self.rotate_affine)
        rotate_menu.add_command(label = "透射", command = self.rotate_transmission)
        rotate_menu.add_command(label = "順時針-無縮放", command = self.rotate_clockwise)
        rotate_menu.add_command(label = "順時針-縮放", command = self.rotate_clockwise_zoom)
        rotate_menu.add_command(label = "逆時針-縮放", command = self.rotate_anti_zoom)
        rotate_menu.add_command(label = "零旋轉-縮放", command = self.rotate_zero_zoom)
        
        # CAM即時影像邊緣處理 
        cam_real_time_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "CAM即時影像邊緣處理", menu = cam_real_time_menu)
        cam_real_time_menu.add_command(label = "CAM_SobelX", command = lambda: self.start_realtime_video('SobelX', is_cam=True))
        cam_real_time_menu.add_command(label = "CAM_SobelY", command = lambda: self.start_realtime_video('SobelY', is_cam=True))
        cam_real_time_menu.add_command(label = "CAM_Laplacian", command = lambda: self.start_realtime_video('Laplacian', is_cam=True))
        cam_real_time_menu.add_command(label = "CAM_Canny", command = lambda: self.start_realtime_video('Canny', is_cam=True))
        cam_real_time_menu.add_command(label = "CAM_MG", command = lambda: self.start_realtime_video('MG', is_cam=True))
        
        # Video即時影像邊緣處理 
        video_real_time_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "Video即時影像邊緣處理", menu = video_real_time_menu)
        video_real_time_menu.add_command(label = "Video_SobelX", command = lambda: self.start_realtime_video('SobelX', is_cam=False))
        video_real_time_menu.add_command(label = "Video_SobelY", command = lambda: self.start_realtime_video('SobelY', is_cam=False))
        video_real_time_menu.add_command(label = "Video_Laplacian", command = lambda: self.start_realtime_video('Laplacian', is_cam=False))
        video_real_time_menu.add_command(label = "Video_Canny", command = lambda: self.start_realtime_video('Canny', is_cam=False))
        video_real_time_menu.add_command(label = "Video_MG", command = lambda: self.start_realtime_video('MG', is_cam=False))
        
        # === Lab 12+: 影像-Panorama全景影像 ===
        image_Panorama_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="影像-Panorama全景影像", menu=image_Panorama_menu)
        image_Panorama_menu.add_command(label="ORB多張影像無痕拼接", command=self.ORB_Panorama)
        image_Panorama_menu.add_command(label="SIFT多張影像無痕拼接", command=self.SIFT_Panorama)
        image_Panorama_menu.add_command(label="BRISK多張影像無痕拼接", command=self.BRISK_Panorama)
        
        # === Lab 13: 人臉辨識系統 ===
        face_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="人臉辨識系統", menu=face_menu)
        face_menu.add_command(label="Haar人臉偵測", command=self.haar_face_detection)
        face_menu.add_command(label="擷取人臉樣本", command=self.get_face_samples)
        face_menu.add_command(label="訓練特徵模型", command=self.train_face_model)
        face_menu.add_command(label="即時人臉辨識(Dlib)", command=self.dlib_face_recognition)
        # ==========================================
        # === Lab 14: Dlib 人臉辨識系統選單 ===
        dlib_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Dlib 人臉辨識系統", menu=dlib_menu)
        dlib_menu.add_command(label="1. Cam人臉註冊 (建檔)", command=self.dlib_register)
        dlib_menu.add_command(label="2. 人臉128D特徵提取 (訓練)", command=self.dlib_extract)
        dlib_menu.add_command(label="3. Cam即時人臉辨識 (Demo)", command=self.dlib_recognize)
        # 💡 新增 DeepFace 多重特徵辨識選項
        dlib_menu.add_command(label="4. DeepFace (年齡/性別/情緒)", command=self.start_deepface_analysis)
        
        # ==========================================
        # === 終極版：ArcFace (InsightFace) 人臉辨識系統 ===
        arcface_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ArcFace 人臉辨識", menu=arcface_menu)
        # 💡 註冊拍照我們可以直接共用 Dlib 那個寫好的順暢版功能！
        arcface_menu.add_command(label="1. Cam人臉註冊 (與Dlib共用)", command=self.dlib_register)
        arcface_menu.add_command(label="2. ArcFace 512D特徵提取", command=self.arcface_extract)
        arcface_menu.add_command(label="3. ArcFace 即時辨識 (含打卡)", command=self.arcface_recognize)
        
        # ==========================================
        # === YOLO v8 物件辨識系統選單 ===
        yolo_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="YOLO 物件辨識", menu=yolo_menu)
        yolo_menu.add_command(label="開啟 YOLO 即時攝影機", command=self.start_yolo_webcam)
        # ==========================================
        
       # === AI 動態手勢辨識系統選單 ===
        gesture_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI 手勢辨識", menu=gesture_menu)
        gesture_menu.add_command(label="啟動 MediaPipe 雙螢幕手勢偵測", command=self.start_gesture_recognition)
        
        # 本機影片版
        gesture_menu.add_command(label="專題: 隔空手勢多媒體播放器 (本機影片)", command=self.start_free_touch_player)
        
        # 💡 加入這行：YouTube 影片版！
        gesture_menu.add_command(label="專題: 隔空手勢多媒體播放器 (YouTube導入)", command=self.start_free_touch_youtube)
        
        # 幫助
        help_menu = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "幫助", menu = help_menu)
        help_menu.add_command(label = "版權", command = self.help_copyright)
        help_menu.add_command(label = "關於", command = self.help_about)                
         
  
        # 創建一個容器,其父容器為self.root
        self.frame_scr = ttk.LabelFrame(self.root, text="Source image") 
        # padx  pady   該容器週邊需要留出的空餘空間
        self.frame_scr.place(x = 80, y = 50, width = 512, height  = 512) 
       
        # 創建一個容器,其父容器為self.root
        self.frame_des = ttk.LabelFrame(self.root, text="Destination image") 
        # padx  pady   該容器週邊需要留出的空餘空間
        self.frame_des.place(x = 632, y = 50, width = 512, height  = 512)    
 
 
        # 創建兩個label
        self.label_scr = ttk.Label(self.root, text = '來源影像', font = 25, foreground = 'blue', anchor = 'center')
        self.label_scr.place(x = 80, y = 562, width = 100, height  = 50)
        
        self.label_des = ttk.Label(self.root, text = '目標影像', font = 25, foreground = 'blue', anchor = 'center')
        self.label_des.place(x = 632, y = 562, width = 100, height  = 50)
        
        # === 請在這裡加入以下程式碼 ===
        # === 動態控制按鈕 (建立後先不顯示) ===
        self.btn_prev = ttk.Button(self.root, text="<< 上一張", command=self.prev_img)
        self.btn_next = ttk.Button(self.root, text="下一張 >>", command=self.next_img)
        # 將原本的 "📸 擷取畫面 (Snapshot)" 改成 "按讚照相！"
        self.btn_snapshot = ttk.Button(self.root, text="照相", command=self.snapshot_webcam)
        
        # 綁定鍵盤左右方向鍵
        self.root.bind("<Left>", self.prev_img)
        self.root.bind("<Right>", self.next_img)

        # 存放檢視器狀態的變數
        self.viewer_dir_paths = []
        self.viewer_idx = 0
        # ==============================
         
        self.label_scr_image = None
        self.label_des_image = None          
        self.path = ''
        self.des_image_pil = None
        # --- 新增這三行用來控制視訊 ---
        self.cap = None             # 存放攝影機物件
        self.video_loop_id = None   # 控制畫面更新的迴圈 ID
        self.current_frame = None   # 存放當前攝影機的畫面
        self.is_recording = False   # 判斷目前是否正在錄影
        self.video_writer = None    # 用來寫入影片的物件
        # --------------------------- 
        # === YOLO 專屬雙層控制面板 ===
        self.frame_yolo_controls = tk.Frame(self.root)
        
        # 第一排：攝影機、影片、RTSP、YouTube、停止、信心度拉桿
        row1 = tk.Frame(self.frame_yolo_controls)
        row1.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.btn_yolo_cam = tk.Button(row1, text="攝影機", command=self.yolo_start_cam, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.btn_yolo_cam.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_video = tk.Button(row1, text="影片", command=self.yolo_open_video, bg="#2196F3", fg="white", font=("Arial", 9, "bold"))
        self.btn_yolo_video.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_rtsp = tk.Button(row1, text="RTSP", command=self.yolo_open_rtsp, bg="#9C27B0", fg="white", font=("Arial", 9, "bold"))
        self.btn_yolo_rtsp.pack(side=tk.LEFT, padx=2)

        # 🌟 1. 新增的 YouTube 按鈕
        self.btn_yolo_youtube = tk.Button(row1, text="YouTube", command=self.yolo_open_youtube, bg="#E53935", fg="white", font=("Arial", 9, "bold"))
        self.btn_yolo_youtube.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_stop = tk.Button(row1, text="停止", command=self.yolo_stop, bg="#f44336", fg="white", font=("Arial", 9, "bold"))
        self.btn_yolo_stop.pack(side=tk.LEFT, padx=2)
        
        tk.Label(row1, text="信心度:").pack(side=tk.LEFT, padx=5)
        self.yolo_conf_slider = tk.Scale(row1, from_=0.1, to=0.9, resolution=0.05, orient="horizontal", length=120, showvalue=1)
        self.yolo_conf_slider.set(0.4)
        self.yolo_conf_slider.pack(side=tk.LEFT, padx=2)

        # 第二排：截圖、錄影、切換英文、清除ROI、匯出CSV、FPS與人數顯示
        row2 = tk.Frame(self.frame_yolo_controls)
        row2.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.btn_yolo_snapshot = tk.Button(row2, text="截圖", command=self.yolo_snapshot, bg="#FF9800", fg="white", font=("Arial", 9, "bold"), width=5)
        self.btn_yolo_snapshot.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_record = tk.Button(row2, text="開始錄影", command=self.yolo_toggle_record, bg="#607D8B", fg="white", font=("Arial", 9, "bold"), width=8)
        self.btn_yolo_record.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_lang = tk.Button(row2, text="切換英文", command=self.yolo_toggle_lang, bg="#009688", fg="white", font=("Arial", 9, "bold"), width=8)
        self.btn_yolo_lang.pack(side=tk.LEFT, padx=2)

        # 🌟 2. 新增的 清除 ROI 按鈕
        self.btn_yolo_clear_roi = tk.Button(row2, text="清除 ROI", command=self.yolo_clear_roi, bg="#795548", fg="white", font=("Arial", 9, "bold"), width=8)
        self.btn_yolo_clear_roi.pack(side=tk.LEFT, padx=2)

        # 🌟 3. 新增的 匯出 CSV 按鈕
        self.btn_yolo_export_csv = tk.Button(row2, text="匯出 CSV", command=self.yolo_export_csv, bg="#FFC107", fg="black", font=("Arial", 9, "bold"), width=8)
        self.btn_yolo_export_csv.pack(side=tk.LEFT, padx=2)
        
        self.lbl_yolo_fps = tk.Label(row2, text="FPS: 0.0", font=("Arial", 11, "bold"), fg="blue")
        self.lbl_yolo_fps.pack(side=tk.LEFT, padx=10)

        # 🌟 新增的人數顯示
        self.lbl_yolo_person = tk.Label(row2, text="人數: 0", font=("Arial", 11, "bold"), fg="red")
        self.lbl_yolo_person.pack(side=tk.LEFT, padx=5)
        
        # YOLO 狀態變數
        self.yolo_lang_mode = "zh"
        self.yolo_recording = False
        self.yolo_video_writer = None
        self.yolo_frame = None
        self.yolo_prev_time = 0
        self.yolo_current_src = 0
        # ========================================================          
        self.root.mainloop()    
             
    def open_test_file(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        self.path = test_file_path
        image = Image.open(self.path)
        test_image = ImageTk.PhotoImage(image)
        if(self.label_des_image != None):
            self.label_des_image.pack_forget()# 隱藏控制項
            self.label_des_image = None
        if(self.label_scr_image == None):
            self.label_scr_image = tk.Label(self.frame_scr,image = test_image)
        self.label_scr_image.configure(image=test_image)
        self.label_scr_image.pack()
        self.root.mainloop()
   
    def open_file(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        # 打開文件對話方塊
        open_img_path =askopenfilename(initialdir = file_path, 
               filetypes=[("jpg格式","jpg"), ("png格式","png"), ("bmp格式","bmp")],
               parent = self.root,
               title = '打開自訂影像')
        if (open_img_path == ''):
            return
        else:                
            if(self.label_des_image != None):
                self.label_des_image.pack_forget()# 隱藏控制項
                self.label_des_image = None
            self.path =  open_img_path
            image = Image.open(self.path)
            tk_image = ImageTk.PhotoImage(image)
            if(self.label_scr_image == None):
                self.label_scr_image = tk.Label(self.frame_scr,image = tk_image)
            self.label_scr_image.configure(image = tk_image)
            self.label_scr_image.pack() # 顯示控制項
            self.root.mainloop()
            
    def save_file(self):
        # 先檢查有沒有圖片可以存
        if self.des_image_pil is None:
            messagebox.showwarning(title='警告', message='目前沒有目標影像可以儲存！')
            return
            
        # 彈出儲存檔案的對話框
        save_path = asksaveasfilename(
               initialdir = file_path, 
               filetypes=[("jpg格式", "*.jpg"), ("png格式", "*.png"), ("bmp格式", "*.bmp")],
               defaultextension=".jpg",
               parent = self.root,
               title = '儲存目標影像')
               
        # 如果使用者有選擇路徑並按下儲存
        if save_path:
            self.des_image_pil.save(save_path)
            messagebox.showinfo(title='成功', message='影像已成功儲存！')
    
    # === 內建影像檢視器功能 ===
    def open_viewer_dir(self):
        # 彈出視窗選擇資料夾
        img_dir = tkfd.askdirectory(initialdir='D:/anaconda/images')
        if not img_dir:
            return
            
        # 抓取資料夾內所有支援的圖片
        accepted_ext = (".jpeg", ".jpg", ".png", ".bmp")
        self.viewer_dir_paths = []
        for ext in accepted_ext:
            self.viewer_dir_paths.extend(glob.glob(os.path.join(img_dir, "*" + ext)))
            self.viewer_dir_paths.extend(glob.glob(os.path.join(img_dir, "*" + ext.upper())))
            
        if len(self.viewer_dir_paths) == 0:
            tkmsg.showwarning("警告", "資料夾內沒有支援的圖片檔。")
            return
        
        # === 控制按鈕顯示與隱藏 ===
        self.btn_snapshot.place_forget() # 隱藏照相按鈕
        self.btn_prev.place(x=180, y=575, width=80, height=30) # 顯示上一張
        self.btn_next.place(x=270, y=575, width=80, height=30) # 顯示下一張
        # ==========================
            
        # 從第一張開始顯示
        self.viewer_idx = 0
        self.show_inline_image()

    def show_inline_image(self):
        if not self.viewer_dir_paths: return
        
        # 1. 將當前圖片路徑存給 self.path (這樣上方的濾波功能才知道要處理這張圖)
        self.path = self.viewer_dir_paths[self.viewer_idx]
        
        # 2. 切換圖片前，先清空右側的目標影像
        if self.label_des_image != None:
            self.label_des_image.pack_forget()
            self.label_des_image = None
            
        # 3. 讀取並顯示在左側的來源影像區塊
        image = Image.open(self.path)
        tk_image = ImageTk.PhotoImage(image)
        
        if self.label_scr_image == None:
            self.label_scr_image = tk.Label(self.frame_scr, image=tk_image)
            self.label_scr_image.pack()
            
        self.label_scr_image.configure(image=tk_image)
        self.label_scr_image.image = tk_image

    def prev_img(self, event=None):
        if self.viewer_dir_paths and self.viewer_idx > 0:
            self.viewer_idx -= 1
            self.show_inline_image()

    def next_img(self, event=None):
        if self.viewer_dir_paths and self.viewer_idx < len(self.viewer_dir_paths) - 1:
            self.viewer_idx += 1
            self.show_inline_image()
    
    def open_image_viewer(self):
        # 建立一個獨立的彈出式子視窗 (Toplevel)
        viewer_window = tk.Toplevel(self.root)
        viewer_window.geometry("800x600") # 設定檢視器大小
        
        # 呼叫你剛剛貼上來的 ImageViewer 類別，並把它綁定在這個子視窗上
        ImageViewer(viewer_window)
            
    def recover(self):
        if(self.path == ''):
            return       
        image = Image.open(self.path)
        tk_image = ImageTk.PhotoImage(image)
        if(self.label_des_image == None):
            return
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()
    def clear(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if(self.label_scr_image != None):
            self.label_scr_image.pack_forget()# 隱藏控制項
            self.label_scr_image = None
            self.path = ''
        if(self.label_des_image != None):
            self.label_des_image.pack_forget()# 隱藏控制項
            self.label_des_image = None
            self.path = ''
    def exit_sys(self):
        quit_root = messagebox.askokcancel('提示', '確定要退出?')
        if (quit_root == True):
            self.root.destroy()
        return
    
    # 1. 開啟攝影機
    def open_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # 0 代表預設攝影機
            # === 控制按鈕顯示與隱藏 ===
        self.btn_prev.place_forget() # 隱藏上一張
        self.btn_next.place_forget() # 隱藏下一張
        self.btn_snapshot.place(x=180, y=575, width=170, height=30) # 顯示擷取按鈕
        # ==========================
        self.update_webcam()

    # 2. 持續更新畫面的迴圈
    def update_webcam(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 畫面左右鏡像翻轉
                self.current_frame = frame 
                
                # === 錄影魔法在這裡：如果正在錄影，就把這一幀寫入影片檔 ===
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                # ===================================================

                # 轉換顏色並顯示在 Tkinter 上
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(cv2image)
                tk_image = ImageTk.PhotoImage(image=image_pil)

                if self.label_scr_image is None:
                    self.label_scr_image = tk.Label(self.frame_scr, image=tk_image)
                    self.label_scr_image.pack()

                self.label_scr_image.configure(image=tk_image)
                self.label_scr_image.image = tk_image

                # 每 30 毫秒呼叫自己一次，達成影片播放的效果
                self.video_loop_id = self.root.after(30, self.update_webcam)

    # 3. 擷取定格畫面
    def snapshot_webcam(self):
        if self.current_frame is not None:
            # 取得當前時間做為檔名
            ts = datetime.datetime.now() 
            filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S")) 
            
            # 設定存檔路徑 (你可以改回 D 槽)
            output_dir = "D:/anaconda/images/"
            save_path = os.path.join(output_dir, filename) 
            
            # 存檔並印出訊息
            try:
                cv2.imwrite(save_path, self.current_frame)
                print("[INFO] saved {}".format(filename))
                self.path = save_path # 指向新拍的照片讓濾波可以用
            except:
                tkmsg.showerror("錯誤", "Can not save picture frame!")
                return
            
            # 轉換顏色顯示在右側目標影像區
            cv2image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(cv2image)
            self.des_image_pil = image_pil
            tk_image = ImageTk.PhotoImage(image_pil)
            
            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image)
                self.label_des_image.pack()
            
            self.label_des_image.configure(image=tk_image)
            self.label_des_image.image = tk_image
    
    # === 錄影專屬功能 ===
    def start_recording(self):
        if self.cap is None or not self.cap.isOpened():
            tkmsg.showwarning("警告", "請先開啟 cam 再開始錄影！")
            return
        if self.is_recording:
            tkmsg.showinfo("提示", "目前已經在錄影中囉！")
            return

        # 彈出視窗讓使用者選擇影片存檔位置 (預設為 D 槽的 images)
        save_path = tkfd.asksaveasfilename(
            initialdir="D:/anaconda/images/",
            title="儲存錄影檔",
            filetypes=(('AVI 影片', '*.avi'), ('MP4 影片', '*.mp4'), ('所有檔案','*')),
            defaultextension=".avi"
        )
        
        if save_path:
            # 取得當前攝影機的寬度與高度
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 設定 VideoWriter 寫入器 (編碼格式使用 XVID)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (width, height))
            
            self.is_recording = True
            tkmsg.showinfo("開始", "錄影開始！\n請對準鏡頭，錄製完畢後記得點選上方選單的「停止錄影」。")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release() # 釋放寫入器，確保影片成功存檔
                self.video_writer = None
            tkmsg.showinfo("完成", "錄影已成功停止並儲存至指定路徑！")
        else:
            tkmsg.showwarning("提示", "目前沒有在錄影喔！")
            
    def video_player(self):
        # 隱藏按鈕防呆
        self.btn_prev.place_forget() 
        self.btn_next.place_forget() 
        self.btn_snapshot.place_forget()

        open_video_path = tkfd.askopenfilename(
            initialdir = 'D:/anaconda/images',
            filetypes=[('AVI', '*.avi'), ('MP4', '*.mp4'), ('MKV', '*.mkv'), ('All Files','*')],
            parent = self.root,
            title = '開啟視訊檔')
            
        if (open_video_path == ''):
            tkmsg.showwarning("警告", "Can not open video file!")
            return
        else:
            self.path = open_video_path
            self.cap = cv2.VideoCapture(self.path)
            self.update_webcam() # 呼叫我們寫好的迴圈來播放

    # 4. 關閉攝影機
    def close_webcam(self):
        # 如果還在錄影就按了關閉，先幫忙自動停止錄影存檔
        if self.is_recording:
            self.stop_recording()
            
        if self.video_loop_id is not None:
            self.root.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.clear() # 清空畫面
        
    # 完全對應老師講義的寫法
    def Media_Player03(self):
        # 1. 建立 Toplevel 子視窗
        self.MP03_root = tk.Toplevel(self.root)
        self.MP03_root.title("VLC 媒體播放器")
        self.MP03_root.geometry("1280x720")
        
        # 2. 將當前容器指向這個新視窗
        self.current_root = self.MP03_root
        
        # 3. 呼叫 MediaPlayerApp 並把 current_root 傳給它
        # (注意：因為我們是用獨立的 Class，所以前面不需要加 self.)
        self.vlc_app = MediaPlayerApp(self.current_root)
        
        # 4. 啟動進度條更新
        self.vlc_app.update_video_progress()
        
        # --- 加上防呆機制：關閉視窗時自動停止影片 ---
        self.current_root.protocol("WM_DELETE_WINDOW", self.close_mp03)
        
    def close_mp03(self):
        if self.vlc_app.playing_video:
            self.vlc_app.stop()
        self.MP03_root.destroy()
        
    def Media_Player04(self):
        # 1. 把畫布、文字、按鈕全部隱藏 (place_forget)
        self.frame_scr.place_forget()
        self.frame_des.place_forget()
        self.label_scr.place_forget()
        self.label_des.place_forget()
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        
        # 2. 將當前容器指向主視窗 self.root
        self.current_root = self.root
        
        # 3. 呼叫 MediaPlayerApp 並掛載到主視窗上
        self.vlc_app_04 = MediaPlayerApp(self.current_root)
        
        # 4. 啟動進度條更新
        self.vlc_app_04.update_video_progress()

    # --- 恢復影像處理介面的功能 ---
    def close_Media_Player04(self):
        # 1. 停止影片並銷毀播放器框架
        if hasattr(self, 'vlc_app_04') and self.vlc_app_04:
            if self.vlc_app_04.playing_video:
                self.vlc_app_04.stop()
            self.vlc_app_04.destroy()
            self.vlc_app_04 = None
            
        # 2. 把原本的畫布和文字重新放回原來的位置 (place)
        self.frame_scr.place(x = 80, y = 50, width = 512, height  = 512) 
        self.frame_des.place(x = 632, y = 50, width = 512, height  = 512)
        self.label_scr.place(x = 80, y = 562, width = 100, height  = 50)
        self.label_des.place(x = 632, y = 562, width = 100, height  = 50)
        
    def flip_horizontal(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return       
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        # Flipped Horizontally 水準翻轉
        image_hflip = cv2.flip(image, 1)
        image_pil_hflip = Image.fromarray(image_hflip)
        self.des_image_pil = image_pil_hflip
        tk_image = ImageTk.PhotoImage(image_pil_hflip)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()    
    def flip_vertical(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return       
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        # Flipped Horizontally 水準翻轉
        image_hflip = cv2.flip(image, 0)# 垂直翻轉
        image_pil_hflip = Image.fromarray(image_hflip)
        self.des_image_pil = image_pil_hflip
        tk_image = ImageTk.PhotoImage(image_pil_hflip)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop() 
     
    def flip_hor_ver(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return       
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        # Flipped Horizontally 水準翻轉
        image_hflip = cv2.flip(image, -1)# 水準垂直翻轉
        image_pil_hflip = Image.fromarray(image_hflip)
        self.des_image_pil = image_pil_hflip
        tk_image = ImageTk.PhotoImage(image_pil_hflip)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()        
 
    def mor_corrosion(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        #kernel = np.ones((5, 5), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_erosion = cv2.erode(image, kernel)  # 腐蝕
        image_pil_erosion = Image.fromarray(img_erosion)
        self.des_image_pil = image_pil_erosion
        tk_image = ImageTk.PhotoImage(image_pil_erosion)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()                
    # 膨脹    
    def mor_expand(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        #kernel = np.ones((5, 5), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_dilation = cv2.dilate(image, kernel) # 膨脹
        image_pil_dilation = Image.fromarray(img_dilation)
        self.des_image_pil = image_pil_dilation
        tk_image = ImageTk.PhotoImage(image_pil_dilation)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop() 
    # 開運算    
    def mor_open_operation(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        #kernel = np.ones((5, 5), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_open_operation = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 開運算
        image_pil_open = Image.fromarray(img_open_operation)
        self.des_image_pil = image_pil_open
        tk_image = ImageTk.PhotoImage(image_pil_open)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop() 
    # 閉運算    
    def mor_close_operation(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        #kernel = np.ones((5, 5), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_close_operation = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)# 閉運算
        image_pil_close = Image.fromarray(img_close_operation)
        self.des_image_pil = image_pil_close
        tk_image = ImageTk.PhotoImage(image_pil_close)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()
    # 形態學梯度：膨脹圖減去腐蝕圖，dilation - erosion，這樣會得到物體的輪廓：
    def mor_gradient(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        #kernel = np.ones((5, 5), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel) # 形態學梯度
        image_pil_gradient = Image.fromarray(img_gradient)
        self.des_image_pil = image_pil_gradient
        tk_image = ImageTk.PhotoImage(image_pil_gradient)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()
    # 頂帽    
    def mor_top_hat(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        kernel = np.ones((7, 7), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel) # 頂帽
        image_pil_top_hat = Image.fromarray(img_top_hat)
        self.des_image_pil = image_pil_top_hat
        tk_image = ImageTk.PhotoImage(image_pil_top_hat)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()
    # 黑帽    
    def mor_black_hat(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        kernel = np.ones((7, 7), np.uint8)# 指定核大小
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 橢圓結構
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形結構
        img_black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel) # 黑帽
        image_pil_black_hat = Image.fromarray(img_black_hat)
        self.des_image_pil = image_pil_black_hat
        tk_image = ImageTk.PhotoImage(image_pil_black_hat)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()
        
    # =========================================================
    # === Lab 08: 形態學 - 取骨架與細線化 (升級版) ===

    def mor_skelton(self):
        if self.path == '' or self.label_scr_image is None: return
        
        # 1. 讀取為灰階影像
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 0)
        
        # 2. 智慧二值化並「反轉」(變成黑底白字)
        # 使用 OTSU 自動找最佳門檻值，將主體變白色(255)，背景變黑色(0)
        _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. 強制轉換為 0 和 1 的 Boolean 矩陣 (skimage 的嚴格要求)
        binary = (binary_img == 255)
        
        # 4. 執行取骨架演算法
        skeleton = morphology.skeletonize(binary)
        
        # 5. 將 True/False 轉換回 0/255 的圖片格式
        img_skel = (skeleton * 255).astype(np.uint8)
        
        # 6. 轉回 RGB 丟給我們的共用顯示引擎
        img_skel_rgb = cv2.cvtColor(img_skel, cv2.COLOR_GRAY2RGB)
        self.show_filtered_image(img_skel_rgb)

    def mor_thinning(self):
        if self.path == '' or self.label_scr_image is None: return
        
        # 1. 讀取為灰階影像
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 0)
        
        # 2. 智慧二值化並「反轉」(變成黑底白字)
        _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. 強制轉換為 0 和 1 的 Boolean 矩陣
        binary = (binary_img == 255)
        
        # 4. 執行細線化演算法
        thinned = morphology.thin(binary)
        
        # 5. 轉換格式並顯示
        img_thin = (thinned * 255).astype(np.uint8)
        img_thin_rgb = cv2.cvtColor(img_thin, cv2.COLOR_GRAY2RGB)
        self.show_filtered_image(img_thin_rgb)
    
    # === Lab 04: 加雜訊 (共用核心程式段) ===
    def addtive_noise(self, file, noise_mode):
        if (file == ''):
            return
        if (self.label_scr_image == None):
            return
            
        # 讀取並轉換為 RGB (配合你原本的架構)
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        
        # 轉換為浮點數以避免 uint8 溢位 (講義要求)
        img = np.asarray(image)
        
        # 使用 skimage 加上指定的雜訊
        image_noise = skimage.util.random_noise(img, mode=noise_mode)
        image_noise = (255 * image_noise).astype(np.uint8)
        
        # 顯示到右側目標影像區
        self.des_image_pil = Image.fromarray(image_noise)
        tk_image = ImageTk.PhotoImage(self.des_image_pil)
        
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des, image=tk_image)
        
        self.label_des_image.configure(image=tk_image)
        self.label_des_image.image = tk_image # 避免被系統記憶體回收
        self.label_des_image.pack()
        
    # Lab 04: 7 種雜訊按鈕觸發的函式
    def noise_gaussian(self):
        self.addtive_noise(self.path, 'gaussian')
        
    def noise_localvar(self):
        self.addtive_noise(self.path, 'localvar')
        
    def noise_poisson(self):
        self.addtive_noise(self.path, 'poisson')
        
    def noise_salt(self):
        self.addtive_noise(self.path, 'salt')
        
    def noise_pepper(self):
        self.addtive_noise(self.path, 'pepper')
        
    def noise_sp(self):
        self.addtive_noise(self.path, 's&p')
        
    def noise_speckle(self):
        self.addtive_noise(self.path, 'speckle')
    
    # =========================================================
    # === 獨立顯示引擎 (升級版：自動清除殘留的圖表) ===
    def show_filtered_image(self, img_filtered):
        # 如果右邊有畫好的直方圖，先把它們清掉
        if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
            self.frame_des.canvas.get_tk_widget().pack_forget()
            self.frame_des.canvas = None
        if hasattr(self, 'toolbar') and self.toolbar != None:
            self.toolbar.pack_forget()
            self.toolbar = None

        image_pil = Image.fromarray(img_filtered)
        self.des_image_pil = image_pil
        tk_image = ImageTk.PhotoImage(image_pil)
        if self.label_des_image is None:
            self.label_des_image = tk.Label(self.frame_des, image=tk_image)
        self.label_des_image.configure(image=tk_image)
        self.label_des_image.image = tk_image
        self.label_des_image.pack()
        self.root.mainloop()

    # === 平滑模糊的 5 個共用核心 ===
    def mean_filter(self, file, mean_size):
        if file == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        img_mean = cv2.blur(image, mean_size)
        self.show_filtered_image(img_mean)

    def box_filter(self, file, box_size, norm):
        if file == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        img_box = cv2.boxFilter(image, -1, box_size, normalize=norm)
        self.show_filtered_image(img_box)

    def gauss_filter(self, file, gauss_size):
        if file == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        # cv2.GaussianBlur 需要指定 sigmaX, 通常設為 0 由 size 自動推算
        img_gauss = cv2.GaussianBlur(image, gauss_size, 0) 
        self.show_filtered_image(img_gauss)

    def mid_filter(self, file, mid_size):
        if file == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        img_mid = cv2.medianBlur(image, mid_size)
        self.show_filtered_image(img_mid)

    def bilateral_filter(self, file, d, sigmaColor, sigmaSpace):
        if file == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        img_bilateral = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        self.show_filtered_image(img_bilateral)
    # =========================================================
    # =========================================================
    # === Lab 07: 影像強化核心功能 ===
    
    def image_negative(self):
        if self.path == '': return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        img_neg = 255 - image # 負片轉換公式 s = 255 - r
        self.show_filtered_image(img_neg)

    def histogram_equalization(self):
        if self.path == '': return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        # 直方圖等化通常在 YUV 或 HSV 色彩空間的明度通道做，效果最好
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) # 修正回 RGB
        self.show_filtered_image(img_eq)

    def log_transformation(self):
        if self.path == '': return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        c = 255 / np.log(1 + np.max(image)) # 對數轉換公式
        img_log = c * (np.log(image + 1))
        img_log = np.array(img_log, dtype=np.uint8)
        self.show_filtered_image(img_log)

    def gamma_transformation(self, gamma):
        if self.path == '': return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        # 建立 Gamma 查表 (Look-up Table) 以加速運算
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img_gamma = cv2.LUT(image, table)
        self.show_filtered_image(img_gamma)

    def beta_transformation(self, a, b_val):
        if self.path == '': return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        x = np.linspace(0, 1, 256)
        table = np.round(special.betainc(a, b_val, x) * 255, 0).astype("uint8")
        img_beta = cv2.LUT(image, table)
        self.show_filtered_image(img_beta)

    # === 直方圖繪製功能 (整合 Matplotlib) ===
    def plot_histogram_base(self, is_color):
        if self.path == '' or self.label_scr_image == None: return
        
        # 隱藏右側的圖片 Label
        if self.label_des_image != None:
            self.label_des_image.pack_forget()
            self.label_des_image = None
            
        self.fig = Figure(figsize=(5, 4), dpi=100)
        ax = self.fig.add_subplot(111)
        
        if not is_color:
            image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 0) # 讀灰階
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
        else:
            image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1) # 讀彩色
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)

        ax.set_title("Histogram", fontsize='large', color='blue')
        ax.set_xlabel("Intensity", fontsize='medium', color='brown')
        ax.set_ylabel("#Intensities", fontsize='medium', color='brown')

        if not hasattr(self.frame_des, 'canvas') or self.frame_des.canvas == None:
            self.frame_des.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_des)
            self.toolbar = NavigationToolbar2Tk(self.frame_des.canvas, self.frame_des)
            self.toolbar.update()
            
        self.frame_des.canvas.draw()
        self.frame_des.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.root.mainloop()

    def gray_histogram(self):
        self.plot_histogram_base(is_color=False)

    def color_histogram(self):
        self.plot_histogram_base(is_color=True)
        
    # ===  梯度邊緣銳化  ===
    def edge_sharpen_filter(self, filter_type):
        if self.path == '' or self.label_scr_image is None: return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])

        # 1. Canny 邊緣偵測 (特殊處理：因為 Canny 不是單純的卷積核)
        if filter_type == 'Canny':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img_filtered = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.show_filtered_image(img_filtered)
            return

        # 2. Laplacian 大尺寸 (直接呼叫內建函式最穩)
        if filter_type in ['Laplacian5x5', 'Laplacian7x7', 'Laplacian9x9']:
            ksize = int(filter_type[-3]) # 聰明抓取字串裡的數字 5, 7, 或 9
            img_filtered = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
            img_filtered = cv2.convertScaleAbs(img_filtered)
            self.show_filtered_image(img_filtered)
            return

        # 3. 建立自訂卷積核 (Kernel)
        kernel = None
        if filter_type == 'SobelGx': kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif filter_type == 'SobelGy': kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif filter_type == 'PrewittGx': kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        elif filter_type == 'PrewittGy': kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif filter_type == 'FreiChenGx': kernel = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])
        elif filter_type == 'FreiChenGy': kernel = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]])
        elif filter_type == 'ScharrGx': kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        elif filter_type == 'ScharrGy': kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        elif filter_type == 'Laplacian3x3-01': kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        elif filter_type == 'Laplacian3x3-02': kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        elif filter_type == 'Laplacian3x3-03': kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        elif filter_type == 'EdgeSharp3x3-01': kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        elif filter_type == 'EdgeSharp3x3-02': kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        elif filter_type == 'EdgeSharp3x3-03': kernel = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
        elif filter_type == 'EdgeSharp5x5':
            kernel = np.ones((5,5), np.float32) * -1
            kernel[2,2] = 25
        elif filter_type == 'EdgeSharp7x7':
            kernel = np.ones((7,7), np.float32) * -1
            kernel[3,3] = 49

        # 4. 套用 cv2.filter2D 並顯示 (完美利用 Lab 05 寫好的顯示引擎)
        if kernel is not None:
            img_filtered = cv2.filter2D(image, -1, kernel)
            self.show_filtered_image(img_filtered)
    
    # =========================================================
    # === Lab 11: RYGB色彩檢測 (精準去背 + 第5張反轉純白) ===
    def color_detection(self):
        # 1. 隱藏不必要的按鈕
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        
        # 2. 讓使用者開啟 RYGB-r90-r2 圖片
        open_img_path = tkfd.askopenfilename(
            initialdir='D:/anaconda/images', 
            filetypes=[("Image Files", "*.jpg *.png *.bmp *.jpeg")], 
            title='開啟測試影像 (RYGB-r90-r2)'
        )
        if not open_img_path: return
        self.path = open_img_path
        
        # 3. 讀取影像並顯示於左側來源窗
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)
        cv2image_src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil_src = Image.fromarray(cv2image_src)
        tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
        
        if self.label_scr_image is None:
            self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
            self.label_scr_image.pack()
        self.label_scr_image.configure(image=tk_image_src)
        self.label_scr_image.image = tk_image_src

        # 4. 轉換至 HSV 空間以利色彩分割
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 💡 修正其他顏色會抓到「灰灰的」背景的問題：
        # 把 S(飽和度) 和 V(明度) 的底限拉高到 100，這樣就不會把暗灰色的軟墊誤認為顏色！
        
        # (1) 紅色 (Red)
        mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        res_red = cv2.bitwise_and(image, image, mask=mask_red)
        
        # (2) 黃色 (Yellow)
        mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
        res_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
        
        # (3) 綠色 (Green)
        mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))
        res_green = cv2.bitwise_and(image, image, mask=mask_green)
        
        # (4) 藍色 (Blue)
        mask_blue = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([130, 255, 255]))
        res_blue = cv2.bitwise_and(image, image, mask=mask_blue)

        # 💡 完美還原第 5 張圖 (Detection: Black) - 終極無縫版
        # 1. 稍微放寬 V(明度) 的上限到 130，包容軟墊的細微反光
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 130]))
        
        # 2. 祭出形態學「連續技」(完全應用 Lab 08 的精華)！
        # 第一招：開運算 (Opening) -> 專門消除積木內部的「細長縫隙 (陰影)」
        # 縫隙可能有點粗，我們用 9x9 的大核心來徹底抹除這些誤判的黑色線條
        kernel_open = np.ones((9, 9), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel_open)
        
        # 第二招：閉運算 (Closing) -> 專門填補軟墊上的「細微反光 (破洞)」
        kernel_close = np.ones((5, 5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. 套用乾淨的遮罩
        res_black = cv2.bitwise_and(image, image, mask=mask_black) 
        
        # 4. 把不是黑色的地方塗成純白色 (此時縫隙已經被抹平，會變成完整的一大塊純白)
        res_black[mask_black == 0] = [255, 255, 255]

        # 5. 彈出 5 個 OpenCV 視窗
        cv2.imshow('Detection: Red', res_red)
        cv2.imshow('Detection: Yellow', res_yellow)
        cv2.imshow('Detection: Green', res_green)
        cv2.imshow('Detection: Blue', res_blue)
        cv2.imshow('Detection: Black', res_black)
        
        # 6. 同步將原圖顯示在右側的「目標影像區」，讓介面不留白
        if self.label_des_image is None:
            self.label_des_image = tk.Label(self.frame_des, image=tk_image_src)
            self.label_des_image.pack()
        self.label_des_image.configure(image=tk_image_src)
        self.label_des_image.image = tk_image_src

    # 即時紅藍綠黃色球(瓶蓋)檢測 
    def redball_tracking(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()

        # 中斷之前的鏡頭占用
        if hasattr(self, 'video_loop_id') and self.video_loop_id is not None:
            self.root.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tkmsg.showwarning("警告", "無法開啟攝影機！")
            return
            
        # 啟動色彩追蹤的專屬迴圈
        self.update_color_tracking_loop()

    def update_color_tracking_loop(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 鏡像翻轉，讓操作更直覺
                
                # A. 來源窗顯示原始視訊
                cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil_src = Image.fromarray(cv2image_src)
                tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
                if self.label_scr_image is None:
                    self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                    self.label_scr_image.pack()
                self.label_scr_image.configure(image=tk_image_src)
                self.label_scr_image.image = tk_image_src

                # B. 轉換至 HSV 空間進行色彩檢測
                hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                imageFrame = frame.copy() # 用來畫框框的畫布
                
                # 定義四種顏色的 HSV 範圍與對應的標籤顏色 (B, G, R)
                colors_data = [
                    ("Red Colour", np.array([136, 87, 111]), np.array([180, 255, 255]), (0, 0, 255)),
                    ("Yellow Colour", np.array([20, 100, 100]), np.array([35, 255, 255]), (0, 255, 255)),
                    ("Green Colour", np.array([25, 52, 72]), np.array([102, 255, 255]), (0, 255, 0)),
                    ("Blue Colour", np.array([94, 80, 2]), np.array([120, 255, 255]), (255, 0, 0))
                ]
                
                # 依序掃描每種顏色
                for color_name, lower, upper, box_color in colors_data:
                    mask = cv2.inRange(hsvFrame, lower, upper)
                    
                    # 使用形態學膨脹 (Dilation) 填補破洞與消除細微雜訊
                    kernel = np.ones((5, 5), "uint8")
                    mask = cv2.dilate(mask, kernel)
                    
                    # 尋找輪廓 (Contours)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        # 面積大於 2000 才畫框 (過濾掉背景雜訊)
                        if area > 2000: 
                            x, y, w, h = cv2.boundingRect(contour)
                            # 畫方框
                            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), box_color, 2)
                            # 寫文字
                            cv2.putText(imageFrame, color_name, (x, y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                # C. 目的窗顯示畫完框框的影像
                cv2image_des = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
                img_pil_des = Image.fromarray(cv2image_des)
                tk_image_des = ImageTk.PhotoImage(image=img_pil_des)

                if self.label_des_image is None:
                    self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                    self.label_des_image.pack()
                self.label_des_image.configure(image=tk_image_des)
                self.label_des_image.image = tk_image_des

                # 循環呼叫達成即時視訊效果 (約 30 毫秒一幀)
                self.video_loop_id = self.root.after(30, self.update_color_tracking_loop)
            else:
                self.cap.release()
    
    def scale_pyrup(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        img_pyrup = cv2.pyrUp(image) # 高斯金字塔
        image_pil_pyrup = Image.fromarray(img_pyrup)
        self.des_image_pil = image_pil_pyrup
        tk_image = ImageTk.PhotoImage(image_pil_pyrup)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.pack()
        self.root.mainloop()        
         
    def scale_pyrdown(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        img_pyrdown = cv2.pyrDown(image) # 高斯金字塔
        image_pil_pyrdown = Image.fromarray(img_pyrdown)
        self.des_image_pil = image_pil_pyrdown
        tk_image = ImageTk.PhotoImage(image_pil_pyrdown)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(relx=0,rely=0)# 放置元件的不同方式
        self.label_des_image.pack()
        self.root.mainloop() 
    # 放大
    def scale_zoom_in(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        size = (2*image.shape[1], 2*image.shape[0])
        img_zoom_in = cv2.resize(image, size) # 放大
        image_pil_zoom_in = Image.fromarray(img_zoom_in)
        self.des_image_pil = image_pil_zoom_in
        tk_image = ImageTk.PhotoImage(image_pil_zoom_in)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        self.label_des_image.place(x=0, y=0)
        # 放置元件的不同方式與金字塔放大相比對齊方式不同顯示不同
        # self.label_des_image.pack()
        self.root.mainloop()
    # 縮小
    def scale_zoom_out(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        size = (int(0.3*image.shape[1]), int(0.3*image.shape[0]))
        img_zoom_out = cv2.resize(image, size) # 放大
        image_pil_zoom_out = Image.fromarray(img_zoom_out)
        self.des_image_pil = image_pil_zoom_out
        tk_image = ImageTk.PhotoImage(image_pil_zoom_out)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 平移
    def rotate_offset(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        direction = np.float32([[1,0,50],[0,1,50]])# 沿x軸移動50，沿y軸移動50
        img_offset = cv2.warpAffine(image, direction, (width, height))
        image_pil_offset = Image.fromarray(img_offset)
        self.des_image_pil = image_pil_offset
        tk_image = ImageTk.PhotoImage(image_pil_offset)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 仿射-需要三個點座標
    def rotate_affine(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        rot_mat = cv2.getAffineTransform(pts1,pts2)# 沿x軸移動50，沿y軸移動50
        img_affine = cv2.warpAffine(image, rot_mat, (width, height))
        image_pil_affine = Image.fromarray(img_affine)
        self.des_image_pil = image_pil_affine
        tk_image = ImageTk.PhotoImage(image_pil_affine)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 透射 -需要四個點的座標
    def rotate_transmission(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        pts1 = np.float32([[56,65],[238,52],[28,237],[239,240]])
        pts2 = np.float32([[0,0],[250,0],[0,250],[250,250]])
        rot_mat = cv2.getPerspectiveTransform(pts1,pts2)
        img_clockwise = cv2.warpPerspective(image, rot_mat, (250, 250))
        # 透射與仿射的函數不一樣
        image_pil_clockwise = Image.fromarray(img_clockwise)
        self.des_image_pil = image_pil_clockwise
        tk_image = ImageTk.PhotoImage(image_pil_clockwise)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
     
     
    # 順時針無縮放   
    def rotate_clockwise(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        rotate_center = (width//2, height//2)
        rot_mat = cv2.getRotationMatrix2D(rotate_center, angle = -45, scale = 1) 
        # 旋轉中心rotate_center，角度degree， 縮放scale
        img_clockwise = cv2.warpAffine(image, rot_mat, (width, height))
        image_pil_clockwise = Image.fromarray(img_clockwise)
        self.des_image_pil = image_pil_clockwise
        tk_image = ImageTk.PhotoImage(image_pil_clockwise)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 順時針-縮放
    def rotate_clockwise_zoom(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        rotate_center = (width//2, height//2)
        rot_mat = cv2.getRotationMatrix2D(rotate_center, angle = -45, scale = 0.6)
        # 旋轉中心rotate_center，角度degree， 縮放scale
        img_clockwise_zoom = cv2.warpAffine(image, rot_mat, (width, height))
        image_pil_clockwise_zoom = Image.fromarray(img_clockwise_zoom)
        self.des_image_pil = image_pil_clockwise_zoom
        tk_image = ImageTk.PhotoImage(image_pil_clockwise_zoom)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 逆時針-縮放
    def rotate_anti_zoom(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        rotate_center = (width//2, height//2)
        rot_mat = cv2.getRotationMatrix2D(rotate_center, angle = 45, scale = 0.6) 
        # 旋轉中心rotate_center，角度degree， 縮放scale
        img_clockwise_zoom = cv2.warpAffine(image, rot_mat, (width, height))
        image_pil_clockwise_zoom = Image.fromarray(img_clockwise_zoom)
        self.des_image_pil = image_pil_clockwise_zoom
        tk_image = ImageTk.PhotoImage(image_pil_clockwise_zoom)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
    # 零旋轉-縮放
    def rotate_zero_zoom(self):
        if(self.path == ''):
            return
        if(self.label_scr_image == None):
            return
        image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), 1)# 讀取圖片
        b, g, r = cv2.split(image)# 三通道分離
        image = cv2.merge([r,g,b])# 三通道合併
        width, height = image.shape[1], image.shape[0]
        rotate_center = (width//2, height//2)
        rot_mat = cv2.getRotationMatrix2D(rotate_center, angle = 0, scale = 0.6) 
        # 旋轉中心rotate_center，角度degree， 縮放scale
        img_zero_zoom = cv2.warpAffine(image, rot_mat, (width, height))
        image_pil_zero_zoom = Image.fromarray(img_zero_zoom)
        self.des_image_pil = image_pil_zero_zoom
        tk_image = ImageTk.PhotoImage(image_pil_zero_zoom)
        if (self.label_des_image == None):
            self.label_des_image = tk.Label(self.frame_des,image = tk_image)
        self.label_des_image.configure(image = tk_image)
        #self.label_des_image.place(x=0, y=0)
        self.label_des_image.pack()
        self.root.mainloop()
        
    # === Lab 09 & Lab 10: 即時影像邊緣處理共用引擎 ===

    def start_realtime_video(self, filter_mode, is_cam=True):
        # 1. 隱藏不必要的介面按鈕
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()

        # 2. 如果之前有影片在播，先中斷並釋放資源
        if self.video_loop_id is not None:
            self.root.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        if self.cap is not None:
            self.cap.release()
        if hasattr(self, 'auto_video_writer') and self.auto_video_writer is not None:
            self.auto_video_writer.release()

        # 3. 判斷是開鏡頭(Lab 9)還是選檔案(Lab 10)
        if is_cam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                tkmsg.showwarning("警告", "無法開啟攝影機！")
                return
        else:
            video_path = tkfd.askopenfilename(
                initialdir='D:/anaconda/images',
                title='選擇視訊檔案',
                filetypes=[('影片檔', '*.mp4 *.avi *.mkv'), ('所有檔案', '*')]
            )
            if not video_path: return
            self.cap = cv2.VideoCapture(video_path)

        # 4. 設定自動存檔 (VideoWriter) - 檔名為 年-月-日-時-分-秒.mp4
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0 if is_cam else self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: fps = 20.0 # 防呆機制

        ts = datetime.datetime.now()
        filename = ts.strftime("%Y-%m-%d-%H-%M-%S.mp4")
        save_path = os.path.join("D:/anaconda/images", filename) # 預設存到 images 資料夾
        
        # 使用 mp4v 編碼器存成 mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.auto_video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        self.realtime_filter_mode = filter_mode
        self.update_realtime_loop()

    def update_realtime_loop(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 為了避免方向相反，若是 CAM 則左右翻轉
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0 or getattr(self, 'realtime_filter_mode', '') != '':
                     # 簡單防呆判斷是否為串流
                     pass

                # === A. 來源窗顯示原始視訊 ===
                cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil_src = Image.fromarray(cv2image_src)
                tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
                if self.label_scr_image is None:
                    self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                    self.label_scr_image.pack()
                self.label_scr_image.configure(image=tk_image_src)
                self.label_scr_image.image = tk_image_src

                # === B. 核心濾波處理 ===
                processed_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.realtime_filter_mode == 'SobelX':
                    edge = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
                    processed_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                elif self.realtime_filter_mode == 'SobelY':
                    edge = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
                    processed_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                elif self.realtime_filter_mode == 'Laplacian':
                    edge = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
                    processed_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                elif self.realtime_filter_mode == 'Canny':
                    edge = cv2.Canny(gray, 100, 200)
                    processed_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                elif self.realtime_filter_mode == 'MG':
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                    processed_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

                # === C. 自動寫入儲存為 mp4 ===
                if hasattr(self, 'auto_video_writer') and self.auto_video_writer is not None:
                    self.auto_video_writer.write(processed_frame)

                # === D. 目的窗顯示處理後影像 ===
                cv2image_des = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil_des = Image.fromarray(cv2image_des)
                tk_image_des = ImageTk.PhotoImage(image=img_pil_des)
                
                # 清除直方圖防呆
                if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                    self.frame_des.canvas.get_tk_widget().pack_forget()
                    self.frame_des.canvas = None
                if hasattr(self, 'toolbar') and self.toolbar != None:
                    self.toolbar.pack_forget()
                    self.toolbar = None

                if self.label_des_image is None:
                    self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                    self.label_des_image.pack()
                self.label_des_image.configure(image=tk_image_des)
                self.label_des_image.image = tk_image_des

                # 循環呼叫自己 (約 30 毫秒一幀，達成影片效果)
                self.video_loop_id = self.root.after(30, self.update_realtime_loop)
            else:
                # 影片播放完畢或是斷線時的收尾工作
                if hasattr(self, 'auto_video_writer') and self.auto_video_writer is not None:
                    self.auto_video_writer.release()
                    self.auto_video_writer = None
                tkmsg.showinfo("完成", "即時處理完畢，影片已自動儲存！")
                self.cap.release()
                
    # =========================================================
    # === Lab 12+: 影像 Panorama 全景影像無痕拼接引擎 ===

    # 綁定選單的三個獨立呼叫點
    def ORB_Panorama(self): self.panorama_stitching_engine("ORB")
    def SIFT_Panorama(self): self.panorama_stitching_engine("SIFT")
    def BRISK_Panorama(self): self.panorama_stitching_engine("BRISK")

    def panorama_stitching_engine(self, feature_type):
        # 1. 彈出對話盒，選取來源影像資料夾
        folder_path = tkfd.askdirectory(title=f"選擇包含要拼接影像的資料夾 ({feature_type})")
        if not folder_path: return

        # 2. 抓取資料夾內的所有支援影像 (改良版，更安全、支援更多副檔名)
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
                      
        if len(image_paths) < 2:
            tkmsg.showwarning("警告", "資料夾內至少需要 2 張圖片才能進行全景拼接！\n(請確認你選的是『資料夾』，且裡面有圖片)")
            return

        # 支援中文路徑的圖片讀取方式
        images = []
        for path in image_paths:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            if img is not None:
                images.append(img)

        # 3. 執行全景拼接
        tkmsg.showinfo("處理中", f"正在使用 {feature_type} 相關技術進行無痕拼接...\n(依圖片數量與解析度，這可能需要數十秒的運算時間，請按確定後耐心等候)")
        
        # 呼叫 OpenCV 的 Stitcher 引擎
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched_img = stitcher.stitch(images)

        # 錯誤捕捉與防呆機制
        if status != cv2.Stitcher_OK:
            error_msg = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "需要更多圖片(圖片間的重疊區域或特徵點不足)",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "單應性矩陣估計失敗(圖片關聯性太低)",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相機參數調整失敗"
            }.get(status, f"未知錯誤代碼: {status}")
            tkmsg.showerror("拼接失敗", f"無法完成全景拼接！\n原因: {error_msg}")
            return

        # 4. 另開一個有游標(Scrollbar)功能的視窗顯示拼接後影像
        self.show_large_panorama_window(stitched_img, f"Panorama 全景拼接結果 - {feature_type}")

        # 5. 系統跳出存檔對話盒
        save_path = tkfd.asksaveasfilename(
            title="儲存拼接後的全景影像",
            defaultextension=".jpg",
            filetypes=[("JPEG 格式", "*.jpg"), ("PNG 格式", "*.png"), ("所有檔案", "*.*")]
        )
        if save_path:
            # 💡 防呆機制：如果使用者忘記打副檔名，自動補上 .jpg
            _, ext = os.path.splitext(save_path)
            if ext == '':
                ext = '.jpg'
                save_path += ext
                
            # 支援中文路徑的存檔方式
            is_success, im_buf_arr = cv2.imencode(ext, stitched_img)
            if is_success:
                im_buf_arr.tofile(save_path)
                tkmsg.showinfo("成功", f"全景影像已成功儲存至指定路徑！")
            else:
                tkmsg.showerror("錯誤", "無法存檔，請確認副檔名是否正確！")

    def show_large_panorama_window(self, cv_img, title):
        # 建立獨立的 Toplevel 子視窗
        top = tk.Toplevel(self.root)
        top.title(title)
        
        # 設定視窗大小為 1024x768 
        top.geometry("1024x768")

        # 將 OpenCV BGR 轉為 PIL 可顯示的 RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # 建立 Canvas 與垂直/水平卷軸
        canvas = tk.Canvas(top, bg="gray")
        vbar = ttk.Scrollbar(top, orient=tk.VERTICAL, command=canvas.yview)
        hbar = ttk.Scrollbar(top, orient=tk.HORIZONTAL, command=canvas.xview)
        
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        # 排版卷軸與畫布
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 將圖片放入 Canvas 中
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        
        # 💡 關鍵防呆機制：必須將圖片物件綁定在 Canvas 上，否則一顯示就會變成空白！
        canvas.image = tk_img 
        
        # 設定卷軸的滑動範圍為整張拼接圖片的真實物理大小
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
    # =========================================================
    # === Lab 13: 人臉辨識系統功能實作 ===

    def haar_face_detection(self):
        # 1. 隱藏不必要的按鈕 (保持介面乾淨)
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()

        # 安全關閉先前的攝影機迴圈
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
        if self.cap is not None:
            self.cap.release()

        # 2. 載入我們剛剛下載的四個特徵模型 (請確認路徑是否正確)
        xml_path = 'D:/anaconda/images/' 
        face_cascade = cv2.CascadeClassifier(xml_path + 'face.xml')
        eye_cascade = cv2.CascadeClassifier(xml_path + 'eye.xml')
        nose_cascade = cv2.CascadeClassifier(xml_path + 'nose.xml')
        mouth_cascade = cv2.CascadeClassifier(xml_path + 'mouth.xml')
        
        # 簡單防呆：檢查模型是否真的有讀到
        if face_cascade.empty():
            tkmsg.showwarning("警告", f"找不到 Haar 模型檔！\n請確認 {xml_path} 下有 face.xml 等檔案。")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tkmsg.showwarning("警告", "無法開啟攝影機！")
            return
        
        def update_haar():
            if self.cap is None or not self.cap.isOpened(): return
            ret, frame = self.cap.read()
            if not ret:
                self.video_loop_id = self.root.after(30, update_haar)
                return
                
            frame = cv2.flip(frame, 1) # 鏡像翻轉
            
            # === A. 來源窗顯示「原始影像」 ===
            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_src = Image.fromarray(cv2image_src)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src
            
            # === B. 進行 Haar 偵測並畫框 ===
            annotated = frame.copy() # 複製一份用來畫框的畫布
            gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
            
            # 先偵測「人臉」
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                # 畫出臉部框框 (藍色)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 鎖定這張臉的區域 (ROI, Region of Interest)，只在這裡面找五官
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = annotated[y:y+h, x:x+w]
                
                # 在臉部區域內找「眼睛」
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                for (ex, ey, ew, eh) in eyes:
                    # 💡 顏色已互換：(0, 0, 255) 在 BGR 格式中是「紅色」
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2) 
                    
                # 在臉部區域內找「鼻子」
                noses = nose_cascade.detectMultiScale(roi_gray, 1.1, 5)
                for (nx, ny, nw, nh) in noses:
                    # 鼻子保持黃色不變
                    cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 2) 
                    
                # 在臉部區域內找「嘴巴」
                mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
                for (mx, my, mw, mh) in mouths:
                    # 💡 顏色已互換：(0, 255, 0) 在 BGR 格式中是「綠色」
                    cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
            
            # === C. 目標窗顯示「偵測結果」 ===
            cv2image_des = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil_des = Image.fromarray(cv2image_des)
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)

            # 防呆：清空其他功能留下的 matplotlib 畫布
            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des
            
            self.video_loop_id = self.root.after(30, update_haar)
        
        update_haar()

    def dlib_face_recognition(self):
        # 此功能需要加載你透過 features_extraction_to_csv.py 產生的 features_all.csv
        # 這裡示範呼叫 face_reco_from_camera 的核心邏輯
        tkmsg.showinfo("提示", "請確保已完成『擷取樣本』與『訓練模型』，系統將載入 CSV 進行比對。")
        # 實作邏輯請參考 face_reco_from_camera.py
        pass

    def get_face_samples(self):
        # 呼叫你上傳的 get_faces_from_camera.py 邏輯
        # 建議直接執行獨立腳本進行採集，以確保儲存路徑正確
        pass
    
    def train_face_model(self):
        # 呼叫你上傳的 features_extraction_to_csv.py 邏輯
        tkmsg.showinfo("提示", "為了確保路徑正確，請開啟 Anaconda Prompt 並輸入：\npython features_extraction_to_csv.py")
        pass
    
  # =========================================================
    # === Lab 14: 1. Cam人臉註冊 (建檔) - 自訂姓名版 ===
    def dlib_register(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()

        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass

        tkmsg.showinfo("註冊說明", "即將開啟鏡頭！\n\n【操作方法】\n👉 按鍵盤 'n'：輸入名字並建立資料夾\n👉 按鍵盤 's'：拍下並儲存臉部照片\n👉 按鍵盤 'q'：結束註冊")

        # 引入彈出輸入框的套件
        from tkinter import simpledialog
        face_cascade = cv2.CascadeClassifier('D:/anaconda/images/face.xml')

        ROOT_DIR = "D:/anaconda/homework/"
        base_dir = ROOT_DIR + "data/data_faces_from_camera/"
        os.makedirs(base_dir, exist_ok=True)

        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(0)

        # 狀態變數
        self.reg_current_dir = ""
        self.reg_pic_cnt = 0
        self.reg_can_save = False
        self.reg_frame = None
        self.reg_face_box = None
        self.reg_person_name = "尚未設定" # 新增名字變數

        def key_handler(event):
            char = event.char.lower()
            if char == 'n':
                # 💡 彈出對話框讓使用者輸入名字，預設帶入「馮志鋒」
                name = simpledialog.askstring("輸入姓名", "請輸入這張臉的名字 (中英文皆可):", parent=self.root, initialvalue="馮志鋒")
                if not name: return # 按下取消則跳出
                
                self.reg_person_name = name
                self.reg_current_dir = os.path.join(base_dir, name)
                os.makedirs(self.reg_current_dir, exist_ok=True)
                self.reg_pic_cnt = 0
                print(f"✅ 已建立專屬資料夾: {self.reg_current_dir}")
                
            elif char == 's':
                if self.reg_current_dir == "":
                    print("⚠️ 請先按 'n' 輸入姓名建立資料夾！")
                    return
                if self.reg_can_save and self.reg_frame is not None and self.reg_face_box is not None:
                    x, y, w, h = self.reg_face_box
                    hh, ww = int(h/4), int(w/4)
                    face_crop = self.reg_frame[max(0, y-hh):min(self.reg_frame.shape[0], y+h+hh), 
                                               max(0, x-ww):min(self.reg_frame.shape[1], x+w+ww)]
                    if face_crop.size > 0:
                        self.reg_pic_cnt += 1
                        filepath = os.path.join(self.reg_current_dir, f"img_face_{self.reg_pic_cnt}.jpg")
                        cv2.imwrite(filepath, face_crop)
                        print(f"📸 儲存照片: {filepath}")
            elif char == 'q':
                self.root.unbind('<Key>') 
                if hasattr(self, 'video_loop_id'): self.root.after_cancel(self.video_loop_id)
                if self.cap is not None: self.cap.release()
                self.clear()
                tkmsg.showinfo("結束", "註冊結束，請點擊「2. 人臉128D特徵提取」。")

        self.root.bind('<Key>', key_handler)

        def update_register():
            if self.cap is None or not self.cap.isOpened(): return
            ret, frame = self.cap.read()
            if not ret:
                self.video_loop_id = self.root.after(30, update_register)
                return

            frame = cv2.flip(frame, 1)
            self.reg_frame = frame.copy()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            self.reg_can_save = False
            if len(faces) == 1:
                self.reg_face_box = faces[0]
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.reg_can_save = True
                cv2.putText(frame, "Face OK! (Press 's' to save)", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
            elif len(faces) > 1:
                cv2.putText(frame, "Too many faces!", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

            # 💡 畫面上顯示目前正在為誰拍照
            status_text = f"Name: {self.reg_person_name} | Saved: {self.reg_pic_cnt} photos"
            cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 1)

            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tk_image_src = ImageTk.PhotoImage(image=Image.fromarray(cv2image_src))
            
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr)
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src
            self.label_scr_image.pack() 

            self.video_loop_id = self.root.after(30, update_register)

        update_register()

    # =========================================================
    # === Lab 14: 2. 人臉128D特徵提取 (訓練) - 姓名綁定版 ===
    def dlib_extract(self):
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
                
        tkmsg.showinfo("特徵提取", "即將開始將照片轉換為 128D 特徵並綁定姓名。\n⚠️ 這可能需要幾十秒鐘，畫面會暫停，請耐心等候...")
        
        import dlib, csv, os, cv2
        import numpy as np
        
        ROOT_DIR = "D:/anaconda/homework/"
        model_68 = ROOT_DIR + 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
        model_128 = ROOT_DIR + 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
        base_dir = ROOT_DIR + "data/data_faces_from_camera/"
        csv_path = ROOT_DIR + "data/features_all.csv"

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_68)
        face_reco_model = dlib.face_recognition_model_v1(model_128)
        
        # 💡 讀取所有資料夾名稱 (人名)
        person_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not person_names:
            tkmsg.showerror("錯誤", f"在 {base_dir} 找不到任何人員資料夾！")
            return
            
        features_all = []
        names_all = []
        
        for name in person_names:
            person_dir = os.path.join(base_dir, name)
            photos = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
            person_features = []
            
            for photo in photos:
                img_path = os.path.join(person_dir, photo)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                faces = detector(img_rgb, 1)
                if len(faces) > 0:
                    shape = predictor(img_rgb, faces[0])
                    face_descriptor = face_reco_model.compute_face_descriptor(img_rgb, shape)
                    person_features.append(face_descriptor)
            
            if person_features:
                features_mean = np.array(person_features).mean(axis=0)
                features_all.append(features_mean)
                names_all.append(name) # 記錄這組特徵對應的名字
        
        # 💡 將 128D 特徵與姓名寫入同一列 (使用 utf-8-sig 確保中文不亂碼)
        with open(csv_path, "w", newline="", encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            for name, feature in zip(names_all, features_all):
                row = list(feature) + [name]
                writer.writerow(row)
                
        tkmsg.showinfo("完成", f"✅ 大功告成！\n已成功將 {len(features_all)} 個人的特徵與姓名寫入資料庫！")

    # =========================================================
    # === Lab 14: 3. Cam即時人臉辨識 (Demo) - 讀取姓名版 ===
    def dlib_recognize(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()

        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
        
        import pandas as pd
        import dlib
        import numpy as np
        import os
        from PIL import ImageDraw, ImageFont
        
        ROOT_DIR = "D:/anaconda/homework/"
        csv_path = ROOT_DIR + "data/features_all.csv"
        model_68 = ROOT_DIR + 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
        model_128 = ROOT_DIR + 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
            
        feature_known_list = []
        name_known_list = []
        
        # 💡 讀取包含中文的 CSV 資料庫
        csv_rd = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(128):
                val = csv_rd.iloc[i, j]
                features_someone_arr.append(float(val) if pd.notna(val) else 0.0)
            feature_known_list.append(np.array(features_someone_arr))
            
            # 從第 129 個欄位 (索引 128) 讀取名字
            if csv_rd.shape[1] > 128:
                name = str(csv_rd.iloc[i, 128])
            else:
                name = f"Unknown_Person_{i}"
            name_known_list.append(name)
            
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_68)
        face_reco_model = dlib.face_recognition_model_v1(model_128)
        
        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(0)
        
        def update_recognize():
            if self.cap is None or not self.cap.isOpened(): return
            ret, frame = self.cap.read()
            if not ret:
                self.video_loop_id = self.root.after(30, update_recognize)
                return
                
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img_pil_src = Image.fromarray(img_rgb)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src

            faces = detector(img_rgb, 0)
            img_pil_des = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil_des)
            try:
                font = ImageFont.truetype("msjh.ttc", 30) 
            except:
                font = ImageFont.load_default()
            
            for face in faces:
                shape = predictor(img_rgb, face)
                face_descriptor = face_reco_model.compute_face_descriptor(img_rgb, shape)
                feature_current = np.array(face_descriptor)
                
                e_distances = []
                for feature_known in feature_known_list:
                    if np.sum(feature_known) == 0.0:
                        e_distances.append(999.0)
                    else:
                        dist = np.linalg.norm(feature_current - feature_known)
                        e_distances.append(dist)
                        
                min_dist = min(e_distances)
                min_idx = e_distances.index(min_dist)
                
                if min_dist < 0.4:
                    name = name_known_list[min_idx]
                    color = (0, 255, 0) 
                else:
                    name = "未知訪客"
                    color = (255, 0, 0) 
                    
                draw.rectangle([(face.left(), face.top()), (face.right(), face.bottom())], outline=color, width=3)
                draw.text((face.left(), face.top() - 40), f"{name} ({min_dist:.2f})", font=font, fill=color)
                            
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)
            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des
            
            self.video_loop_id = self.root.after(30, update_recognize)
            
        update_recognize()
        
    # =========================================================
    # === 終極版：ArcFace (InsightFace) 特徵提取 ===
    def arcface_extract(self):
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
                
        tkmsg.showinfo("特徵提取", "即將開始將照片轉換為 ArcFace 512D 特徵。\n⚠️ 首次載入模型可能需要幾十秒鐘，請耐心等候...")
        
        import csv, os, cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        
        ROOT_DIR = "D:/anaconda/homework/"
        base_dir = ROOT_DIR + "data/data_faces_from_camera/"
        csv_path = ROOT_DIR + "data/arcface_features.csv" # 存成全新的 CSV 資料庫

        # 1. 載入 ArcFace 引擎 (buffalo_l 模型)
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        person_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not person_names:
            tkmsg.showerror("錯誤", f"在 {base_dir} 找不到任何人員資料夾！")
            return
            
        features_all = []
        names_all = []
        
        for name in person_names:
            person_dir = os.path.join(base_dir, name)
            photos = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
            person_features = []
            
            for photo in photos:
                img_path = os.path.join(person_dir, photo)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                
                # InsightFace 預設就是吃 BGR 格式，直接丟進去即可！
                faces = app.get(img_bgr)
                if len(faces) > 0:
                    # 取出這張臉的 512 維特徵
                    person_features.append(faces[0].embedding)
            
            if person_features:
                features_mean = np.array(person_features).mean(axis=0)
                features_all.append(features_mean)
                names_all.append(name) 
        
        # 寫入 CSV
        with open(csv_path, "w", newline="", encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            for name, feature in zip(names_all, features_all):
                row = list(feature) + [name]
                writer.writerow(row)
                
        tkmsg.showinfo("完成", f"✅ ArcFace 訓練完成！\n已成功將 {len(features_all)} 個人的特徵與姓名寫入資料庫！")

    # =========================================================
    # === 終極版：ArcFace (InsightFace) 即時辨識與 SQLite 打卡 ===
    def arcface_recognize(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()

        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
        
        import pandas as pd
        import numpy as np
        import os, cv2, time, sqlite3, datetime
        from PIL import Image, ImageTk, ImageDraw, ImageFont
        from insightface.app import FaceAnalysis
        from sklearn.metrics.pairwise import cosine_similarity
        
        ROOT_DIR = "D:/anaconda/homework/"
        csv_path = ROOT_DIR + "data/arcface_features.csv"
        db_path = ROOT_DIR + "data/face_log.db"
        
        if not os.path.exists(csv_path):
            tkmsg.showerror("錯誤", f"找不到 {csv_path}！\n請先執行第 2 步 ArcFace 特徵提取。")
            return
            
        feature_known_list = []
        name_known_list = []
        
        # 1. 讀取 CSV 建立 512 維資料庫
        csv_rd = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(512): # 💡 ArcFace 特徵是 512 維
                val = csv_rd.iloc[i, j]
                features_someone_arr.append(float(val) if pd.notna(val) else 0.0)
            feature_known_list.append(np.array(features_someone_arr))
            
            if csv_rd.shape[1] > 512:
                name = str(csv_rd.iloc[i, 512])
            else:
                name = f"Unknown_{i}"
            name_known_list.append(name)
            
        # 2. 載入 ArcFace 引擎
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 3. 建立 SQLite 資料庫連線 (沿用你上傳的邏輯)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS face_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                confidence REAL,
                time TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        last_seen = {} # 用來防止同一秒內重複瘋狂打卡
        
        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(0)
        
        def update_recognize():
            if self.cap is None or not self.cap.isOpened(): return
            ret, frame = self.cap.read()
            if not ret:
                self.video_loop_id = self.root.after(30, update_recognize)
                return
                
            frame = cv2.flip(frame, 1)
            
            # --- A. 左側顯示原始畫面 ---
            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_src = Image.fromarray(cv2image_src)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src

            # --- B. ArcFace 辨識與打卡核心 ---
            faces = app.get(frame) # InsightFace 直接吃 BGR 影像
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_des = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil_des)
            try: font = ImageFont.truetype("msjh.ttc", 30) 
            except: font = ImageFont.load_default()
            
            for face in faces:
                emb = face.embedding
                
                # 💡 使用你上傳檔案裡的「餘弦相似度 (Cosine Similarity)」來計算
                if len(feature_known_list) > 0:
                    sims = cosine_similarity([emb], feature_known_list)[0]
                    best_idx = np.argmax(sims)
                    conf = sims[best_idx]
                else:
                    conf = 0; best_idx = -1
                
                x1, y1, x2, y2 = face.bbox.astype(int)
                
                # 門檻值設定 0.5 (InsightFace 的建議標準)
                if conf >= 0.5 and best_idx != -1:
                    name = name_known_list[best_idx]
                    color = (0, 255, 0) # 綠色
                    
                    # 💡 打卡寫入資料庫：每 3 秒才會記錄一次同一個人
                    now = time.time()
                    if name not in last_seen or now - last_seen[name] > 3:
                        conn = sqlite3.connect(db_path)
                        c = conn.cursor()
                        c.execute("INSERT INTO face_log (name, confidence, time) VALUES (?, ?, ?)", 
                                  (name, float(conf), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        conn.commit()
                        conn.close()
                        last_seen[name] = now
                else:
                    name = "未知訪客"
                    color = (255, 0, 0) # 紅色
                    
                # 畫綠色/紅色框框與名字
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                draw.text((x1, y1 - 40), f"{name} ({conf*100:.1f}%)", font=font, fill=color)
                            
            # --- C. 右側顯示結果 ---
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)
            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des
            
            self.video_loop_id = self.root.after(30, update_recognize)
            
        update_recognize()
        
   # =========================================================
    # === YOLO v8 企業級即時雙螢幕整合引擎 (全功能完整版) ===
    # =========================================================

    def find_chinese_font(self):
        import os
        candidates = [
            "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc",
            "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simhei.ttf"
        ]
        for path in candidates:
            if os.path.exists(path): return path
        return None

    def draw_label_pil(self, img_bgr, text, pos):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 兼容不同版本的 PIL 套件
        try:
            bbox = draw.textbbox((0, 0), text, font=self.yolo_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=self.yolo_font)
            
        x, y = pos
        if y < 0: y = 0
        draw.rectangle([x, y, x + text_w + 6, y + text_h + 6], fill=(0, 0, 0))
        draw.text((x + 3, y + 3), text, font=self.yolo_font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def init_yolo(self):
        from ultralytics import YOLO
        import torch
        from PIL import ImageFont
        if not hasattr(self, 'yolo_model'):
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"YOLO loaded on: {self.yolo_device}")
            font_path = self.find_chinese_font()
            self.yolo_font = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
            
            # COCO 80 類別中英對照表
            self.EN_TO_ZH = {
                "person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "機車", 
                "airplane": "飛機", "bus": "公車", "train": "火車", "truck": "卡車", "boat": "船",
                "traffic light": "紅綠燈", "fire hydrant": "消防栓", "stop sign": "停止標誌", 
                "parking meter": "停車計時器", "bench": "長椅", "bird": "鳥", "cat": "貓", 
                "dog": "狗", "horse": "馬", "sheep": "羊", "cow": "牛", "elephant": "大象", 
                "bear": "熊", "zebra": "斑馬", "giraffe": "長頸鹿", "backpack": "背包", 
                "umbrella": "雨傘", "handbag": "手提包", "tie": "領帶", "suitcase": "行李箱", 
                "frisbee": "飛盤", "skis": "滑雪板", "snowboard": "滑雪單板", "sports ball": "球", 
                "kite": "風箏", "baseball bat": "棒球棒", "baseball glove": "棒球手套", 
                "skateboard": "滑板", "surfboard": "衝浪板", "tennis racket": "網球拍", 
                "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子", "fork": "叉子", 
                "knife": "刀子", "spoon": "湯匙", "bowl": "碗", "banana": "香蕉", "apple": "蘋果", 
                "sandwich": "三明治", "orange": "橘子", "broccoli": "花椰菜", "carrot": "胡蘿蔔", 
                "hot dog": "熱狗", "pizza": "披薩", "donut": "甜甜圈", "cake": "蛋糕", 
                "chair": "椅子", "couch": "沙發", "potted plant": "盆栽", "bed": "床", 
                "dining table": "餐桌", "toilet": "馬桶", "tv": "電視", "laptop": "筆電", 
                "mouse": "滑鼠", "remote": "遙控器", "keyboard": "鍵盤", "cell phone": "手機", 
                "microwave": "微波爐", "oven": "烤箱", "toaster": "烤麵包機", "sink": "水槽", 
                "refrigerator": "冰箱", "book": "書", "clock": "時鐘", "vase": "花瓶", 
                "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹風機", "toothbrush": "牙刷"
            }

    def init_yolo_db(self):
        import sqlite3, os
        self.yolo_db_path = "D:/anaconda/homework/data/yolo_detection.db"
        os.makedirs(os.path.dirname(self.yolo_db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.yolo_db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      label TEXT,
                      track_id INTEGER,
                      confidence REAL,
                      bbox TEXT,
                      roi_hit BOOLEAN,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        
        self.yolo_db_cooldown = {} # 防止同一個人在危險區瘋狂重複寫入資料庫
        # 預設危險區域 (ROI)
        self.roi_points = [[150, 150], [450, 150], [450, 350], [150, 350]]

    def insert_yolo_event(self, label, track_id, conf, bbox, roi_hit):
        import sqlite3
        conn = sqlite3.connect(self.yolo_db_path)
        c = conn.cursor()
        c.execute("INSERT INTO events (label, track_id, confidence, bbox, roi_hit) VALUES (?, ?, ?, ?, ?)",
                  (label, track_id, float(conf), str(bbox), roi_hit))
        conn.commit()
        conn.close()
        print(f"⚠️ 警報寫入資料庫: {label} (ID:{track_id}) 進入危險區域！")

    # === YOLO 功能按鈕邏輯 ===
    def yolo_toggle_lang(self):
        if getattr(self, 'yolo_lang_mode', 'zh') == "zh":
            self.yolo_lang_mode = "en"
            self.btn_yolo_lang.config(text="切換中文", bg="#9C27B0")
        else:
            self.yolo_lang_mode = "zh"
            self.btn_yolo_lang.config(text="切換英文", bg="#009688")

    def yolo_snapshot(self):
        import datetime, os, cv2
        if hasattr(self, 'yolo_frame') and self.yolo_frame is not None:
            ts = datetime.datetime.now()
            filename = f"YOLO_Snapshot_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            save_path = os.path.join("D:/anaconda/images/", filename)
            cv2.imwrite(save_path, self.yolo_frame)
            tkmsg.showinfo("截圖成功", f"YOLO 截圖已儲存至：\n{save_path}")

    def yolo_toggle_record(self):
        import datetime, os, cv2
        if self.cap is None or not self.cap.isOpened():
            tkmsg.showwarning("警告", "請先開啟 YOLO 視訊來源！")
            return
        
        # 如果還沒錄影，就開始錄影
        if not getattr(self, 'yolo_recording', False):
            ts = datetime.datetime.now()
            save_path = os.path.join("D:/anaconda/images/", f"YOLO_Record_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
            fps = 20.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.yolo_video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            self.yolo_recording = True
            self.btn_yolo_record.config(text="停止錄影", bg="#f44336")
            tkmsg.showinfo("開始錄影", "YOLO 錄影啟動！")
        else:
            if hasattr(self, 'yolo_video_writer') and self.yolo_video_writer: 
                self.yolo_video_writer.release()
                self.yolo_video_writer = None
            self.yolo_recording = False
            self.btn_yolo_record.config(text="開始錄影", bg="#607D8B")
            tkmsg.showinfo("完成", "YOLO 錄影已成功儲存！")

    def yolo_start_cam(self):
        self.yolo_start_source(0)

    def yolo_open_video(self):
        path = tkfd.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path: self.yolo_start_source(path)

    def yolo_open_rtsp(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("RTSP 設定")
        dialog.geometry("420x140")
        tk.Label(dialog, text="輸入 RTSP URL:").pack(pady=5)
        entry = tk.Entry(dialog, width=55)
        entry.pack(pady=5)
        def confirm():
            url = entry.get().strip()
            if url:
                dialog.destroy()
                self.yolo_start_source(url)
            else:
                tkmsg.showerror("錯誤", "RTSP URL 不可空白")
        tk.Button(dialog, text="確定", command=confirm).pack(pady=10)

    def yolo_open_youtube(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("YouTube 設定")
        dialog.geometry("450x140")
        tk.Label(dialog, text="輸入 YouTube 影片網址:").pack(pady=5)
        entry = tk.Entry(dialog, width=55)
        entry.pack(pady=5)
        def confirm():
            url = entry.get().strip()
            if url:
                dialog.destroy()
                self.yolo_start_youtube_stream(url)
            else:
                tkmsg.showerror("錯誤", "YouTube URL 不可空白")
        tk.Button(dialog, text="確定", command=confirm).pack(pady=10)

    def yolo_start_youtube_stream(self, yt_url):
        import yt_dlp
        tkmsg.showinfo("處理中", "正在解析 YouTube 影片串流，這可能需要幾秒鐘，請按確定後稍候...")
        try:
            ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(yt_url, download=False)
                video_url = info_dict.get('url', info_dict.get('entries', [{}])[0].get('url'))
            
            if video_url:
                self.yolo_start_source(video_url)
            else:
                tkmsg.showerror("錯誤", "無法提取影片的直接播放網址！")
        except Exception as e:
            tkmsg.showerror("錯誤", f"YouTube 解析失敗！請確認網址是否正確或影片是否為公開。\n{e}")

    def yolo_clear_roi(self):
        self.roi_points = []
        tkmsg.showinfo("提示", "ROI 危險區域已清除！\n👉 現在你可以直接在「右側目標影像」上點擊滑鼠，畫出新的危險區域。")

    def yolo_export_csv(self):
        import sqlite3, os
        import pandas as pd
        if not hasattr(self, 'yolo_db_path') or not os.path.exists(self.yolo_db_path):
            tkmsg.showwarning("警告", "目前沒有資料庫紀錄可以匯出！")
            return
        save_path = tkfd.asksaveasfilename(
            title="匯出資料庫為 CSV",
            defaultextension=".csv",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")]
        )
        if save_path:
            try:
                conn = sqlite3.connect(self.yolo_db_path)
                df = pd.read_sql_query("SELECT * FROM events", conn)
                conn.close()
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                tkmsg.showinfo("成功", f"資料已成功匯出至：\n{save_path}")
            except Exception as e:
                tkmsg.showerror("錯誤", f"匯出失敗！\n{e}")

    def on_yolo_mouse_click(self, event):
        if not hasattr(self, 'roi_points'): return
        import numpy as np
        if isinstance(self.roi_points, np.ndarray):
            self.roi_points = self.roi_points.tolist()
        self.roi_points.append([event.x, event.y])
        print(f"📍 新增 ROI 頂點: ({event.x}, {event.y})")

    def yolo_stop(self):
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
            self.video_loop_id = None
            
        if hasattr(self, 'yolo_recording') and self.yolo_recording:
            if hasattr(self, 'yolo_video_writer') and self.yolo_video_writer:
                self.yolo_video_writer.release()
                self.yolo_video_writer = None
            self.btn_yolo_record.config(text="開始錄影", bg="#607D8B")
            self.yolo_recording = False
            
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if hasattr(self, 'lbl_yolo_fps'):
            self.lbl_yolo_fps.config(text="FPS: 0.0")

    def yolo_start_source(self, src):
        import cv2, time
        self.yolo_stop() 
        self.init_yolo()
        if not hasattr(self, 'yolo_db_path'):
            self.init_yolo_db() 
        self.yolo_current_src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            tkmsg.showwarning("警告", f"無法開啟來源: {src}")
            return
        self.yolo_prev_time = time.time()
        self.update_yolo_loop()

    def start_yolo_webcam(self):
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        self.frame_yolo_controls.place(x=632, y=570, width=515, height=75)
        tkmsg.showinfo("啟動", "YOLO 控制面板已展開！\n系統預設為您開啟本機攝影機。")
        self.yolo_start_cam()

    def update_yolo_loop(self):
        import cv2, time
        from PIL import Image, ImageTk
        import numpy as np

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.yolo_stop()
                return

            if self.yolo_current_src == 0:
                frame = cv2.flip(frame, 1)

            # --- A. 來源窗顯示 ---
            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_src = Image.fromarray(cv2image_src)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src

            # --- B. YOLO 物件追蹤與 ROI 偵測 ---
            conf_val = self.yolo_conf_slider.get()
            results = self.yolo_model.track(frame, conf=conf_val, persist=True, tracker="botsort.yaml", device=self.yolo_device, verbose=False)
            annotated = frame.copy()

            if hasattr(self, 'roi_points') and len(self.roi_points) >= 3:
                cv2.polylines(annotated, [np.array(self.roi_points, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(annotated, "Danger Zone", (self.roi_points[0][0], self.roi_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            person_count = 0 
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)
                    en_label = self.yolo_model.names[int(cls_id)]
                    show_label = getattr(self, 'EN_TO_ZH', {}).get(en_label, en_label) if getattr(self, 'yolo_lang_mode', 'zh') == "zh" else en_label

                    if en_label == 'person':
                        person_count += 1

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    roi_hit = False
                    if hasattr(self, 'roi_points') and len(self.roi_points) >= 3:
                        roi_hit = cv2.pointPolygonTest(np.array(self.roi_points, np.int32), (cx, cy), False) >= 0

                    box_color = (0, 0, 255) if roi_hit else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    annotated = self.draw_label_pil(annotated, f"{show_label} ID:{track_id} {conf:.2f}", (x1, y1 - 30))

                    if roi_hit and en_label == 'person':
                        now = time.time()
                        if track_id not in self.yolo_db_cooldown or now - self.yolo_db_cooldown[track_id] > 5:
                            self.insert_yolo_event(show_label, track_id, conf, (x1, y1, x2, y2), True)
                            self.yolo_db_cooldown[track_id] = now
            else:
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        en_label = self.yolo_model.names[cls_id]
                        if en_label == 'person': person_count += 1
                        show_label = getattr(self, 'EN_TO_ZH', {}).get(en_label, en_label) if getattr(self, 'yolo_lang_mode', 'zh') == "zh" else en_label
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        annotated = self.draw_label_pil(annotated, f"{show_label} {conf:.2f}", (x1, y1 - 30))

            # --- 更新 UI ---
            curr_time = time.time()
            fps = 1 / (curr_time - self.yolo_prev_time) if (curr_time - self.yolo_prev_time) > 0 else 0
            self.yolo_prev_time = curr_time
            if hasattr(self, 'lbl_yolo_fps'):
                self.lbl_yolo_fps.config(text=f"FPS: {fps:.1f}")
                self.lbl_yolo_person.config(text=f"人數: {person_count}")

            self.yolo_frame = annotated.copy()
            if hasattr(self, 'yolo_recording') and self.yolo_recording and hasattr(self, 'yolo_video_writer') and self.yolo_video_writer is not None:
                self.yolo_video_writer.write(annotated)

            # --- C. 目標窗顯示 ---
            cv2image_des = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil_des = Image.fromarray(cv2image_des)
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)

            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
                self.label_des_image.bind("<Button-1>", self.on_yolo_mouse_click)
                
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des

            self.video_loop_id = self.root.after(30, self.update_yolo_loop)
    
  # =========================================================
    # === AI 動態手勢辨識 雙螢幕整合引擎 (100% 原版邏輯移植) ===

    def init_gesture_model(self):
        if not hasattr(self, 'gesture_model'):
            self.gesture_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gesture_model = GestureTransformer().to(self.gesture_device)
            
            weight_path = "D:/anaconda/homework/gesture_transformer_m.pth"
            if not os.path.exists(weight_path):
                tkmsg.showerror("錯誤", f"找不到模型權重檔：{weight_path}")
                return False
            
            self.gesture_model.load_state_dict(torch.load(weight_path, map_location=self.gesture_device))
            self.gesture_model.eval()
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.mp_draw = mp.solutions.drawing_utils
            
            # 對應 14 個類別
            self.gesture_classes = [
                "1","2","3","4","5", "6","7","8","9","10",
                "OK", "NG", "EXIT", "MIDDLE_FINGER"
            ]
        return True

    # --- 移植原版的 4 個強大輔助函式 ---
    def gesture_extract(self, hand):
        data = []
        for lm in hand.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return data

    def gesture_split_hands(self, hand_list):
        if len(hand_list) == 1:
            return None, hand_list[0]
        elif len(hand_list) >= 2:
            sorted_hands = sorted(hand_list, key=lambda h: h.landmark[0].x)
            left = sorted_hands[0]
            right = sorted_hands[1]
            return left, right
        return None, None

    def gesture_hand_bbox(self, hand, w, h):
        xs = [lm.x * w for lm in hand.landmark]
        ys = [lm.y * h for lm in hand.landmark]
        margin = 40
        x1 = max(0, int(min(xs) - margin))
        y1 = max(0, int(min(ys) - margin))
        x2 = min(w, int(max(xs) + margin))
        y2 = min(h, int(max(ys) + margin))
        return x1, y1, x2, y2

    def gesture_mosaic(self, frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return frame
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // 15), max(1, h // 15)), interpolation=cv2.INTER_LINEAR)
        blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = blur
        return frame
    # ------------------------------------

    def start_gesture_recognition(self):
        # 隱藏其他按鈕
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()
        
        # 關閉正在執行的影像或鏡頭
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass
        if self.cap is not None:
            self.cap.release()

        # 載入模型
        if not self.init_gesture_model(): return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tkmsg.showwarning("警告", "無法開啟攝影機！")
            return
            
        # 完全依照原版 V4r5_m 的初始變數
        self.seq_len = 30
        self.right_sequence = deque(maxlen=self.seq_len)
        self.exit_buffer = deque(maxlen=10)
        self.left_arm = False
        self.right_gesture = "NONE"
        self.fsm_state = "IDLE"
        self.state_time = time.time()
        self.gesture_prev_time = time.time()
        
        # 啟動 Tkinter 迴圈
        self.update_gesture_loop()

    def update_gesture_loop(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                return

            frame = cv2.flip(frame, 1) # 鏡像翻轉
            
            # === A. 來源窗顯示原始影像 ===
            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_src = Image.fromarray(cv2image_src)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src
            
            # === B. 原版手勢追蹤與特徵提取邏輯 ===
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            annotated = frame.copy()
            
            self.left_arm = False
            right_hand = None
            
            if result.multi_hand_landmarks:
                # 💡 使用原版的聰明拆分法，無視 MediaPipe 的左右手錯誤標籤！
                left_hand, right_hand = self.gesture_split_hands(result.multi_hand_landmarks)
                
                for h in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(annotated, h, self.mp_hands.HAND_CONNECTIONS)
                    
                if left_hand is not None:
                    self.left_arm = True
                    
                if right_hand is not None:
                    self.right_sequence.append(self.gesture_extract(right_hand))
                    
            # === C. 模型推論 ===
            if len(self.right_sequence) == self.seq_len:
                x = torch.tensor([list(self.right_sequence)], dtype=torch.float32).to(self.gesture_device)
                with torch.no_grad():
                    pred = self.gesture_model(x)
                    idx = pred.argmax(dim=1).item()
                    self.right_gesture = self.gesture_classes[idx]
                    
            # === D. EXIT 判斷與 FSM 狀態機 (原版邏輯) ===
            if self.right_gesture == "EXIT":
                self.exit_buffer.append(1)
            else:
                self.exit_buffer.append(0)
                
            exit_score = sum(self.exit_buffer) / len(self.exit_buffer) if len(self.exit_buffer) > 0 else 0
            can_exit = self.left_arm and exit_score > 0.8
            
            if self.fsm_state == "IDLE":
                if can_exit:
                    self.fsm_state = "DETECT"
                    self.state_time = time.time()
            elif self.fsm_state == "DETECT":
                if can_exit and time.time() - self.state_time > 0.5:
                    self.fsm_state = "CONFIRM"
                elif not can_exit:
                    self.fsm_state = "IDLE"
            elif self.fsm_state == "CONFIRM":
                if can_exit:
                    self.fsm_state = "EXIT"
                else:
                    self.fsm_state = "IDLE"
                    
            if self.fsm_state == "EXIT":
                tkmsg.showinfo("系統提示", "偵測到雙手 EXIT 動作，手勢系統關閉！")
                self.cap.release()
                return

            # === E. 計算 FPS ===
            curr_time = time.time()
            fps = 1 / (curr_time - self.gesture_prev_time) if self.gesture_prev_time else 0
            self.gesture_prev_time = curr_time

            # === F. UI 文字顯示 ===
            cv2.putText(annotated, f"Left ARM: {self.left_arm}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"Right: {self.right_gesture}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            cv2.putText(annotated, f"Exit Score: {exit_score:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(annotated, f"FSM: {self.fsm_state}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(annotated, f"FPS: {int(fps)}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # === G. 終極彩蛋：中指自動馬賽克防護 (呼叫原版函式) ===
            if self.right_gesture == "MIDDLE_FINGER" and right_hand is not None:
                h, w = annotated.shape[:2]
                x1, y1, x2, y2 = self.gesture_hand_bbox(right_hand, w, h)
                annotated = self.gesture_mosaic(annotated, x1, y1, x2, y2)
                cv2.putText(annotated, "MOSAIC ACTIVE", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # === H. 目標窗顯示結果影像 ===
            cv2image_des = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil_des = Image.fromarray(cv2image_des)
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)

            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des

            self.video_loop_id = self.root.after(30, self.update_gesture_loop)
            
    # =========================================================
    # === Lab 15: DeepFace 即時人種/性別/年齡/情緒分析 ===
    def start_deepface_analysis(self):
        # 1. 隱藏不必要的按鈕，保持介面乾淨
        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()

        # 2. 安全關閉之前的視訊迴圈
        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except ValueError: pass

        import threading
        import time
        from deepface import DeepFace
        from ultralytics import YOLO

        # 3. 載入 YOLOv8 人臉偵測模型 (用來快速抓出臉部位置)
        if not hasattr(self, 'yolo_face_model'):
            try:
                # 確保 yolov8n-face.pt 放在這絕對路徑下
                self.yolo_face_model = YOLO("D:/anaconda/homework/yolov8n-face.pt") 
            except Exception as e:
                tkmsg.showerror("錯誤", f"載入 yolov8n-face.pt 失敗！\n{e}")
                return

        # 4. 初始化快取 (記錄每張臉的 DeepFace 結果與背景運算狀態)
        self.deepface_cache = {}  
        self.analyzing_threads = {} 

        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tkmsg.showwarning("警告", "無法開啟攝影機！")
            return

        # --- 獨立執行緒：讓 DeepFace 在背景慢慢算，不卡死主畫面 ---
        def analyze_face(track_id, face_img):
            try:
                # enforce_detection=False 避免 AI 偶爾沒抓到臉時崩潰
                results = DeepFace.analyze(
                    face_img, 
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False,
                    silent=True
                )
                res = results[0]
                
                # 整理輸出的文字格式 (例如：M, 22 | happy | asian)
                gender = "M" if res['dominant_gender'] == "Man" else "F"
                emotion = res['dominant_emotion']
                race = res['dominant_race']
                age = res['age']
                
                # 算完後，把結果寫入快取記憶體
                self.deepface_cache[track_id] = {
                    "text": f"{gender},{age} | {emotion} | {race}",
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"DeepFace 分析略過: {e}")
            finally:
                # 分析完畢，解除這張臉的鎖定狀態
                if track_id in self.analyzing_threads:
                    del self.analyzing_threads[track_id]

        # --- 視訊主迴圈 ---
        def update_deepface_loop():
            if self.cap is None or not self.cap.isOpened(): return
            ret, frame = self.cap.read()
            if not ret:
                self.video_loop_id = self.root.after(30, update_deepface_loop)
                return
            
            frame = cv2.flip(frame, 1)

            # === A. 左側來源窗：顯示乾淨的原畫面 ===
            cv2image_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_src = Image.fromarray(cv2image_src)
            tk_image_src = ImageTk.PhotoImage(image=img_pil_src)
            if self.label_scr_image is None:
                self.label_scr_image = tk.Label(self.frame_scr, image=tk_image_src)
                self.label_scr_image.pack()
            self.label_scr_image.configure(image=tk_image_src)
            self.label_scr_image.image = tk_image_src

            # === B. 影像分析與畫框 ===
            annotated = frame.copy()
            
            # 使用 YOLO 進行人臉追蹤 (botsort)
            results = self.yolo_face_model.track(annotated, persist=True, verbose=False, tracker="botsort.yaml")
            
            active_ids = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    active_ids.append(track_id)

                    # 畫上 YOLO 給的藍橘色人臉追蹤框
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 150, 0), 2)

                    # 💡 如果這張臉是新的 (沒分析過且沒人在算)，就派一個背景執行緒去算
                    if track_id not in self.deepface_cache and track_id not in self.analyzing_threads:
                        # 稍微擴大截圖範圍給 DeepFace，提高年齡情緒準確率
                        h, w = frame.shape[:2]
                        face_crop = frame[max(0, y1-20):min(h, y2+20), 
                                          max(0, x1-20):min(w, x2+20)]
                        if face_crop.size > 0:
                            self.analyzing_threads[track_id] = True
                            threading.Thread(target=analyze_face, args=(track_id, face_crop.copy()), daemon=True).start()
                    
                    # 💡 如果這張臉已經算完了，就把快取裡的黃色字體印在頭上
                    if track_id in self.deepface_cache:
                        text = self.deepface_cache[track_id]["text"]
                        
                        # 畫黑色半透明底框讓字體看起來更清楚
                        cv2.rectangle(annotated, (x1, y1 - 30), (x1 + len(text)*8, y1), (30, 30, 30), -1)
                        cv2.putText(annotated, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        
                        # 更新這張臉的存活時間
                        self.deepface_cache[track_id]["timestamp"] = time.time()

            # 💡 清理過期快取 (如果這張臉離開鏡頭超過 3 秒，就刪除他的分析記憶)
            expired_ids = [tid for tid in list(self.deepface_cache.keys()) if tid not in active_ids]
            for tid in expired_ids:
                if time.time() - self.deepface_cache[tid]["timestamp"] > 3.0:
                    del self.deepface_cache[tid]

            # === C. 右側目標窗：顯示帶有框框與文字的結果影像 ===
            cv2image_des = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil_des = Image.fromarray(cv2image_des)
            tk_image_des = ImageTk.PhotoImage(image=img_pil_des)

            if hasattr(self.frame_des, 'canvas') and self.frame_des.canvas != None:
                self.frame_des.canvas.get_tk_widget().pack_forget()
                self.frame_des.canvas = None

            if self.label_des_image is None:
                self.label_des_image = tk.Label(self.frame_des, image=tk_image_des)
                self.label_des_image.pack()
            self.label_des_image.configure(image=tk_image_des)
            self.label_des_image.image = tk_image_des

            self.video_loop_id = self.root.after(30, update_deepface_loop)

        # 啟動迴圈
        update_deepface_loop()
        
   # =========================================================
    # === 白板專題：Free-Touch Panel 隔空手勢影音播放器 ===
    # =========================================================
    def start_free_touch_player(self):
        import tkinter.filedialog as tkfd
        import mediapipe as mp
        import time

        video_path = tkfd.askopenfilename(
            initialdir='D:/anaconda/images',
            title='🎥 專題展示：請先選擇要導入的影片',
            filetypes=[('影片檔', '*.mp4 *.avi *.mkv'), ('所有檔案', '*')]
        )
        if not video_path: return

        if hasattr(self, 'video_loop_id') and self.video_loop_id:
            try: self.root.after_cancel(self.video_loop_id)
            except: pass

        self.btn_prev.place_forget()
        self.btn_next.place_forget()
        self.btn_snapshot.place_forget()
        if hasattr(self, 'frame_yolo_controls'):
            self.frame_yolo_controls.place_forget()
            
        if not hasattr(self, 'MP03_root') or not self.MP03_root.winfo_exists():
            self.MP03_root = tk.Toplevel(self.root)
            self.MP03_root.title("🖐️ Free-Touch Gesture Media Player")
            self.MP03_root.geometry("1024x600")
            self.vlc_app = MediaPlayerApp(self.MP03_root)
            self.MP03_root.protocol("WM_DELETE_WINDOW", self.close_mp03)
            
        self.vlc_app.current_file = video_path
        self.vlc_app.play_video()
        self.vlc_app.update_video_progress()
            
        if self.cap is not None: self.cap.release()
        
        # 💡 1. 移除容易造成筆電黑屏死機的 cv2.CAP_DSHOW
        self.cap = cv2.VideoCapture(0) 
        
        # 💡 2. 加入防呆錯誤提示，避免鏡頭打不開時無聲無息
        if not self.cap.isOpened():
            tkmsg.showerror("錯誤", "無法開啟攝影機！請確認是否有其他軟體(如 Zoom, Line, 或其他 Python 視窗)佔用鏡頭！")
            return
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_cooldown = time.time() 
        
        print("✅ 本機影片啟動，鏡頭開啟中...")
        self.update_free_touch_loop()

    def start_free_touch_youtube(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("📺 導入 YouTube 影片")
        dialog.geometry("500x150")
        
        tk.Label(dialog, text="請輸入 YouTube 影片網址:", font=("Arial", 12, "bold")).pack(pady=10)
        url_entry = tk.Entry(dialog, width=55, font=("Arial", 11))
        url_entry.pack(pady=5)
        
        def confirm():
            yt_url = url_entry.get().strip()
            if not yt_url:
                tkmsg.showwarning("警告", "網址不能為空！")
                return
            dialog.destroy()
            self._process_and_start_youtube_touch(yt_url)
            
        tk.Button(dialog, text="確定並啟動", command=confirm, bg="#E53935", fg="white", font=("Arial", 10, "bold")).pack(pady=10)

    def _process_and_start_youtube_touch(self, yt_url):
        import yt_dlp
        import mediapipe as mp
        import time
        
        tkmsg.showinfo("處理中", "正在解析 YouTube 影片串流，這可能需要幾秒鐘...\n(請按「確定」後稍候)")
        
        try:
            ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(yt_url, download=False)
                video_url = info_dict.get('url', info_dict.get('entries', [{}])[0].get('url'))
                
            if not video_url:
                tkmsg.showerror("錯誤", "無法提取 YouTube 的直接播放網址！")
                return
                
            if hasattr(self, 'video_loop_id') and self.video_loop_id:
                try: self.root.after_cancel(self.video_loop_id)
                except: pass
                
            self.btn_prev.place_forget()
            self.btn_next.place_forget()
            self.btn_snapshot.place_forget()
            if hasattr(self, 'frame_yolo_controls'):
                self.frame_yolo_controls.place_forget()
                
            if not hasattr(self, 'MP03_root') or not self.MP03_root.winfo_exists():
                self.MP03_root = tk.Toplevel(self.root)
                self.MP03_root.title("🖐️ Free-Touch Media Player (YouTube Mode)")
                self.MP03_root.geometry("1024x600")
                self.vlc_app = MediaPlayerApp(self.MP03_root)
                self.MP03_root.protocol("WM_DELETE_WINDOW", self.close_mp03)
                
            self.vlc_app.current_file = video_url
            self.vlc_app.play_video()
            self.vlc_app.update_video_progress()
                
            if self.cap is not None: self.cap.release()
            
            # 💡 1. 移除 cv2.CAP_DSHOW
            self.cap = cv2.VideoCapture(0)
            
            # 💡 2. 防呆檢查
            if not self.cap.isOpened():
                tkmsg.showerror("錯誤", "無法開啟攝影機！請確認鏡頭沒有被佔用。")
                return
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
            self.mp_draw = mp.solutions.drawing_utils
            self.gesture_cooldown = time.time()
            
            print("✅ YouTube 串流已成功連接！鏡頭啟動中...")
            self.update_free_touch_loop()
            
        except Exception as e:
            tkmsg.showerror("錯誤", f"YouTube 解析失敗！請確認網址是否正確。\n{e}")

    def update_free_touch_loop(self):
        import time
        from PIL import Image, ImageTk
        import cv2
        
        if self.cap is None or not self.cap.isOpened(): 
            print("⚠️ 攝影機未就緒，迴圈終止！")
            return
        
        ret, frame = self.cap.read()
        if not ret: 
            self.video_loop_id = self.root.after(30, self.update_free_touch_loop)
            return
            
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        annotated = frame.copy()
        
        detected_gestures = []
        
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                self.mp_draw.draw_landmarks(annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                label = handedness.classification[0].label 
                tips = [4, 8, 12, 16, 20] 
                pips = [2, 6, 10, 14, 18] 
                
                fingers_up = []
                if label == 'Right': 
                    fingers_up.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x else 0)
                else:
                    fingers_up.append(1 if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[pips[0]].x else 0)
                
                for i in range(1, 5):
                    fingers_up.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y else 0)
                    
                up_count = sum(fingers_up)
                ges = "UNKNOWN"
                
                if up_count == 0: ges = "FIST"             
                elif up_count == 5: ges = "PALM"           
                elif fingers_up == [0, 1, 1, 0, 0]: ges = "V_SIGN" 
                elif fingers_up[1:] == [0, 0, 0, 0]:       
                    thumb_tip_x = hand_landmarks.landmark[tips[0]].x
                    thumb_base_x = hand_landmarks.landmark[1].x
                    if thumb_tip_x < thumb_base_x - 0.05: ges = "THUMB_LEFT"   
                    elif thumb_tip_x > thumb_base_x + 0.05: ges = "THUMB_RIGHT" 
                    
                detected_gestures.append(ges)
                
                h, w, c = annotated.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(annotated, ges, (cx, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        current_time = time.time()
        if current_time - self.gesture_cooldown > 1.0: 
            if detected_gestures.count("FIST") == 2:
                if hasattr(self, 'vlc_app') and self.vlc_app: self.vlc_app.stop()
                if hasattr(self, 'MP03_root') and self.MP03_root.winfo_exists(): self.MP03_root.destroy()
                self.cap.release()
                cv2.putText(annotated, "EXITING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                cv2image_src = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                tk_image_src = ImageTk.PhotoImage(image=Image.fromarray(cv2image_src))
                if self.label_scr_image is None:
                    self.label_scr_image = tk.Label(self.frame_scr)
                self.label_scr_image.configure(image=tk_image_src)
                self.label_scr_image.image = tk_image_src
                self.label_scr_image.pack() # 💡 強制顯示
                return 
                
            elif len(detected_gestures) > 0 and hasattr(self, 'vlc_app') and self.vlc_app.playing_video:
                main_ges = detected_gestures[0]
                action_text = ""
                
                if main_ges == "FIST":           
                    self.vlc_app.stop()
                    action_text = "STOPPED!"
                elif main_ges == "PALM":         
                    if not self.vlc_app.video_paused:
                        self.vlc_app.pause_video()
                        action_text = "PAUSED!"
                elif main_ges == "V_SIGN":       
                    if self.vlc_app.video_paused:
                        self.vlc_app.pause_video()
                        action_text = "PLAYING!"
                elif main_ges == "THUMB_LEFT":   
                    self.vlc_app.rewind()
                    action_text = "<< REWIND"
                elif main_ges == "THUMB_RIGHT":  
                    self.vlc_app.fast_forward()
                    action_text = "FORWARD >>"

                if action_text:
                    cv2.putText(annotated, action_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    self.gesture_cooldown = current_time 
                    
        cv2image_src = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        tk_image_src = ImageTk.PhotoImage(image=Image.fromarray(cv2image_src))
        
        # 💡 3. 強制防呆顯示邏輯 (不管元件有沒有被隱藏，強制把它抓出來顯示)
        if self.label_scr_image is None:
            self.label_scr_image = tk.Label(self.frame_scr)
        self.label_scr_image.configure(image=tk_image_src)
        self.label_scr_image.image = tk_image_src
        self.label_scr_image.pack() 
        
        self.video_loop_id = self.root.after(30, self.update_free_touch_loop)
            
    def help_copyright(self):
        tk.messagebox.showinfo(title='版權', message='電子三乙 U1222139 馮志鋒 DIP學習平台')
    def help_about(self):
        tk.messagebox.showinfo(title='關於', message='數位影像處理-2026')
         
if __name__ == '__main__':        
    Image_sys()
     
'''
tkinter.messagebox中有如下函數：
askokcancel(title=None, message=None, **options)
Ask if operation should proceed; return true if the ans教學展示用wer is ok
askquestion(title=None, message=None, **options)
Ask a question
askretrycancel(title=None, message=None, **options)
Ask if operation should be retried; return true if the answer is yes
askyesno(title=None, message=None, **options)
Ask a question; return true if the answer is yes
askyesnocancel(title=None, message=None, **options)
Ask a question; return true if the answer is yes, None if cancelled.
showerror(title=None, message=None, **options)
Show an error message
showinfo(title=None, message=None, **options)
Show an info message
showwarning(title=None, message=None, **options)
Show a warning message
'''   