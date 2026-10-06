數位影像處理與 AI 視覺整合平台 (DIP-2026)

這是一個基於 Python 與 Tkinter 建立的綜合型數位影像處理 (Digital Image Processing) 專題應用程式。本專案不僅包含了基礎的影像處理、濾波、形態學操作，還整合了多種先進的 AI 視覺辨識技術，如 YOLOv8 物件偵測、Dlib/ArcFace 人臉辨識、DeepFace 屬性分析，以及基於 MediaPipe 的自訂手勢辨識與多媒體控制系統。

作者： 馮志鋒 

🌟 主要功能特色

本系統提供直覺的雙螢幕 GUI 介面（來源影像與目標影像比對），功能涵蓋以下核心模組：

1. 基礎影像處理與強化

影像強化： 負片、直方圖等化、對數轉換、Gamma 轉換、Beta 轉換。

幾何操作： 水平/垂直翻轉、縮放 (金字塔、Resize)、旋轉 (平移、仿射、透射)。

雜訊添加： 高斯、區域變異數、波義森、鹽、胡椒、胡椒鹽、史培克雜訊。

2. 濾波器與邊緣偵測

平滑模糊： 均值、方框、高斯、中值、雙邊濾波 (支援多種 Kernel 大小)。

邊緣銳化： Sobel, Prewitt, FreiChen, Scharr, Laplacian, Canny, EdgeSharp。

形態學： 腐蝕、膨脹、開/閉運算、形態學梯度、頂帽/黑帽、取骨架、細線化。

3. 進階電腦視覺與 AI 辨識

色彩檢測： RYGB 精準去背與即時顏色球 (瓶蓋) 追蹤。

全景拼接： 支援 ORB, SIFT, BRISK 特徵點的多張影像無痕拼接 (Panorama)。

人臉辨識系統：

Haar 級聯五官偵測。

Dlib (128D) 與 ArcFace (InsightFace 512D) 臉部特徵提取與即時辨識。

整合 SQLite 進行即時人臉打卡記錄。

DeepFace 即時人種、性別、年齡、情緒分析。

YOLOv8 企業級物件偵測：

支援 Webcam, 影片檔, RTSP, YouTube 串流。

提供 ROI (危險區域) 繪製、入侵警報記錄與 CSV 匯出功能。

AI 動態手勢辨識：

基於 MediaPipe 與自訂 Transformer 模型 (14 種手勢)。

隔空手勢影音播放器 (Free-Touch Player)： 透過手勢控制 VLC 播放器 (支援本機影片與 YouTube 串流)，手勢包含暫停、播放、快進、倒退等。

🛠️ 環境需求與依賴套件

請確保您的電腦已安裝 Python 3.8+ 以及 VLC Media Player 主程式。
接著透過 pip 安裝以下所需的 Python 套件：

pip install opencv-python Pillow numpy scipy scikit-image matplotlib pandas
pip install torch torchvision
pip install ultralytics mediapipe dlib deepface insightface scikit-learn
pip install python-vlc yt-dlp


📂 資料夾結構與模型準備 (重要)

⚠️️ 注意： 本程式內部使用了「絕對路徑」，請務必在您的 D槽 建立以下資料夾結構，並將對應的預訓練模型檔案放入指定位置，否則程式將無法正常運行。

D:/
└── anaconda/
    ├── images/                          # 預設圖片、影片存檔與讀取位置
    │   ├── face.xml                     # Haar 人臉特徵檔
    │   ├── eye.xml                      # Haar 眼睛特徵檔
    │   ├── nose.xml                     # Haar 鼻子特徵檔
    │   ├── mouth.xml                    # Haar 嘴巴特徵檔
    │   └── lena_std_512.jpg             # 預設測試圖片
    │
    └── homework/
        ├── gesture_transformer_m.pth    # 自訂手勢辨識模型權重檔
        ├── yolov8n.pt                   # YOLOv8 預設物件模型
        ├── yolov8n-face.pt              # YOLOv8 人臉偵測模型
        │
        └── data/
            ├── face_log.db              # (自動生成) 人臉打卡資料庫
            ├── yolo_detection.db        # (自動生成) YOLO 警報資料庫
            ├── features_all.csv         # (自動生成) Dlib 特徵資料庫
            ├── arcface_features.csv     # (自動生成) ArcFace 特徵資料庫
            ├── data_faces_from_camera/  # (自動生成) 人臉註冊照片存放區
            │
            └── data_dlib/
                ├── shape_predictor_68_face_landmarks.dat        # Dlib 68特徵點模型
                └── dlib_face_recognition_resnet_model_v1.dat    # Dlib 人臉辨識模型


🚀 執行方式

確認所有套件與檔案路徑都設定完畢後，直接執行 Python 檔案即可啟動 GUI 介面：

python DIP-2026-Beta.py


📝 注意事項

攝影機占用問題： 若使用 YOLO 或手勢辨識等需要開啟 Webcam 的功能，請確保其他軟體（如 Zoom, LINE）沒有佔用您的攝影機。

中文字體顯示： YOLO 與部分繪圖功能依賴 Windows 內建的微軟正黑體 (msjh.ttc)。

YouTube 串流： 解析 YouTube 影片需依賴 yt-dlp，若遇到無法播放的情況，請嘗試更新該套件 (pip install --upgrade yt-dlp)。
