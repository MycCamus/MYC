import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel,QVBoxLayout,QHBoxLayout,QPushButton,QLineEdit,QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import cv2
from ultralytics import YOLO


class YOLORealTimeUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8实时检测系统")
        self.setGeometry(100,100,800,600)

        # 初始化YOLO模型
        self.model = YOLO("yolov8s.pt")
        self.model.eval()

        # 创建界面布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 视频显示区域
        self.video_label = QLabel("实时视频流")
        self.layout.addWidget(self.video_label)

        # 结果显示区域
        self.result_panel = QWidget()
        self.result_layout = QHBoxLayout(self.result_panel)
        self.result_image_label = QLabel("检测结果")
        self.result_text_label = QLabel("检测统计：")
        self.result_layout.addWidget(self.result_image_label)
        self.result_layout.addWidget(self.result_text_label)
        self.layout.addWidget(self.result_panel)

        # 控制按钮
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.camera_select = QLineEdit("0")
        self.control_layout.addWidget(self.start_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.control_layout.addWidget(self.camera_select)
        self.layout.addWidget(self.control_panel)

        # 信号与槽连接
        self.start_btn.clicked.connect(self.start_detect)
        self.stop_btn.clicked.connect(self.stop_detect)

        # 初始化摄像头
        self.cap = None
        self.timer = QTimer()
        self.detection_result = []

        # 设置默认参数
        self.conf_threshold = 0.5  # 置信度阈值
        self.iou_threshold = 0.45  # NMS IOU阈值

    def start_detect(self):
        if not self.cap:
            self.open_camera(int(self.camera_select.text()))
            if not self.cap.isOpened():
                return

        self.timer.start(300)  # 每300ms检测一次
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detect(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cap.release()
        self.cap = None

    def open_camera(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.video_label.setText("打开摄像头失败！")
            return

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detect()
            return

        # 模型推理
        results = self.model(frame, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)
        self.detection_result = results[0].boxes  # 保存检测结果

        # 绘制检测框和标签
        output_frame = frame.copy()
        for box in self.detection_result:
            x1, y1, x2, y2 = map(int,box.xyxy[0])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output_frame, f"{box.cls:.1f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示视频流
        self.update_video_label(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

        # 显示检测统计
        if self.detection_result:
            self.result_text_label.setText(f"检测到 {len(self.detection_result)} 个目标")
        else:
            self.result_text_label.setText("未检测到目标")

    def update_video_label(self, frame):
        # 将OpenCV图像转换为QImage
        h, w, ch = frame.shape
        bytes_per_line = w * ch
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.stop_detect()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = YOLORealTimeUI()
    ui.show()
    sys.exit(app.exec())


