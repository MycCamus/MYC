import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

model_yaml_path = 'ultralytics/cfg/models/v8/yolov8n-CBAM.yaml'

data_yaml_path = 'ultralytics/cfg/datasets/safety_helmet.yaml'

pre_model_name = 'yolov8s.pt'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/v8/yolov8n-CBAM.yaml').load(pre_model_name)
    results = model.train(data = data_yaml_path,
                          epochs = 50,
                          batch = 16,
                          workers = 4,
                          device = 0,
                          close_mosaic = 10,
                          project = 'runs/train',
                          name='train_v8OCBAMTEST',
                          )
