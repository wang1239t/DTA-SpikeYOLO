import os

from ultralytics import YOLO

os.environ['WANDB_DISABLED'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 屏蔽其他GPU
fr_dict = {}

model = YOLO("snn_yolov8s.yaml")

model.train(data="UAV.yaml", device=[4], epochs=72)

# 直接加载已保存的权重文件（包含模型结构+参数）
# model = YOLO("/home/wanghechong/mapProject/Spikeyolo-220/runs/detect/train69/weights/last.pt")
# #
# # 继续训练（自动继承原训练参数，epochs=100会覆盖原设置）
# model.train(data="UAV.yaml", device=[4], resume=True)

# 测试模型
