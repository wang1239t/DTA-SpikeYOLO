# DTA-SpikeYOLO: A Bio-inspired SNN with Dynamic Temporal Adaptation Sampling for Event-based UAV Detection
***
![DTA-SpikeYOLO](https://github.com/wang1239t/myimg/blob/main/DTA-SpikeYOLO.png)
## Installation
### Conda
Our environment runs on CUDA 11.8, one RTX 3090 GPUs and Ubuntu 22.04
```
conda create -y -n dtaspikeyolo python=3.9
conda activate dtaspikeyolo
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Venv
Install the dependencies for the environment. The rest of the environment configuration is the same as [SpikeYOLO](https://github.com/BICLab/SpikeYOLO).
```
pip install -r requirements.txt
```
## Date preparation
We used the [EMUAV Dataset](https://pan.baidu.com/share/init?surl=G30vw2NO3WvHO5Q2EViI6g) dataset.
The directory structure after processing is as follows:
```
EMUAV
├── train/
│  ├── images/
│  │  ├── normal/
│  │  │  ├── 0001_normal/
│  │  │  │  ├── 0001_normal_001.png
│  │  │  │  ├── 0001_normal_002.png
│  │  │  │  ├── ...
│  │  │  │  └── 0001_normal_100.png
│  │  │  ├── 0002_normal/
│  │  │  ├── ...
│  │  │  └── 0076_normal/
│  │  ├── low_light/
│  │  └── motion_blur/
│  ├── events/
│  │  ├── normal/
│  │  ├── low_light/
│  │  └── motion_blur/
│  └── labels/
│     ├── normal/
│     ├── low_light/
│     └── motion_blur/
└── val/
```
## Training
Select the UAV.yaml dataset configuration file and the snn_yolov8.yaml model configuration file
```
python train.py 
```

## Test
Test with a pre-trained model can be used to calculate the average spike firing rate of the DTA-SNN module and the overall spike firing rate of the model.
```
python test.py 
```

## Code Acknowledgments
We used the code from the following project. <br>
[SpikeYOLO](https://github.com/BICLab/SpikeYOLO) for network model framework. <br>
[EAS-SNN](https://github.com/Windere/EAS-SNN) for event adaptive sampling method.
