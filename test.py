import os
from ultralytics import YOLO
import torch.nn as nn
import torch
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode

from ultralytics.nn.modules import DTASNNEmbedding
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 屏蔽其他GPU
model = YOLO("/home/wanghechong/mapProject/Spikeyolo-220/runs/detect/train86/weights/best.pt")

fr_dict = {}  # 存储LIF模块的发放率
adaptive_fr_per_step_dict = {}  # 存储AdaptiveRSNNEmbedding每个时间步的发放率
iter = 162


# 修改后的钩子函数
def forward_hook_fn(module, input, output):
    global i
    if module.name == 'model.model.2.Conv.lif1':
        i = i + 1
        print("i:", i)

    # 统计LIF模块（保持原有逻辑）
    if isinstance(module, MultiStepLIFNode):
        if module.name not in fr_dict:
            fr_dict[module.name] = output.detach().mean() / iter
        else:
            fr_dict[module.name] += output.detach().mean() / iter

    # 统计DTASNNEmbedding模块（按时间步）
    elif isinstance(module, DTASNNEmbedding):
        if module.name not in adaptive_fr_per_step_dict:
            # 初始化每个时间步的累加器
            adaptive_fr_per_step_dict[module.name] = {
                'step0': 0.0,
                'step1': 0.0,
                'step2': 0.0,
                'step3': 0.0,
                'step4': 0.0,
                'count': 0
            }

        # 获取当前模块的时间步发放率
        fr_per_step = module.get_fr_per_step()

        # 累加到全局统计（按时间步）
        adaptive_fr_per_step_dict[module.name]['step0'] += fr_per_step[0]
        adaptive_fr_per_step_dict[module.name]['step1'] += fr_per_step[1]
        adaptive_fr_per_step_dict[module.name]['step2'] += fr_per_step[2]
        adaptive_fr_per_step_dict[module.name]['step3'] += fr_per_step[3]
        adaptive_fr_per_step_dict[module.name]['step4'] += fr_per_step[4]
        adaptive_fr_per_step_dict[module.name]['count'] += 1


# 注册钩子
for n, m in model.named_modules():
    if isinstance(m, (MultiStepLIFNode, DTASNNEmbedding)):
        print("Registering hook for:", n)
        m.name = n
        m.register_forward_hook(forward_hook_fn)

        # 重置发放率统计
        if isinstance(m, DTASNNEmbedding):
            m.reset_fr()


# 验证前重置所有模块的统计
def reset_all_fr():
    for m in model.modules():
        if isinstance(m, DTASNNEmbedding):
            m.reset_fr()


# 运行验证
reset_all_fr()
model.val(data="UAV.yaml", device=[3])

# 打印结果
print("LIF Firing Rates:", fr_dict)
print("\nAdaptiveRSNN Firing Rates per Step:")
for module_name, stats in adaptive_fr_per_step_dict.items():
    count = max(1, stats['count'])
    print(f"Module: {module_name}")
    print(f"  Step0: {stats['step0'] / count:.4f}")
    print(f"  Step1: {stats['step1'] / count:.4f}")
    print(f"  Step2: {stats['step2'] / count:.4f}")
    print(f"  Step3: {stats['step3'] / count:.4f}")
    print(f"  Step4: {stats['step4'] / count:.4f}")
