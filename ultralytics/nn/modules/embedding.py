import copy
import os
import uuid

import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F

from .layer import tdLayer, LIFLayer, LIFCell
from .activation import *


# class SpikeBalanceLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=0.1, gamma=0.2):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         # 使用平滑L1损失代替MSE，更稳定
#         self.loss_fn = nn.SmoothL1Loss(reduction='mean')
#
#     def forward(self, first_spike_vmem, first_spike_thresh, last_vmem, last_thresh, late_spike_count):
#         """
#         修改：移除所有.float()转换，保持原始数据类型
#         """
#         losses = []
#         device = first_spike_vmem.device if first_spike_vmem.numel() > 0 else last_vmem.device
#
#         # 首次发放约束
#         if first_spike_vmem.numel() > 0:
#             # 确保阈值和膜电位在同一设备
#             first_spike_thresh = first_spike_thresh.to(device)
#             spike_loss = self.loss_fn(first_spike_vmem, first_spike_thresh)
#             losses.append(self.alpha * spike_loss)
#
#         # 未发放约束
#         if last_vmem.numel() > 0:
#             target_nospike = 0.8 * last_thresh.to(device)
#             nospike_loss = self.loss_fn(last_vmem, target_nospike)
#             losses.append(self.beta * nospike_loss)
#
#         # 后期发放率约束
#         if late_spike_count.numel() > 0:
#             target = 0.3 * torch.ones_like(late_spike_count)
#             late_loss = self.loss_fn(late_spike_count / late_spike_count.max(), target)
#             losses.append(self.gamma * late_loss)
#
#         return sum(losses) if losses else torch.tensor(0.0, device=device, requires_grad=True)


# class DecayingSpikeBalanceLoss(nn.Module):
#     def __init__(self, nb_steps, early_ratio=0.3, late_ratio=0.7, min_late_rate=0.15):
#         """
#         针对硬复位机制设计的脉冲平衡损失
#         :param nb_steps: 总时间步数
#         :param early_ratio: 早期时间步比例（0-1）
#         :param late_ratio: 后期时间步比例（0-1）
#         :param min_late_rate: 后期最小发放率（占早期平均发放率的比例）
#         """
#         super().__init__()
#         self.nb_steps = nb_steps
#         self.early_ratio = early_ratio
#         self.late_ratio = late_ratio
#         self.min_late_rate = min_late_rate
#
#         # 创建理想递减曲线（指数衰减）
#         self.register_buffer('ideal_decay', torch.tensor(
#             [0.9 * (0.85 ** t) for t in range(nb_steps)]
#         ))
#
#     def forward(self, spike_rates, membrane_stats, thresholds):
#         """
#         :param spike_rates: 各时间步脉冲发放率 [T]
#         :param membrane_stats: 各时间步膜电位统计量 [T, bins]
#         :param thresholds: 各时间步阈值统计量 [T, C]
#         """
#         # 1. 脉冲递减趋势损失
#         # 计算实际发放率与理想递减曲线的差异
#         trend_diff = spike_rates - self.ideal_decay[:len(spike_rates)]
#         trend_loss = F.smooth_l1_loss(trend_diff, torch.zeros_like(trend_diff))
#
#         # 2. 后期脉冲保障损失
#         early_end = int(self.nb_steps * self.early_ratio)
#         late_start = int(self.nb_steps * self.late_ratio)
#
#         early_avg = spike_rates[:early_end].mean()
#         late_avg = spike_rates[late_start:].mean() if late_start < self.nb_steps else 0.0
#
#         # 后期平均发放率应不低于早期平均发放率的min_late_rate
#         late_loss = F.relu(self.min_late_rate * early_avg - late_avg)
#
#         # 3. 膜电位利用率损失（确保信息被有效捕获）
#         # 计算膜电位平均值与阈值的比例
#         membrane_avg = membrane_stats.mean(dim=0).mean()
#         threshold_avg = thresholds.mean()
#         utilization_ratio = membrane_avg / threshold_avg.clamp(min=1e-6)
#
#         # 目标利用率：后期时间步应达到80%
#         utilization_loss = F.relu(0.8 - utilization_ratio)
#
#         # 4. 递减稳定性损失（防止后期骤降）
#         if self.nb_steps > 3:
#             # 计算最后三个时间步的平均变化率
#             last_rates = spike_rates[-3:]
#             decay_rates = last_rates[:-1] - last_rates[1:]
#             avg_decay = decay_rates.mean()
#
#             # 允许衰减但不能加速衰减
#             stability_loss = F.relu(avg_decay - decay_rates[-1])
#         else:
#             stability_loss = torch.tensor(0.0)
#
#         # 组合损失（加权求和）
#         total_loss = (
#                 0.4 * trend_loss +
#                 0.3 * late_loss +
#                 0.2 * utilization_loss +
#                 0.1 * stability_loss
#         )
#
#         return total_loss

# class SimplifiedSpikeBalanceLoss(nn.Module):
#     def __init__(self, nb_steps, early_steps=1, target_late_rate=0.15):
#         """
#         简化的脉冲平衡损失
#         :param nb_steps: 总时间步数
#         :param early_steps: 定义为“早期”的时间步数（例如前1或2步）
#         :param target_late_rate: 后期时间步期望的平均发放率（绝对目标值，例如0.05代表5%）
#         """
#         super().__init__()
#         self.nb_steps = nb_steps
#         self.early_steps = early_steps
#         self.target_late_rate = target_late_rate
#
#     def __call__(self, spike_rates):
#         # 确保输入为FP32精度
#         spike_rates = spike_rates.float() if spike_rates.dtype != torch.float32 else spike_rates
#
#         # 确保所有计算在FP32精度下进行
#         early_avg = spike_rates[:self.early_steps].mean()
#         early_loss = F.mse_loss(early_avg, torch.tensor(0.7, device=spike_rates.device, dtype=torch.float32))
#
#         late_avg = spike_rates[self.early_steps:].mean()
#         late_loss = F.smooth_l1_loss(
#             late_avg,
#             torch.tensor(self.target_late_rate, device=spike_rates.device, dtype=torch.float32)
#         )
#
#         total_loss = early_loss + late_loss
#         return total_loss  # 确保返回FP32精度
#
# class AdaptiveMembraneLoss:
#     def __init__(self, time_balance_weight=0.2, membrane_util_weight=0.1):
#         """
#         简化的膜电位损失函数
#         :param target_spike_rate: 目标脉冲发放率 (0-1之间)
#         :param time_balance_weight: 时间平衡权重
#         :param membrane_util_weight: 膜电位利用率权重
#         """
#         super().__init__()
#         self.time_balance_weight = time_balance_weight
#         self.membrane_util_weight = membrane_util_weight
#
#     def __call__(self, spike_history, membrane_history, current_history, thresh_history):
#         """
#         :param spike_history: 各时间步脉冲列表 [T, B, C, H, W]
#         :param membrane_history: 各时间步膜电位列表 [T, B, C, H, W]
#         """
#         loss = 0.0
#         T, B, _, _, _ = spike_history.shape
#
#         # 1. 时间步平衡损失 - 鼓励各时间步脉冲率相对均衡
#         spike_rates = torch.mean(spike_history.float(), dim=[2, 3, 4])  # [T, B]
#         time_balance_loss = torch.var(spike_rates, dim=0).mean()  # 计算时间维度上的方差
#
#         loss += self.time_balance_weight * time_balance_loss
#
#         # 2. 膜电位利用率损失 - 鼓励有效利用膜电位
#         # 计算未发放脉冲的神经元的膜电位与阈值的平均比率
#         non_spiking_mask = (spike_history == 0).float()
#         membrane_ratio = (membrane_history * non_spiking_mask) / (thresh_history+ 1e-7)  # 假设阈值为1.0
#         membrane_util_loss = -torch.mean(membrane_ratio)  # 负号表示鼓励更高的利用率
#
#         loss += self.membrane_util_weight * membrane_util_loss
#
#         return loss


# class MembraneLoss:
#     def __init__(self,
#                  target_min_rate=0.04,  # 进一步降低最小脉冲率
#                  target_max_rate=0.3,  # 提高最大脉冲率
#                  decay_weight=0.7,  # 衰减损失权重（主要影响阈值参数）
#                  range_weight=0.3,  # 脉冲率范围权重（影响卷积参数）
#                  per_channel=True):
#         self.target_min_rate = target_min_rate
#         self.target_max_rate = target_max_rate
#         self.decay_weight = decay_weight
#         self.range_weight = range_weight
#         self.per_channel = per_channel
#
#     def __call__(self, spike_history, membrane_history, thresh_history,
#                  gate_conv_params=None, input_conv_params=None, thresh_decay_params=None):
#         T, B, C, H, W = spike_history.shape
#
#         # 分通道计算脉冲率 [T, C]
#         spike_rates = torch.mean(spike_history.float(), dim=[1, 3, 4])
#         target_decay = 0.7
#         # 1. 时间衰减损失 - 主要影响 thresh_decay 参数
#         if T > 1:
#             if self.per_channel:
#                 actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)
#
#                 # 使用更宽松的衰减约束
#                 decay_loss = torch.mean(F.softplus(
#                     torch.abs(actual_decay_rates - target_decay) - 0.25, beta=2  # 宽松约束
#                 ))
#             else:
#                 actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)
#                 decay_loss = torch.mean(F.softplus(
#                     torch.abs(actual_decay_rates - target_decay) - 0.25, beta=2
#                 ))
#         else:
#             decay_loss = torch.tensor(0.0, device=spike_history.device)
#
#         # 2. 脉冲率范围约束 - 主要影响卷积参数
#         late_rates = spike_rates[T // 2:]
#         mean_late_rates = torch.mean(late_rates, dim=0)
#
#         # 使用非常宽松的范围约束
#         min_rate_loss = F.softplus(self.target_min_rate - mean_late_rates, beta=1)
#         max_rate_loss = F.softplus(mean_late_rates - self.target_max_rate, beta=1)
#         range_loss = torch.mean(min_rate_loss + max_rate_loss)
#
#         # 组合损失
#         total_loss = self.decay_weight * decay_loss + self.range_weight * range_loss
#
#         # 记录统计信息
#         self.stats = {
#             'total_loss': total_loss.detach().item(),
#             'decay_loss': decay_loss.detach().item() if T > 1 else 0.0,
#             'range_loss': range_loss.detach().item(),
#             'avg_spike_rate': torch.mean(spike_rates).detach().item(),
#             'late_spike_rate': torch.mean(mean_late_rates).detach().item(),
#             'actual_decay': torch.mean(actual_decay_rates).detach().item() if T > 1 else 1.0
#         }
#
#         return total_loss


class SpikeLossMonitor:
    def __init__(self):
        self.history = {
            'total_loss': [],
            'balance_loss': [],
            'util_loss': [],
            'avg_spike_rate': [],
            'late_spike_rate': [],
            'decay_rate': [],
            'thresh_decay': [],
            'adapt_factor': []
        }

    def update(self, stats):
        """更新监控数据"""
        for key in self.history:
            if key in stats:
                self.history[key].append(stats[key])

    def get_summary(self):
        """获取统计摘要"""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f'avg_{key}'] = sum(values) / len(values)
                summary[f'min_{key}'] = min(values)
                summary[f'max_{key}'] = max(values)
                summary[f'last_{key}'] = values[-1] if values else 0
        return summary

    def plot_history(self, save_path='enhanced_spike_history.png'):
        """绘制历史数据图表"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # 损失历史
        axes[0, 0].plot(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')

        axes[0, 1].plot(self.history['balance_loss'])
        axes[0, 1].set_title('Balance Loss')

        axes[0, 2].plot(self.history['util_loss'])
        axes[0, 2].set_title('Utilization Loss')

        # 脉冲统计历史
        axes[0, 3].plot(self.history['avg_spike_rate'])
        axes[0, 3].set_title('Avg Spike Rate')

        axes[1, 0].plot(self.history['late_spike_rate'])
        axes[1, 0].set_title('Late Spike Rate')
        axes[1, 0].axhline(y=0.05, color='r', linestyle='--', label='Target')
        axes[1, 0].legend()

        axes[1, 1].plot(self.history['decay_rate'])
        axes[1, 1].set_title('Decay Rate')
        axes[1, 1].axhline(y=0.7, color='r', linestyle='--', label='Target')
        axes[1, 1].legend()

        # 参数历史
        axes[1, 2].plot(self.history['thresh_decay'])
        axes[1, 2].set_title('Threshold Decay')

        axes[1, 3].plot(self.history['adapt_factor'])
        axes[1, 3].set_title('Adaptation Factor')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def reset(self):
        """重置监控数据"""
        for key in self.history:
            self.history[key] = []


# class MembraneLoss:
#     def __init__(self,
#                  target_min_rate=0.02,  # 降低最小目标脉冲率
#                  target_decay_rate=0.7,  # 允许更快的衰减
#                  balance_weight=0.7,  # 降低平衡权重
#                  util_weight=0.3,  # 降低利用率权重
#                  per_channel=True):  # 使用通道独立计算
#         self.target_min_rate = target_min_rate
#         self.target_decay_rate = target_decay_rate
#         self.balance_weight = balance_weight
#         self.util_weight = util_weight
#         self.per_channel = per_channel
#
#     def __call__(self, spike_history, membrane_history, thresh_history):
#         T = spike_history.shape[0]
#         if self.per_channel:
#             # 按通道计算脉冲率 [T, C]
#             spike_rates = torch.mean(spike_history.float(), dim=[1, 3, 4])  # [T, C]
#         else:
#             # 全局计算脉冲率 [T]
#             spike_rates = torch.mean(spike_history.float(), dim=[1, 2, 3, 4])  # [T]
#
#         # 时间平衡损失 - 减弱约束
#         if T > 1:
#             if self.per_channel:
#                 actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)  # [T-1, C]
#                 decay_loss = torch.mean((actual_decay_rates - self.target_decay_rate) ** 2)
#             else:
#                 actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)  # [T-1]
#                 decay_loss = torch.mean((actual_decay_rates - self.target_decay_rate) ** 2)
#         else:
#             decay_loss = torch.tensor(0.0, device=spike_history.device)
#
#         # 后期脉冲率损失 - 减弱约束
#         if self.per_channel:
#             late_spike_rate = torch.mean(spike_rates[T // 2:], dim=0)  # [C]
#             min_rate_loss = torch.mean(F.softplus(self.target_min_rate - late_spike_rate, beta=5))
#         else:
#             late_spike_rate = torch.mean(spike_rates[T // 2:])
#             min_rate_loss = F.softplus(self.target_min_rate - late_spike_rate, beta=5)
#
#         balance_loss = decay_loss + min_rate_loss
#         upper_bound = 0.8
#         # 膜电位利用率损失 - 只保留上限约束
#         non_spiking_mask = (spike_history == 0).float()
#         membrane_ratio = (membrane_history * non_spiking_mask) / (thresh_history + 1e-7)
#
#         # 只惩罚过高的膜电位利用率
#         upper_loss = F.softplus(membrane_ratio - upper_bound, beta=5)
#
#         if self.per_channel:
#             util_loss = torch.mean(upper_loss, dim=[0, 1, 3, 4])  # [C]
#             util_loss = torch.mean(util_loss)  # 平均所有通道的损失
#         else:
#             util_loss = torch.mean(upper_loss)
#
#
#         # 组合总损失
#         total_loss = self.balance_weight * balance_loss + self.util_weight * util_loss
#
#         # 记录统计信息
#         self.stats = {
#             'balance_loss': balance_loss.detach().item(),
#             'util_loss': util_loss.detach().item(),
#             'avg_spike_rate': torch.mean(spike_rates).detach().item(),
#             'late_spike_rate': torch.mean(late_spike_rate).detach().item(),
#             'membrane_ratio_mean': torch.mean(membrane_ratio).detach().item(),
#             'actual_decay_rate': torch.mean(
#                 (spike_rates[-1] / (spike_rates[0] + 1e-7))).detach().item() if T > 1 else 1.0
#         }
#
#         return total_loss

class MembraneLoss:
    def __init__(self,
                 target_min_rate=0.05,  # 降低最小目标脉冲率
                 target_decay_rate=0.7,  # 允许更快的衰减
                 balance_weight=0.6,  # 降低平衡权重
                 util_weight=0.4,  # 降低利用率权重
                 per_channel=True):  # 使用通道独立计算
        self.target_min_rate = target_min_rate
        self.target_decay_rate = target_decay_rate
        self.balance_weight = balance_weight
        self.util_weight = util_weight
        self.per_channel = per_channel

    def __call__(self, spike_history, membrane_history, thresh_history):
        T = spike_history.shape[0]
        if self.per_channel:
            # 按通道计算脉冲率 [T, C]
            spike_rates = torch.mean(spike_history.float(), dim=[1, 3, 4])  # [T, C]
        else:
            # 全局计算脉冲率 [T]
            spike_rates = torch.mean(spike_history.float(), dim=[1, 2, 3, 4])  # [T]

        # 时间平衡损失 - 减弱约束
        if T > 1:
            if self.per_channel:
                actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)  # [T-1, C]
                decay_loss = torch.mean((actual_decay_rates - self.target_decay_rate) ** 2)
            else:
                actual_decay_rates = spike_rates[1:] / (spike_rates[:-1] + 1e-7)  # [T-1]
                decay_loss = torch.mean((actual_decay_rates - self.target_decay_rate) ** 2)
        else:
            decay_loss = torch.tensor(0.0, device=spike_history.device)

        # 后期脉冲率损失 - 减弱约束
        if self.per_channel:
            late_spike_rate = torch.mean(spike_rates[T // 2:], dim=0)  # [C]
            min_rate_loss = torch.mean(F.softplus(self.target_min_rate - late_spike_rate, beta=5))
        else:
            late_spike_rate = torch.mean(spike_rates[T // 2:])
            min_rate_loss = F.softplus(self.target_min_rate - late_spike_rate, beta=5)

        balance_loss = decay_loss + min_rate_loss

        # 膜电位利用率损失 - 减弱约束
        non_spiking_mask = (spike_history == 0).float()
        membrane_ratio = (membrane_history * non_spiking_mask) / (thresh_history + 1e-7)

        # 扩大可接受的范围
        lower_bound = 0.2  # 从0.2降低到0.1
        upper_bound = 0.7  # 从0.7提高到0.8

        # 使用更温和的惩罚函数
        lower_loss = F.softplus(lower_bound - membrane_ratio, beta=5)
        upper_loss = F.softplus(membrane_ratio - upper_bound, beta=5)

        if self.per_channel:
            # 按通道计算利用率损失
            util_loss = torch.mean(lower_loss + upper_loss, dim=[0, 1, 3, 4])  # [C]
            util_loss = torch.mean(util_loss)  # 平均所有通道的损失
        else:
            util_loss = torch.mean(lower_loss + upper_loss)

        # 组合总损失
        total_loss = self.balance_weight * balance_loss + self.util_weight * util_loss

        # 记录统计信息
        self.stats = {
            'balance_loss': balance_loss.detach().item(),
            'util_loss': util_loss.detach().item(),
            'avg_spike_rate': torch.mean(spike_rates).detach().item(),
            'late_spike_rate': torch.mean(late_spike_rate).detach().item(),
            'membrane_ratio_mean': torch.mean(membrane_ratio).detach().item(),
            'actual_decay_rate': torch.mean(
                (spike_rates[-1] / (spike_rates[0] + 1e-7))).detach().item() if T > 1 else 1.0
        }

        return total_loss


class DTASNNEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, Ts, kwargs_spikes, split=False, spike_attach=False,
                 write_zero=False, abs=False, depth=1,
                 readout='avg',
                 ):
        super(DTASNNEmbedding, self).__init__()
        self.kernel_size = kernel_size
        self.kwargs_spikes = kwargs_spikes
        self.Ts = Ts  # 生成张量的时间步长
        self.abs = abs
        self.split = split
        self.readout = readout
        # self.record = record
        self.write_zero = write_zero
        # 安全获取参数（带默认值）
        self.nb_steps = self.kwargs_spikes.get('nb_steps', 5)  # 兼容Tm别名，原始输入数据的时间步长
        self.thresh = self.kwargs_spikes.get('thresh', 1.0)
        self.vreset = self.kwargs_spikes.get('vreset', 0.0)  # 如果 'vreset' 是 None，直接赋 None，不需要深拷贝
        self.vreset = None if str(self.vreset).lower() == "none" else self.vreset
        # 动态解析spike_fn
        spike_fn = self.kwargs_spikes.get('spike_fn', 'Rectangle')
        if isinstance(spike_fn, str):
            spike_fn = globals().get(spike_fn, Rectangle)  # 从全局查找函数/类
        self.act_fun = self.warp_spike_fn(spike_fn)
        self.depth = int(depth)
        self.gate_conv = self.build_conv(out_channel, out_channel * 2, kernel_size, depth=self.depth)
        self.input_conv = self.build_conv(in_channel, out_channel * 2, kernel_size, depth=self.depth)
        if self.split:
            self.gate_conv_agg = nn.Conv2d(out_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
            self.input_conv_agg = nn.Conv2d(in_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
        self.spike_attach = spike_attach
        self.thresh_decay = nn.Parameter(torch.ones(out_channel) * 0.9)
        self._init_weight()
        self.membrane_loss = MembraneLoss(target_min_rate=0.05,  # 降低最小目标脉冲率
                                          target_decay_rate=0.7,  # 允许更快的衰减
                                          balance_weight=0.6,  # 降低平衡权重
                                          util_weight=0.4,  # 降低利用率权重
                                          per_channel=True)
        # 新增每个时间步的脉冲发放率统计
        self.fr_per_step = [0.0, 0.0, 0.0, 0.0, 0.0]  # 存储每个时间步的发放率
        self.fr_count_per_step = [0, 0, 0, 0, 0]  # 存储每个时间步的统计次数
        # 脉冲损失监控器
        self.monitor = SpikeLossMonitor()
        self.loss_stats = None

    def reset_fr(self):
        """重置发放率统计"""
        self.fr_per_step = [0.0, 0.0, 0.0, 0.0, 0.0]  # 存储每个时间步的发放率
        self.fr_count_per_step = [0, 0, 0, 0, 0]  # 存储每个时间步的统计次数

    def get_fr_per_step(self):
        """获取每个时间步的平均发放率"""
        return [fr / max(1, count) for fr, count in zip(self.fr_per_step, self.fr_count_per_step)]

    def build_conv(self, in_channel, out_channel, kernel_size, depth=1):
        convs = [nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)]
        for _ in range(depth - 1):
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))
        return nn.Sequential(*convs)

    def warp_spike_fn(self, spike_fn):
        if isinstance(spike_fn, nn.Module):
            return copy.deepcopy(spike_fn)
        elif issubclass(spike_fn, torch.autograd.Function):
            return spike_fn.apply
        elif issubclass(spike_fn, torch.nn.Module):
            return spike_fn()

    def _init_weight(self):
        for m in self.input_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        for m in self.gate_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='sigmoid')
        if self.split:
            nn.init.kaiming_uniform_(self.input_conv_agg.weight, nonlinearity='sigmoid')
            nn.init.orthogonal_(self.gate_conv_agg.weight, gain=nn.init.calculate_gain('relu'))

    def update(self, vmem, gate, current, t):
        current_thresh = self.thresh * (self.thresh_decay.view(1, -1, 1, 1) ** t)
        vmem = gate * vmem + current  # gate 可以看作是一个调节因子，如果 gate 为 0，膜电位不受影响；如果为 1，则膜电位完全由 current 决定。
        spike = self.act_fun(vmem - current_thresh)
        if self.vreset is None:
            vmem_update = vmem - self.thresh * spike
        else:
            vmem_update = vmem * (1 - spike) + self.vreset * spike
        return vmem_update, vmem, spike, current_thresh

    def forward(self, events, record=False, v_record=False):
        # print("start embedding")
        # if events.dim() == 5:
        #     # 生成一次 random_str
        #     random_str = uuid.uuid4().hex[:8]
        # pltimage(events, save_dir='./before_embedding/', random_str=random_str)
        param_dtype = next(self.parameters()).dtype  # 获取模型参数类型
        events = events.to(param_dtype)  # 强制转换输入类型,BTCHW
        # 初始化当前批次的每个时间步发放率
        current_fr_per_step = [0.0] * self.nb_steps

        if events.dim() < 5:  # handle with the input for prameter registering
            events, _ = torch.broadcast_tensors(events, torch.zeros((self.Ts,) + events.shape))
            return events
        elif events.dim() > 5:  # handle with the input for prameter registering
            # B*T*T'*C*H*W,
            shape = events.shape[:-4]  # 去掉后四个维度
            events = events.flatten(end_dim=-5)  # 从倒数第5个维度后续全都展平
            events = events.transpose(0, 1)  # get the shape of T'*(B*T)*C*H*W
        else:  # BTCHW
            events = events.transpose(0, 1)  # get TBCHW

        # stats = compute_pixel_stats_tensor(events)
        # #
        # # 查看第一个时间步、第一个通道的统计结果
        # print("时间步 0, 通道 0 的统计:", stats[0, 0])
        # todo: revert the sequence of the input [need for test] # TBCHW
        indices = torch.arange(events.size(0) - 1, -1, -1).to(events.device)  # 生成一个与时间部相同的逆序张量
        events = torch.index_select(events, 0, indices)  # 把事件T时间步从正序改成逆序
        spike_last = torch.zeros_like(events[0], dtype=param_dtype)
        vmem = torch.zeros_like(events[0], dtype=param_dtype)
        # if self.split:
        #     spike_last_agg = torch.zeros_like(events[0])
        #     vmem_agg = torch.zeros_like(events[0])
        aggregation = torch.zeros([self.Ts] + list(events.shape[1:]), device=events.device, dtype=param_dtype)
        seg_ind = torch.zeros_like(events[0]).long()
        vmem_avg = torch.zeros_like(events[0])  # memory for the ouptut of the embedding
        t_last = torch.zeros_like(events[0]).long() - 1  # 构建全为-1的tensor
        t_record = []
        v_record_list = []
        # 只在训练时分配历史记录张量
        if self.training:
            membrane_history = torch.zeros_like(events, dtype=param_dtype, device=events.device)
            spike_history = torch.zeros_like(events, dtype=param_dtype, device=events.device)
            current_history = torch.zeros_like(events, dtype=param_dtype, device=events.device)
            C = self.thresh_decay.size(0)
            thresh_history = torch.zeros(
                (self.nb_steps, 1, C, 1, 1),
                dtype=param_dtype, device=events.device
            )
        else:
            # 在评估模式下，不需要记录历史
            membrane_history = None
            spike_history = None
            current_history = None
            thresh_history = None
        with torch.cuda.amp.autocast():
            for t in range(self.nb_steps):
                state = self.gate_conv(spike_last)  # (B,4,H,W)
                g_rec, c_rec = state.chunk(2, dim=-3)  # 沿倒数第三个维度进行拆分，（B，2，640，640）
                inpt = self.input_conv(events[t])  # 取时间步第0个维度，进行卷积操作
                g_in, c_in = inpt.chunk(2, dim=-3)
                gate = torch.sigmoid(g_in + g_rec)  # （1，2，640，640）,代表门控状态
                # 非对称激活策略
                # 电流路径应用ReLU保证非负性
                # c_rec = torch.relu(c_rec)  # 循环电流
                # c_in = torch.relu(c_in)  # 输入电流

                current = (c_in + c_rec)  # 代表现在的输入电流，代表当前和上个时刻电流和
                # todo: should call hard reset to clear the history
                vmem, vmem_no_reset, spike_last, current_thresh = self.update(vmem, gate, current,
                                                                              t)  # LIF神经元，vmem膜电位更新后，vmem_no_reset

                # 记录历史（用于损失计算）
                if self.training:
                    membrane_history[t] = vmem_no_reset  # 记录更新后的膜电位
                    spike_history[t] = spike_last  # 记录脉冲
                    current_history[t] = current  # 记录电流
                    thresh_history[t] = current_thresh

                # # 统计脉冲发放率
                # counts = torch.count_nonzero(spike_last, dim=(2, 3))  # shape: (2, 2)
                #
                # # 打印结果
                # print("Non-zero pixel counts per batch and channel:")
                # for batch in range(counts.shape[0]):
                #     for channel in range(counts.shape[1]):
                #         print(f"Batch {batch}, Channel {channel}: {counts[batch, channel].item()}")

                # 计算当前时间步的发放率
                fr_t = spike_last.mean().item()
                current_fr_per_step[t] = fr_t

                # 为膜电位未重置
                vmem_avg += vmem_no_reset  # 加上未激活的膜电位
                v_record_list.append(vmem_no_reset[(
                        1 - spike_last).bool()])  # 将当前膜电位（未重置）添加到记录列表中，只记录那些没有发射脉冲的膜电位。 转化为一个list中的tensor(
                # 1615210)代表所有未发放脉冲的膜电位
                spike_pos = spike_last.nonzero()  # 找到所有发射脉冲的位置，返回这些位置的索引。为（10541，4）代表每个维度的索引, 例如(0,0,78,
                # 568)代表，张量中这个位置发射脉冲
                seg_pos = seg_ind[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:,
                                                                                     3]]  # 使用脉冲位置的索引获取对应的分段位置
                # seg_pos。代表第几段的脉冲
                valid_pos = seg_pos < self.Ts  # 检查 seg_pos 是否小于阈值 self.Ts，用于过滤有效的位置。
                seg_pos, spike_pos = seg_pos[valid_pos], spike_pos[valid_pos]  # 只保留有效的位置，更新 seg_pos 和 spike_pos。
                if self.readout == 'sum':
                    v = vmem_avg[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]  # 把所有发放脉冲位置的值记录下来
                elif self.readout == 'last':
                    v = vmem[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
                elif self.readout == 'avg':
                    v = vmem_avg[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]] / (
                            t - t_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:,
                                                                                          3]])  # 第0步，把所有发射脉冲的位置记为1
                    # ，第1步，率先发射脉冲的记为0，已经发射过脉冲的记为1
                if self.spike_attach:
                    v *= spike_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
                aggregation[seg_pos, spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:,
                                                                                        3]] += v  # 将调整后的膜电位 v 加到聚合张量上面
                # aggregation 张量的相应位置。
                seg_ind[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:,
                                                                           3]] += 1  # 将 seg_ind 中对应的分段索引增加，所有发射过脉冲的位置加1
                # 1，表示在该位置的脉冲数量增加。
                t_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:,
                                                                          3]] = t  # 将当前时间 t 更新到 t_last，
                # 张量中，标记每个发射脉冲的最新时间。
                vmem_avg[spike_last.bool()] = 0  # 发射脉冲的位置记为0，重置膜电位
                if record:
                    t_record.append(t_last.clone())
                # print(f"seg_ind的最小值为:{seg_ind.min()}")
                if seg_ind.min() >= self.Ts:
                    break
                    # 更新全局统计
            for t in range(self.nb_steps):
                if t < len(current_fr_per_step):  # 确保索引有效
                    self.fr_per_step[t] += current_fr_per_step[t]
                    self.fr_count_per_step[t] += 1
            # handle the remained segment with no spikes in the last time step
            no_spike_pos = (1 - spike_last).nonzero()  # 计算最后一个时间步，没有发射脉冲的神经元的位置。
            seg_pos = seg_ind[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            valid_pos = seg_pos < self.Ts
            seg_pos, no_spike_pos = seg_pos[valid_pos], no_spike_pos[valid_pos]
            if self.readout == 'sum':
                v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            elif self.readout == 'last':
                v = vmem[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            elif self.readout == 'avg':
                v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] / (
                        self.nb_steps - 1 - t_last[
                    no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]])
            if self.write_zero:
                v *= 0
            aggregation[seg_pos, no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] += v
            # max_val = torch.max(aggregation)  # tensor(9)
            # min_val = torch.min(aggregation)  # tensor(1)
            # print(f"处理后张量数据范围：最大值为 {max_val}，最小值为 {min_val}")
            # if aggregation.dim() == 5:
            #     print("embedding end")
            #     pltimage(aggregation, save_dir='./after_embedding_1depth_4ts_memloss/',
            #              random_str=random_str)
            # for ts in range(self.Ts):
            #     for b in range(aggregation[ts].size(0)):
            #         spike_count = (aggregation[ts][b] > 0).sum().item()
            #         print(f"TimeStep {ts} and batch {b}: {spike_count} spikes")
            if self.training:
                aux_loss = self.membrane_loss(spike_history, membrane_history, thresh_history)
                # 记录统计信息
                self.loss_stats = self.membrane_loss.stats

                # 从阈值历史计算实际衰减率
                T = thresh_history.shape[0]
                thresh_values = thresh_history.view(T, -1).mean(dim=1)  # [T]
                early_thresh = thresh_values[:T // 2].mean()
                late_thresh = thresh_values[T // 2:].mean()
                actual_decay = late_thresh / (early_thresh + 1e-7)

                # 更新监控，包含实际衰减率
                self.loss_stats['actual_decay'] = actual_decay.item()
                self.monitor.update(self.loss_stats)
            else:
                aux_loss = None
            if self.abs:
                # print('use abs')
                aggregation = torch.nn.functional.relu(aggregation)
            if record:
                return aggregation, torch.stack(t_record, axis=0)
            elif v_record:
                return aggregation, torch.concatenate(v_record_list)
            else:
                if self.training:
                    return aggregation, aux_loss
                else:
                    return aggregation


def pltimage(aggregation, save_dir='./after_embedding/', random_str=None):
    os.makedirs(save_dir, exist_ok=True)

    # 原始尺寸和填充后尺寸
    H_orig, W_orig = 260, 346
    H_pad, W_pad = 640, 640

    # 计算LetterBox变换的参数
    r = min(H_pad / H_orig, W_pad / W_orig)  # 缩放比例
    new_unpad = int(round(W_orig * r)), int(round(H_orig * r))  # 缩放后的尺寸 (w, h)

    # 计算填充量
    dw, dh = W_pad - new_unpad[0], H_pad - new_unpad[1]  # wh padding
    dw /= 2  # 左右平分
    dh /= 2  # 上下平分

    # 计算填充的具体像素值
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 遍历时间步和 batch
    for ts in range(aggregation.shape[0]):
        for b in range(aggregation.shape[1]):
            # 提取单张图像 (C, H, W)
            img_tensor = aggregation[ts, b].detach().cpu()  # 移动到 CPU
            img = img_tensor.numpy()

            # 裁剪回原始尺寸 (C, H_orig, W_orig)
            # 先去除填充，然后缩放到原始尺寸
            img_cropped = img[:, top:top + new_unpad[1], left:left + new_unpad[0]]  # 去除填充
            img_resized = np.zeros((2, H_orig, W_orig), dtype=np.float32)

            # 对每个通道进行缩放
            for c in range(2):
                img_resized[c] = cv2.resize(img_cropped[c], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

            # 创建RGBA图像
            rgba_image = create_rgba_image(img_resized)

            # 保存图像
            filename = os.path.join(save_dir, f"ts_{ts}_batch_{b}_{random_str}.png")
            cv2.imwrite(filename, rgba_image)


def create_rgba_image(img: np.ndarray) -> np.ndarray:
    """
    创建RGBA图像，使用分位数截断增强对比度
    """
    # 确保输入是浮点数类型
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)

    # 分离正负极性
    positive = img[0]  # 正极性
    negative = img[1]  # 负极性

    # 使用分位数截断进行归一化，增强对比度
    def normalize_channel_with_clamp(channel, lower_percentile=1, upper_percentile=99):
        # 计算分位数
        lower_bound = np.percentile(channel, lower_percentile)
        upper_bound = np.percentile(channel, upper_percentile)

        # 截断极值
        channel_clipped = np.clip(channel, lower_bound, upper_bound)

        # 归一化到[0, 1]
        epsilon = 1e-8
        normalized = (channel_clipped - lower_bound) / (upper_bound - lower_bound + epsilon)

        # 应用gamma校正进一步增强对比度
        gamma = 0.8  # 小于1的值会增强暗部对比度
        normalized = np.power(normalized, gamma)

        return normalized

    # 分别归一化正负极性
    positive_norm = normalize_channel_with_clamp(positive)
    negative_norm = normalize_channel_with_clamp(negative)

    # 计算整体强度（用于透明度）
    intensity = np.maximum(positive_norm, negative_norm)

    # 创建RGBA图像
    H, W = positive.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # 设置红色通道（正极性）- 使用更强的红色
    rgba[..., 0] = (positive_norm * 255).astype(np.uint8)

    # 设置蓝色通道（负极性）- 使用更强的蓝色
    rgba[..., 2] = (negative_norm * 255).astype(np.uint8)

    # 设置绿色通道为0
    rgba[..., 1] = 0

    # 设置透明度通道（强度越高越不透明）
    rgba[..., 3] = (intensity * 255).astype(np.uint8)

    # 创建白色背景
    white_bg = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 将RGBA图像合成到白色背景上
    alpha = rgba[..., 3] / 255.0
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # 合成公式: result = foreground * alpha + background * (1 - alpha)
    result = rgba[..., :3] * alpha + white_bg * (1 - alpha)
    result = result.astype(np.uint8)

    return result


def compute_pixel_stats_tensor(im):
    """
    统计每个时间步、每个通道中：
    - 像素值为 0 的数量
    - 像素值为 1 的数量
    - 像素值大于 30 的数量
    - 像素值小于 30 且大于 1 的数量

    参数:
        im (torch.Tensor): 形状为 (T, C, H, W) 的 PyTorch 张量，其中：
                           T = 时间步数
                           C = 通道数
                           H = 图像高度
                           W = 图像宽度

    返回:
        torch.Tensor: 形状为 (T, C, 4) 的张量，每个元素对应上述四个统计量
    """
    # 定义四个布尔掩码
    mask_zero = (im == 0)
    mask_one = (im == 1)
    mask_gt30 = (im > 30)
    mask_lt30 = (im < 30) & (im > 1)

    # 沿空间维度（H, W）求和
    count_zero = torch.sum(mask_zero, dim=(3, 4))
    count_one = torch.sum(mask_one, dim=(3, 4))
    count_gt30 = torch.sum(mask_gt30, dim=(3, 4))
    count_lt30 = torch.sum(mask_lt30, dim=(3, 4))

    # 合并结果，形状为 (T, C, 4)
    stats = torch.stack([count_zero, count_one, count_gt30, count_lt30], dim=-1)

    return stats
