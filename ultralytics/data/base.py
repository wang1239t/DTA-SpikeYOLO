# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from matplotlib import patches, pyplot as plt
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from .scripts.generate_slices import to_voxel_cube_numpy, to_timesurface_numpy, to_voxel_grid_numpy
from skimage import exposure

from .utils import HELP_URL, IMG_FORMATS


def slice_events(events, num_slice, overlap=0):
    # events = self.read_ATIS(data_path, is_stream=False)
    times = events[:, 3]
    if len(times) <= 0:
        return [None for i in range(num_slice)], 0
    # logger.info(times.max(), times.min())
    time_window = (times[-1] - times[0]) // (
            num_slice * (
            1 - overlap) + overlap)  # 等分时间窗口，每个时间窗口的长度  # todo: check here, ignore some special streams
    stride = (1 - overlap) * time_window
    window_start_times = np.arange(num_slice) * stride + times[0]  # 获取每个窗口初始时间戳
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)  # 返回window_start_times所对应的索引
    indices_end = np.searchsorted(times, window_end_times)
    slices = [events[start:end] for start, end in list(zip(indices_start, indices_end))]
    # frames = np.stack([self.agrregate(slice) for slice in slices], axis=0)
    return slices, stride


def compute_pixel_stats(im):
    """
    统计每个时间步、每个通道中：
    - 像素值为 0 的数量
    - 像素值为 1 的数量
    - 像素值大于 30 的数量
    - 像素值小于 30 的数量

    参数:
        im (np.ndarray): 形状为 (T, C, H, W) 的 NumPy 数组，其中：
                         T = 时间步数
                         C = 通道数
                         H = 图像高度
                         W = 图像宽度

    返回:
        np.ndarray: 形状为 (T, C, 4) 的数组，每个元素对应上述四个统计量
    """
    # 定义四个布尔掩码
    mask_zero = (im == 0)
    mask_one = (im == 1)
    mask_gt30 = (im > 30)
    mask_lt30 = (im < 30) & (im > 1)

    # 沿空间维度（H, W）求和
    count_zero = np.sum(mask_zero, axis=(2, 3))
    count_one = np.sum(mask_one, axis=(2, 3))
    count_gt30 = np.sum(mask_gt30, axis=(2, 3))
    count_lt30 = np.sum(mask_lt30, axis=(2, 3))

    # 合并结果，形状为 (T, C, 4)
    stats = np.stack([count_zero, count_one, count_gt30, count_lt30], axis=-1)

    return stats


def voxel_deal(voxel_grid):
    output_img = []
    for i in range(voxel_grid.shape[0]):
        img = voxel_grid[i, 0]  # 获取第i个通道的数据

        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        var_pos = np.var(img[img > 0])
        var_neg = np.var(img[img < 0])

        # 对图像数据进行归一化处理
        img = np.clip(img, a_min=3 * mean_neg, a_max=3 * mean_pos)

        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        var_pos = np.var(img[img > 0])
        var_neg = np.var(img[img < 0])
        img = np.clip(img, a_min=mean_neg - 3 * var_neg, a_max=mean_pos + 3 * var_pos)

        max_val = np.max(img)
        min_val = np.min(img)

        # 将正负值分别归一化
        img[img > 0] /= max_val
        img[img < 0] /= abs(min_val)

        # 将归一化后的值映射到[0, 255]范围
        map_img = np.zeros_like(img)
        map_img[img < 0] = img[img < 0] * 128 + 128
        map_img[img >= 0] = img[img >= 0] * 127 + 128

        output_img.append(map_img)

    output_img = np.stack(output_img, axis=0)
    return output_img


def dynamic_clipping(data, percentile=99.5, min_threshold=30):
    nonzero_data = data[data > 0]
    if len(nonzero_data) == 0:
        return data
    threshold = np.percentile(nonzero_data, percentile)
    threshold = max(threshold, min_threshold)  # 保证最低截断值
    return np.clip(data, a_min=None, a_max=threshold)


def micro_sum_deal(data):
    time_size, channels, height, width = data.shape
    normalized_data = np.zeros_like(data)

    for t in range(time_size):
        for c in range(channels):
            channel_data = data[t, c, :, :]

            clipped_data = dynamic_clipping(channel_data)

            normalized_data[t, c, :, :] = clipped_data  # 正确赋值

    return normalized_data


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        """Initialize BaseDataset with given configuration and options."""
        # self.image_size = (240, 304)  # 原始
        self.image_size = (260, 346)  # 原始
        self.micro_slice = 5  # 拆分时间步长
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)  # 遍历文件夹目录
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache stuff
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):  # 返回文件夹的列表
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:  # 遍历文件夹
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)  # 返回文件夹中的所有文件
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found in {img_path}'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    # def load_image(self, i, rect_mode=True):
    #     """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR  cv2默认读取的格式是hwc
    #             if im is None:
    #                 raise FileNotFoundError(f'Image Not Found {f}')
    #             h0, w0 = im.shape[:2]  # orig hw
    #         if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
    #             r = self.imgsz / max(h0, w0)  # ratio
    #             if r != 1:  # if sizes are not equal
    #                 w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
    #                 im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    #                 #这里resize还没改 虽然im对应numpy是hwc格式的，但resize这里还要写resize成(w，h)这样的，返回的numpy结果还是hwc
    #         elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
    #             im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
    #
    #         # Add to buffer if training with augmentations
    #         if self.augment:
    #             self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #             self.buffer.append(i)
    #             if len(self.buffer) >= self.max_buffer_length:
    #                 j = self.buffer.pop(0)
    #                 self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
    #
    #         return im, (h0, w0), im.shape[:2]
    #
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        index = i
        im, f, fn = self.ims[index], self.im_files[index], self.npy_files[index]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                event = np.load(fn)
                # event = event[:, [0, 1, 3, 2]]
                im = self.agrregate(event, method='micro_sum')  # (T,C,H,W)

                # im = voxel_deal(im)
                # im = micro_sum_deal(im)
                # max_value = np.max(im)
                # min_value = np.min(im)
                # print(f"帧数据范围：最大值为 {max_value}，最小值为 {min_value}")
                # stats = compute_pixel_stats(im)
                # #
                # # 查看第一个时间步、第一个通道的统计结果
                # print("时间步 0, 通道 0 的统计:", stats[0, 0])
                # zeros_to_add = np.zeros((im.shape[0], 1, *im.shape[2:]), dtype=im.dtype)  # (T,3,260,346)
                #
                # # 沿第二个维度（轴=1）拼接
                # im = np.concatenate((im, zeros_to_add), axis=1)

                # voxel表示
                # im = im.reshape(4, 1, 260, 346)  # 添加通道维度
                # im = np.tile(im, (1, 3, 1, 1))  # 沿通道维度复制3次

                im = im.transpose(0, 3, 2, 1)  # (T,W,H,C)
                h0, w0 = self.image_size[0], self.image_size[1]  # orig hw 统一读取格式
            else:  # read image
                im = None  # BGR  cv2默认读取的格式是hwc
                h0, w0 = None, None
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            # pltimage1(im, self.labels[index], index)  # im(3,346,260,3)
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    if len(im.shape) == 4:
                        T, W, H, C = im.shape
                        img_new = np.zeros([T, w, h, C])
                        for i in range(T):
                            img_new[i] = cv2.resize(im[i], (h, w), interpolation=cv2.INTER_LINEAR)
                        im = img_new
                    else:
                        im = cv2.resize(im, (h, w), interpolation=cv2.INTER_LINEAR)
                    # 这里resize还没改 虽然im对应numpy是hwc格式的，但resize这里还要写resize成(w，h)这样的，返回的numpy结果还是hwc
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            im = im.transpose(0, 2, 1, 3)  # (T,H,W,C)
            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[index], self.im_hw0[index], self.im_hw[index] = im, (h0, w0), im.shape[
                                                                                       :2]  # im, hw_original, hw_resized
                self.buffer.append(index)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            if len(im.shape) == 4:
                im_shape = im.shape[1:3]
            else:
                im_shape = im.shape[:2]
            # pltimage2(im, self.labels[index], index, save_dir='./save_rect/')  # im(3,241,320,3)
            return im, (h0, w0), im_shape  # 这里也要修改，因为label缩放通过这里得到

        return self.ims[index], self.im_hw0[index], self.im_hw[index]  #

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)  # 这里还是初始图片的比例
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],  # 这里resized_shape有问题
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError

    def agrregate(self, events, method):
        # logger.info(events)

        if method == 'sum':
            frame = np.zeros(shape=[2, self.image_size[0] * self.image_size[1]])  # todo: check the datatype here
            if events is None:
                # logger.info('Warning: representation without events')
                return frame.reshape((2, self.image_size[0], self.image_size[1]))
            x = events[:, 0].astype(int)  # avoid overflow
            y = events[:, 1].astype(int)
            p = events[:, 2]
            # todo: simplify the following code with np.add.at
            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))  # 区别0和1的掩码
            for c in range(2):
                position = y[mask[c]] * self.image_size[1] + x[mask[c]]
                events_number_per_pos = np.bincount(position)
                frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
            frame = frame.reshape((2, self.image_size[0], self.image_size[1]))
        elif method == 'time':
            frame = np.zeros(shape=[2, self.image_size[0] * self.image_size[1]])  # todo: check the datatype here
            if events is None:
                # logger.info('Warning: representation without events')
                return frame.reshape((2, self.image_size[0], self.image_size[1]))
            x = events[:, 0].astype(int)  # avoid overflow
            y = events[:, 1].astype(int)
            p = events[:, 2]
            t = events[:, 3].astype(np.int64)
            t = t - t.min()
            time_interval = t.max() - t.min()
            t_s = t / time_interval
            # todo: simplify the following code with np.add.at
            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))  # 区别0和1的掩码
            for c in range(2):
                position = y[mask[c]] * self.image_size[1] + x[mask[c]]
                events_time = np.bincount(position, weights=t_s[mask[c]])
                frame[c][np.arange(len(events_time))] += events_time
            frame = frame.reshape((2, self.image_size[0], self.image_size[1]))
        elif method == 'voxel_grid':
            # todo:  try to merge it into measure_func
            frame = to_voxel_grid_numpy(events, sensor_size=[self.image_size[1], self.image_size[0], 2],
                                        n_time_bins=self.micro_slice)
            # frame = frame / np.abs(frame).max()
        elif method == 'micro_sum':
            if events is None:
                # logger.info('Warning: representation without events')
                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            slices, _ = slice_events(events, self.micro_slice)
            frame = np.stack([self.agrregate(slice, method='sum') for slice in slices])

        elif method == 'micro_time':
            if events is None:
                # logger.info('Warning: representation without events')
                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            slices, _ = slice_events(events, self.micro_slice)
            frame = np.stack([self.agrregate(slice, method='time') for slice in slices])

        elif method == 'timesurface':
            # radius = 1
            if events is None:
                # logger.info('Warning: representation without events')
                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            # print("!!!debug!!!",len(events))
            slices, dt = slice_events(events, self.micro_slice)
            frame = to_timesurface_numpy(slices, sensor_size=[self.image_size[1], self.image_size[0], 2], dt=dt,
                                         tau=50e3)

        elif method == 'voxel_cube':
            frame = to_voxel_cube_numpy(events, sensor_size=[self.image_size[1], self.image_size[0], 2],
                                        num_slices=self.micro_slice)
        else:
            frame = None

        return frame


def pltimage1(im, labels, index, save_dir='./save_before_rect/'):
    os.makedirs(save_dir, exist_ok=True)
    print(im.shape)
    classes = {
        0: 'car',
        1: 'two-wheel',
        2: 'pedestrian',
        3: 'bus',
        4: 'truck',
    }
    # (4,w,h,3)
    T, W, H, C = im.shape
    im = im.transpose(0, 2, 1, 3)
    bboxes = labels['bboxes']
    cls_ids = labels['cls']
    for i in range(im.shape[0]):
        # 将通道赋值给 RGB 图像的对应通道
        rgb_image = im[i]
        # 将数据类型转换为 float，以便进行归一化

        # 创建 Figure 并设置尺寸和分辨率
        dpi = 100  # 可调整的 dpi 值，与显示清晰度相关
        fig = plt.figure(
            figsize=(W / dpi, H / dpi),  # 根据图像实际宽高计算画布尺寸（英寸）
            dpi=dpi
        )
        ax = plt.Axes(fig, [0, 0, 1, 1])  # 坐标轴覆盖整个画布
        ax.set_axis_off()  # 关闭坐标轴显示
        fig.add_axes(ax)
        ax.imshow(rgb_image)
        # 绘制边界框
        for bbox, cls_id in zip(bboxes, cls_ids):
            cx, cy, w, h = bbox  # 解包归一化的中心坐标和宽高

            # 转换为绝对坐标
            abs_cx = cx * W  # 宽度方向（对应图像的列）
            abs_cy = cy * H  # 高度方向（对应图像的行）
            abs_w = w * W
            abs_h = h * H

            # 计算角点坐标
            xmin = abs_cx - abs_w / 2
            ymin = abs_cy - abs_h / 2
            xmax = abs_cx + abs_w / 2
            ymax = abs_cy + abs_h / 2

            # 确保坐标不越界
            xmin, xmax = np.clip([xmin, xmax], 0, W)
            ymin, ymax = np.clip([ymin, ymax], 0, H)

            # 绘制矩形框
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='y', facecolor='none'
            )
            ax.add_patch(rect)

            # 添加类别标签（可选：在框左上方显示）
            label = classes[int(cls_id)]  # 将 cls_id 转换为类别名称
            ax.text(
                xmin, ymin - 2, label,  # 将文字放在框顶部上方
                color='red', fontsize=10,
                verticalalignment='top',
                backgroundcolor='none'
            )
        # 设置标题并隐藏坐标轴
        plt.title(f"样本 {i} - 示例数据")
        plt.axis('off')

        # 保存图形而不是显示
        save_path = os.path.join(save_dir, f'output_image_{index}_frame_{i + 1}.png')
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',  # 移除画布外空白
            pad_inches=0  # 内边距设为0
        )
        plt.close()  # 关闭当前 Figure 释放内存


def pltimage2(im, labels, index, save_dir='./save_rect/'):
    os.makedirs(save_dir, exist_ok=True)
    classes = {
        0: 'car',
        1: 'two-wheel',
        2: 'pedestrian',
        3: 'bus',
        4: 'truck',
    }
    # (4,w,h,3)
    im = im.astype(np.uint8)
    T, H, W, C = im.shape
    bboxes = labels['bboxes']
    cls_ids = labels['cls']
    for i in range(im.shape[0]):
        # 将通道赋值给 RGB 图像的对应通道
        rgb_image = im[i]
        # 将数据类型转换为 float，以便进行归一化

        # 创建 Figure 并设置尺寸和分辨率
        dpi = 100  # 可调整的 dpi 值，与显示清晰度相关
        fig = plt.figure(
            figsize=(W / dpi, H / dpi),  # 根据图像实际宽高计算画布尺寸（英寸）
            dpi=dpi
        )
        ax = plt.Axes(fig, [0, 0, 1, 1])  # 坐标轴覆盖整个画布
        ax.set_axis_off()  # 关闭坐标轴显示
        fig.add_axes(ax)
        ax.imshow(rgb_image)
        # 绘制边界框
        for bbox, cls_id in zip(bboxes, cls_ids):
            cx, cy, w, h = bbox  # 解包归一化的中心坐标和宽高

            # 转换为绝对坐标
            abs_cx = cx * W  # 宽度方向（对应图像的列）
            abs_cy = cy * H  # 高度方向（对应图像的行）
            abs_w = w * W
            abs_h = h * H

            # 计算角点坐标
            xmin = abs_cx - abs_w / 2
            ymin = abs_cy - abs_h / 2
            xmax = abs_cx + abs_w / 2
            ymax = abs_cy + abs_h / 2

            # 确保坐标不越界
            xmin, xmax = np.clip([xmin, xmax], 0, W)
            ymin, ymax = np.clip([ymin, ymax], 0, H)

            # 绘制矩形框
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='y', facecolor='none'
            )
            ax.add_patch(rect)

            # 添加类别标签（可选：在框左上方显示）
            label = classes[int(cls_id)]  # 将 cls_id 转换为类别名称
            ax.text(
                xmin, ymin - 2, label,  # 将文字放在框顶部上方
                color='red', fontsize=10,
                verticalalignment='top',
                backgroundcolor='none'
            )
        # 设置标题并隐藏坐标轴
        plt.title(f"样本 {i} - 示例数据")
        plt.axis('off')

        # 保存图形而不是显示
        save_path = os.path.join(save_dir, f'output_image_{index}_frame_{i + 1}.png')
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',  # 移除画布外空白
            pad_inches=0  # 内边距设为0
        )
        plt.close()  # 关闭当前 Figure 释放内存
