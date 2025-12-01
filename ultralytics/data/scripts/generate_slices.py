import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured


def VoxelGrid(events, num_bins, height, width):
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    last_stamp = events[-1][3]
    first_stamp = events[0][3]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 3] = (num_bins - 1) * (events[:, 3] - first_stamp) / deltaT
    ts = events[:, 3]
    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    pols = events[:, 2]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    voxel_grid = voxel_grid.transpose(1, 2, 0)

    return voxel_grid


def slice_events(events, num_slice, overlap=0):
    # events = self.read_ATIS(data_path, is_stream=False)
    times = events["t"]
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


# Code adapted from Tonic
def to_voxel_grid_numpy(events, sensor_size, n_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical
    flow, depth, and egomotion.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        n_time_bins: number of bins in the temporal axis of the voxel grid.

    Returns:
        numpy array of n event volumes (n,w,h,t)
    """
    assert sensor_size[2] == 2

    if len(events) == 0:
        return np.zeros(((n_time_bins, 1, sensor_size[1], sensor_size[0])), float)

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()

    # normalize the event timestamps so that they lie between 0 and n_time_bins
    ts = (
            n_time_bins
            * (events[:, 3].astype(float) - events[:, 3][0])
            / (events[:, 3][-1] - events[:, 3][0])
    )
    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    pols = events[:, 2]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid


def to_voxel_cube_numpy(events, sensor_size, num_slices, tbins=2):
    """ Representation that creates voxel cube in paper "Object detection with spiking neural networks on automotive \
        event data[C]2022 International Joint Conference on Neural Networks (IJCNN)"
        Parameters:
            events: ndarray of shape [num_events, num_event_channels]
            sensor_size: size of the sensor that was [W, H].
            num_slices: n slices of the voxel cube.
            tbins: number of micro bins in a slice
        Returns:
            numpy array of voxel cube (n,2*tbin,h,w)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2
    if len(events) == 0:
        # logger.info('Warning: representation without events')
        return np.zeros(shape=[num_slices, 2 * tbins, sensor_size[0], sensor_size[1]])
    events['t'] -= events['t'][0]
    times = events['t']
    time_window = (times[-1] - times[0]) // num_slices
    events = events[events['t'] < time_window * num_slices]
    # feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long),
    #                                     2 * tbins)  # 2*tbin 通道维度

    coords = torch.from_numpy(
        structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

    # Bin the events on T timesteps
    coords = torch.floor(coords / torch.tensor([time_window, 1, 1]))

    # TBIN computations
    tbin_size = time_window / tbins

    # get for each ts the corresponding tbin index
    tbin_coords = (events['t'] % time_window) // tbin_size
    # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
    tbin_feats = ((events['p'] + 1) * (tbin_coords + 1)) - 1

    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * tbins)

    sparse_tensor = torch.sparse_coo_tensor(
        coords.t().to(torch.int32),
        feats,
        (num_slices, sensor_size[1], sensor_size[0], 2 * tbins)
    )

    voxel_cube = sparse_tensor.coalesce().to(float).to_dense().permute(0, 3, 1, 2)  # torch.Tensor [n, 2*tbin, H, W]
    return voxel_cube.numpy()


def to_timesurface_numpy(slices, sensor_size, dt, tau, overlap=0):
    assert dt >= 0, print("Parameter delta_t cannot be negative.")
    if slices[0] is None:
        return np.zeros(shape=[len(slices), 2, sensor_size[1], sensor_size[0]])
    all_surfaces = []
    memory = np.zeros((sensor_size[::-1]), dtype=int)  # p y x
    x_index, y_index, p_index, t_index = 0, 1, 2, 3  # 固定列索引
    start_t = slices[0][0, t_index]
    for i, slice in enumerate(slices):
        # 提取字段
        p = slice[:, p_index].astype(int)
        y = slice[:, y_index].astype(int)
        x = slice[:, x_index].astype(int)
        t = slice[:, t_index].astype(np.float64)

        # 更新 memory 中的时间戳
        indices = (p, y, x)
        memory[indices] = t

        diff = -((i + 1) * dt + start_t - memory)
        surf = np.exp(diff / tau)
        all_surfaces.append(surf)  # [n,p, H, W]
    return np.stack(all_surfaces, axis=0)
