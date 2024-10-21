import torch
import torch.nn.functional as F

def find_peaks(confidence_maps, threshold=0.01, window_size=3):
    num_keypoints, H, W = confidence_maps.shape

    max_pooled_maps = F.max_pool2d(confidence_maps.unsqueeze(1),
                                   kernel_size=window_size, stride=1, padding=window_size//2)
    peaks_mask = (confidence_maps == max_pooled_maps.squeeze(1)) & (confidence_maps > threshold)

    peaks_list = []
    for i in range(num_keypoints):
        peaks = torch.nonzero(peaks_mask[i], as_tuple=False)
        peaks_list.append(peaks.tolist())

    return peaks_list
