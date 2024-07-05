import csv
import io
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data.dataset import Dataset

import animatediff.data.video_transformer as video_transforms
from animatediff.utils.util import detect_edges, zero_rank_print


try:
    from petrel_client.client import Client
except ImportError as e:
    print(e)


def get_score(video_data, cond_frame_idx, weight=[1.0, 1.0, 1.0, 1.0], use_edge=True):
    """
    Similar to get_score under utils/util.py/detect_edges
    """
    """
        the shape of video_data is f c h w, np.ndarray
    """
    h, w = video_data.shape[1], video_data.shape[2]

    cond_frame = video_data[cond_frame_idx]
    cond_hsv_list = list(cv2.split(cv2.cvtColor(cond_frame.astype(np.float32), cv2.COLOR_RGB2HSV)))

    if use_edge:
        cond_frame_lum = cond_hsv_list[-1]
        cond_frame_edge = detect_edges(cond_frame_lum.astype(np.uint8))
        cond_hsv_list.append(cond_frame_edge)

    score_sum = []

    for frame_idx in range(video_data.shape[0]):
        frame = video_data[frame_idx]
        hsv_list = list(cv2.split(cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2HSV)))

        if use_edge:
            frame_img_lum = hsv_list[-1]
            frame_img_edge = detect_edges(lum=frame_img_lum.astype(np.uint8))
            hsv_list.append(frame_img_edge)

        hsv_diff = [np.abs(hsv_list[c] - cond_hsv_list[c]) for c in range(len(weight))]
        hsv_mse = [np.sum(hsv_diff[c]) * weight[c] for c in range(len(weight))]
        score_sum.append(sum(hsv_mse) / (h * w) / (sum(weight)))

    return score_sum


class WebVid10M(Dataset):
    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
        use_petreloss=False,
        conf_path=None,
    ):
        if use_petreloss:
            self._client = Client(conf_path=conf_path, enable_mc=True)
        else:
            self._client = None
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        self.temporal_sampler = video_transforms.TemporalRandomCrop(sample_n_frames * sample_stride)

        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict["videoid"], video_dict["name"], video_dict["page_dir"]

        if self._client is not None:
            video_dir = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
            video_bytes = self._client.Get(video_dir)
            video_bytes = io.BytesIO(video_bytes)
            # ensure not reading zero byte
            assert video_bytes.getbuffer().nbytes != 0
            video_reader = VideoReader(video_bytes)
        else:
            video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
            video_reader = VideoReader(video_dir)

        total_frames = len(video_reader)
        if not self.is_image:
            start_frame_ind, end_frame_ind = self.temporal_sampler(total_frames)
            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)
        else:
            frame_indice = [random.randint(0, total_frames - 1)]

        pixel_values_np = video_reader.get_batch(frame_indice).asnumpy()
        cond_frames = random.randint(0, self.sample_n_frames - 1)

        # f h w c -> f c h w
        pixel_values = torch.from_numpy(pixel_values_np).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name, cond_frames, videoid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, name, cond_frames, videoid = self.get_batch(idx)
                break

            except Exception:
                zero_rank_print("Error loading video, retrying...")
                idx = random.randint(0, self.length - 1)

        video = self.pixel_transforms(video)
        video_ = video.clone().permute(0, 2, 3, 1).numpy() / 2 + 0.5
        video_ = video_ * 255
        score = get_score(video_, cond_frame_idx=cond_frames)
        del video_
        sample = {"pixel_values": video, "text": name, "score": score, "cond_frames": cond_frames, "vid": videoid}
        return sample
