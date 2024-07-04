import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
from torchvision.io import read_video

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x





def load_video_frames(video_path):
    """
    Load video frames from a file and convert them to grayscale.
    """
    # 使用torchvision读取视频文件
    video, _, _ = read_video(video_path, pts_unit='sec')
    # Change from TCHW (time, channel, height, width) to THWC for easier handling
    video = video.permute(0, 2, 3, 1).numpy()
    # Convert RGB frames to grayscale
    video_gray = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video])
    # 确保维度是(N, H, W, 1)形式，便于后续处理
    video_gray = video_gray[..., np.newaxis]

    return video_gray


class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_height=224, frame_width=224, patch_size=16, embed_dim=768):
        self.video_paths = video_paths
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=frame_height, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_frames = load_video_frames(self.video_paths[idx])
        neighbor_encoded = self.neighbor_sequence_encoding(video_frames)
        stochastic_encoded = self.stochastic_frame_grab(video_frames)

        combined_data = []
        stochastic_index = 0
        for i in range(0, len(neighbor_encoded), 3):
            # Add three consecutive neighbor encoded data units
            combined_data.extend(neighbor_encoded[i:i + 3])
            # Add one stochastic encoded data unit
            if stochastic_index < len(stochastic_encoded):
                combined_data.append(stochastic_encoded[stochastic_index])
                stochastic_index += 1

        return torch.tensor(combined_data)

    def neighbor_sequence_encoding(self, video_frames):
        N, H, W, C = video_frames.shape
        num_data_units = N // 3
        encoded_matrices = []
        for i in range(num_data_units):
            frames = video_frames[i*3:(i+3)*3]
            data_unit = torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
            data_unit_patched = self.patch_embed(data_unit.float())
            encoded_matrices.append(data_unit_patched.squeeze(0).detach().numpy())
        return np.array(encoded_matrices)


    def stochastic_frame_grab(self, video_frames):
        N, H, W, C = video_frames.shape
        num_data_units = N // 3
        encoded_matrices = []
        sampled_indices = random.sample(range(N), num_data_units * 3)
        for i in range(num_data_units):
            indices = sampled_indices[i*3:(i+1)*3]
            frames = video_frames[indices]
            data_unit = torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
            data_unit_patched = self.patch_embed(data_unit.float())
            encoded_matrices.append(data_unit_patched.squeeze(0).detach().numpy())
        return np.array(encoded_matrices)

# Example usage:
# dataset = VideoDataset(['path_to_video1.mp4', 'path_to_video2.mp4'])
# encoded_frames, random_encoded_frames = dataset[0]
