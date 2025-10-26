# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Motion library for handling motion data and sampling.

import numpy as np
import os
import torch
import yaml
from typing import Optional, Tuple

from .motion import Motion, LoopMode, load_motion
from ..utils import torch_util
from ..utils.logger import Logger


def extract_pose_data(frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract pose data from a motion frame.
    
    Args:
        frame: Motion frame tensor with shape (..., n), where n >= 6
            Format: [root_pos (3), root_rot (3), joint_dof (...)]
    
    Returns:
        Tuple of (root_pos, root_rot, joint_dof)
    """
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof


class MotionLib:
    """Motion library for handling motion data, sampling, and interpolation.
    
    This class is compatible with Isaac Lab and handles motion data loading,
    sampling, and frame interpolation for character animation.
    """

    def __init__(self, motion_file: str, kin_char_model, device: str):
        """Initialize the motion library.
        
        Args:
            motion_file: Path to motion file or dataset configuration
            kin_char_model: Kinematic character model for DoF/rotation conversions
            device: Device to store tensors on (e.g., "cuda:0" or "cpu")
        """
        self._device = device
        self._kin_char_model = kin_char_model
        self._load_motions(motion_file)

    def get_num_motions(self) -> int:
        """Get the number of motions in the library."""
        return self._motion_lengths.shape[0]

    def get_total_length(self) -> float:
        """Get the total length of all motions in seconds."""
        return torch.sum(self._motion_lengths).item()

    def get_motion(self, motion_id: int) -> Motion:
        """Get a specific motion by ID."""
        return self._motions[motion_id]

    def sample_motions(self, n: int) -> torch.Tensor:
        """Sample motion IDs using weighted sampling.
        
        Args:
            n: Number of motions to sample
            
        Returns:
            Tensor of shape (n,) with motion IDs
        """
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None) -> torch.Tensor:
        """Sample random time points within motions.
        
        Args:
            motion_ids: Tensor of motion IDs
            truncate_time: Optional time to truncate from end of motion
            
        Returns:
            Tensor of shape (n,) with sampled time points
        """
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the length of motions."""
        return self._motion_lengths[motion_ids]
    
    def get_motion_loop_mode(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the loop mode of motions."""
        return self._motion_loop_modes[motion_ids]
    
    def calc_motion_phase(self, motion_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Calculate motion phase (0-1) at given times.
        
        Args:
            motion_ids: Tensor of motion IDs
            times: Tensor of time points
            
        Returns:
            Tensor of phases in range [0, 1]
        """
        motion_len = self._motion_lengths[motion_ids]
        loop_mode = self._motion_loop_modes[motion_ids]
        phase = calc_phase(times=times, motion_len=motion_len, loop_mode=loop_mode)
        return phase

    def calc_motion_frame(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Calculate interpolated motion frame at given times.
        
        Args:
            motion_ids: Tensor of motion IDs
            motion_times: Tensor of time points
            
        Returns:
            Tuple of (root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel)
        """
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._frame_root_pos[frame_idx0]
        root_pos1 = self._frame_root_pos[frame_idx1]
        
        root_rot0 = self._frame_root_rot[frame_idx0]
        root_rot1 = self._frame_root_rot[frame_idx1]

        root_vel = self._frame_root_vel[frame_idx0]
        root_ang_vel = self._frame_root_ang_vel[frame_idx0]

        joint_rot0 = self._frame_joint_rot[frame_idx0]
        joint_rot1 = self._frame_joint_rot[frame_idx1]

        dof_vel = self._frame_dof_vel[frame_idx0]

        blend_unsq = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsq) * root_pos0 + blend_unsq * root_pos1
        root_rot = torch_util.slerp(root_rot0, root_rot1, blend)
        
        joint_rot = torch_util.slerp(joint_rot0, joint_rot1, blend_unsq)

        root_pos_offset = self._calc_loop_offset(motion_ids, motion_times)
        root_pos += root_pos_offset

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel

    def joint_rot_to_dof(self, joint_rot: torch.Tensor) -> torch.Tensor:
        """Convert joint rotations to DoF representation."""
        joint_dof = self._kin_char_model.rot_to_dof(joint_rot)
        return joint_dof

    def get_motion_lengths(self) -> torch.Tensor:
        """Get all motion lengths."""
        return self._motion_lengths

    def get_motion_weights(self) -> torch.Tensor:
        """Get all motion weights."""
        return self._motion_weights
    
    def _extract_frame_data(
        self, frame: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and convert frame data to tensors."""
        root_pos, root_rot, joint_dof = extract_pose_data(frame)
        root_pos = torch.tensor(root_pos, dtype=torch.float32, device=self._device)
        root_rot = torch.tensor(root_rot, dtype=torch.float32, device=self._device)
        joint_dof = torch.tensor(joint_dof, dtype=torch.float32, device=self._device)

        root_rot_quat = torch_util.exp_map_to_quat(root_rot)

        joint_rot = self._kin_char_model.dof_to_rot(joint_dof)
        joint_rot = torch_util.quat_pos(joint_rot)

        return root_pos, root_rot_quat, joint_rot

    def _calc_frame_blend(
        self, motion_ids: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate frame indices and blend weights for interpolation."""
        num_frames = self._motion_num_frames[motion_ids]
        frame_start_idx = self._motion_start_idx[motion_ids]
        
        phase = self.calc_motion_phase(motion_ids, times)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0
        
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx
        

        return frame_idx0, frame_idx1, blend
    
    def _calc_loop_offset(self, motion_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Calculate root position offset for wrapping motions."""
        loop_mode = self._motion_loop_modes[motion_ids]
        wrap_mask = (loop_mode == LoopMode.WRAP.value)

        wrap_motion_ids = motion_ids[wrap_mask]
        times = times[wrap_mask]

        motion_len = self._motion_lengths[wrap_motion_ids]
        root_pos_deltas = self._motion_root_pos_delta[wrap_motion_ids]

        phase = times / motion_len
        phase = torch.floor(phase)
        phase = phase.unsqueeze(-1)
        
        root_pos_offset = torch.zeros((motion_ids.shape[0], 3), device=self._device)
        root_pos_offset[wrap_mask] = phase * root_pos_deltas

        return root_pos_offset

    def _load_motions(self, motion_file: str):
        """Load motions from file."""
        self._load_motion_pkl(motion_file)
        
        num_motions = self.get_num_motions()
        total_len = self.get_total_length()
        Logger.print(f"Loaded motion file: {motion_file}")
        Logger.print(f"Loaded {num_motions} motions with a total length of {total_len:.3f}s.")

    def _load_motion_pkl(self, motion_file: str):
        """Load motion data from pickle files."""
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_loop_modes = []
        self._motion_root_pos_delta = []
        self._motion_files = []
        self._motions = []
        
        self._frame_root_pos = []
        self._frame_root_rot = []
        self._frame_root_vel = []
        self._frame_root_ang_vel = []
        self._frame_joint_rot = []
        self._frame_dof_vel = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            Logger.print(f"Loading {f + 1}/{num_motion_files} motion files: {curr_file}")
            
            curr_motion = load_motion(curr_file)
            fps = curr_motion.fps
            loop_mode = curr_motion.loop_mode
            frames = curr_motion.frames
            curr_weight = motion_weights[f]

            loop_mode = loop_mode.value
            dt = 1.0 / fps

            num_frames = frames.shape[0]
            curr_len = 1.0 / fps * (num_frames - 1)

            root_pos, root_rot, joint_rot = self._extract_frame_data(frames)
            root_pos_delta = root_pos[-1] - root_pos[0]
            root_pos_delta[..., -1] = 0.0

            root_vel = torch.zeros_like(root_pos)
            root_vel[..., :-1, :] = fps * (root_pos[..., 1:, :] - root_pos[..., :-1, :])
            root_vel[..., -1, :] = root_vel[..., -2, :]
                
            root_ang_vel = torch.zeros_like(root_pos)
            root_drot = torch_util.quat_diff(root_rot[..., :-1, :], root_rot[..., 1:, :])
            root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
            root_ang_vel[..., -1, :] = root_ang_vel[..., -2, :]

            dof_vel = self._kin_char_model.compute_frame_dof_vel(joint_rot, dt)

            self._motion_weights.append(curr_weight)
            self._motion_fps.append(fps)
            self._motion_dt.append(dt)
            self._motion_num_frames.append(num_frames)
            self._motion_lengths.append(curr_len)
            self._motion_loop_modes.append(loop_mode)
            self._motion_root_pos_delta.append(root_pos_delta)
            self._motion_files.append(curr_file)
            self._motions.append(curr_motion)
                
            self._frame_root_pos.append(root_pos)
            self._frame_root_rot.append(root_rot)
            self._frame_root_vel.append(root_vel)
            self._frame_root_ang_vel.append(root_ang_vel)
            self._frame_joint_rot.append(joint_rot)
            self._frame_dof_vel.append(dof_vel)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_loop_modes = torch.tensor(self._motion_loop_modes, dtype=torch.int, device=self._device)
        
        self._motion_root_pos_delta = torch.stack(self._motion_root_pos_delta, dim=0)
        
        self._frame_root_pos = torch.cat(self._frame_root_pos, dim=0)
        self._frame_root_rot = torch.cat(self._frame_root_rot, dim=0)
        self._frame_root_vel = torch.cat(self._frame_root_vel, dim=0)
        self._frame_root_ang_vel = torch.cat(self._frame_root_ang_vel, dim=0)
        self._frame_joint_rot = torch.cat(self._frame_joint_rot, dim=0)
        self._frame_dof_vel = torch.cat(self._frame_dof_vel, dim=0)
        
        num_motions = self.get_num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)

    def _fetch_motion_files(self, motion_file: str) -> Tuple[list, list]:
        """Fetch motion files and weights from configuration."""
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            motion_files = []
            motion_weights = []

            with open(motion_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert curr_weight >= 0

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights


@torch.jit.script
def calc_phase(times: torch.Tensor, motion_len: torch.Tensor, loop_mode: torch.Tensor) -> torch.Tensor:
    """Calculate motion phase (0-1) considering loop mode.
    
    Args:
        times: Time points in the motion
        motion_len: Length of each motion
        loop_mode: Loop mode for each motion
        
    Returns:
        Phase values in range [0, 1]
    """
    phase = times / motion_len
        
    # Handle wrapping motions
    loop_wrap_mask = (loop_mode == LoopMode.WRAP.value)
    phase_wrap = phase[loop_wrap_mask]
    phase_wrap = phase_wrap - torch.floor(phase_wrap)
    phase[loop_wrap_mask] = phase_wrap
        
    phase = torch.clip(phase, 0.0, 1.0)

    return phase

