# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# G1-specific DeepMimic environment.

from __future__ import annotations

import os
import torch
import logging

from .deepmimic_env import DeepMimicEnv
from .deepmimic_env_cfg import DeepMimicEnvCfg
from ....anim import KinCharModelG1, MotionLib

# Setup logger
logger = logging.getLogger(__name__)


class DeepMimicG1Env(DeepMimicEnv):
    """DeepMimic environment for G1 robot with G1 motion data.
    
    This environment extends the base DeepMimicEnv to support the Unitree G1
    robot with 29 DOF. It handles the retargeting from SMPL motion data to
    G1's specific joint configuration.
    
    Key differences from base DeepMimicEnv:
    - Uses KinCharModelG1 for G1's 29 DOF structure
    - Uses standard MotionLib for G1 motion data loading
    - Maps G1's 29 DOF directly to USD's 29 DOF joint order
    """
    
    cfg: DeepMimicEnvCfg
    
    def __init__(self, cfg: DeepMimicEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the DeepMimic G1 environment.
        
        Args:
            cfg: Configuration for the environment
            render_mode: Rendering mode ("human", "rgb_array", or None)
            **kwargs: Additional arguments passed to parent class
        """
        # Store configuration
        self.cfg = cfg
        self._device = cfg.sim.device
        
        # Create G1 kinematic model BEFORE calling parent __init__
        # This way when parent creates MotionLib, it will use our G1 model
        self._kin_char_model = KinCharModelG1(self._device)
        
        # Initialize parent class (this will call _setup_scene and create robot)
        # Parent will try to create its own kin_char_model but we override it first
        # However, parent will also create motion_lib using whatever _kin_char_model exists
        from isaaclab.envs import DirectRLEnv
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        
        # After robot is created, get joint information from USD
        self._action_size = self.character.num_joints
        # Note: cfg.action_space will be handled by parent DirectRLEnv
        
        # Build DOF mapping with G1 model
        self._build_dof_mapping()
        
        # Load motion library with G1 kinematic model
        self._motion_lib = MotionLib(
            motion_file=cfg.motion_file,
            kin_char_model=self._kin_char_model,
            device=self.device
        )
        
        # Setup additional buffers (from parent class)
        self._setup_motion_buffers()
        self._setup_reward_buffers()
        
        # Parse joint error weights
        self._parse_joint_err_weights()
        
        # Parse other configuration
        self._parse_config()
        
        print(f"[INFO] DeepMimic G1 environment initialized with {self.num_envs} environments")
        print(f"[INFO] Loaded {self._motion_lib.get_num_motions()} motions")
        print(f"[INFO] Action space: {self._action_size} DoFs")
        
    def _setup_scene(self):
        """Setup scene with G1 robot.
        
        Uses parent's implementation but with G1 configuration.
        """
        super()._setup_scene()
        
    def _build_dof_mapping(self):
        """Build mapping from G1 motion DOF (29) to USD DOF order (29).
        
        Motion data order (MimicKit G1):
        [0-5]   Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        [6-11]  Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        [12-14] Waist: yaw, roll, pitch
        [15-21] Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        [22-28] Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        
        USD order (Isaac Lab G1):
        [0-1]   Hip pitch: left, right
        [2]     Waist yaw
        [3-4]   Hip roll: left, right
        [5]     Waist roll
        [6-7]   Hip yaw: left, right
        [8]     Waist pitch
        [9-10]  Knee: left, right
        [11-12] Shoulder pitch: left, right
        [13-14] Ankle pitch: left, right
        [15-16] Shoulder roll: left, right
        [17-18] Ankle roll: left, right
        [19-20] Shoulder yaw: left, right
        [21-22] Elbow: left, right
        [23-24] Wrist roll: left, right
        [25-26] Wrist pitch: left, right
        [27-28] Wrist yaw: left, right
        """
        # Get actual USD joint names
        usd_joint_names = list(self.character.data.joint_names)
        
        # Motion data joint names (in motion file order)
        motion_joint_names = self._kin_char_model.G1_29DOF_JOINT_NAMES
        
        # Create mapping: motion_idx -> usd_idx
        motion_to_usd_indices = []
        for motion_name in motion_joint_names:
            if motion_name in usd_joint_names:
                usd_idx = usd_joint_names.index(motion_name)
                motion_to_usd_indices.append(usd_idx)
            else:
                raise ValueError(f"Motion joint '{motion_name}' not found in USD model!")
        
        self._g1_to_usd_dof_indices = torch.tensor(
            motion_to_usd_indices,
            dtype=torch.long,
            device=self.device
        )
        
        logger.info(f"DOF mapping created: {len(motion_to_usd_indices)} joints mapped")
        
    def _map_dof_g1_to_usd(self, dof_g1: torch.Tensor) -> torch.Tensor:
        """Convert DOF values from G1 motion (29) to USD order (29).
        
        Uses name-based mapping to reorder DOF values.
        
        Args:
            dof_g1: G1 motion DOF values, shape (..., 29)
            
        Returns:
            USD DOF values, shape (..., 29)
        """
        # Create output tensor in USD order
        dof_usd = torch.zeros_like(dof_g1)
        
        # Place each motion DOF value into its corresponding USD position
        # self._g1_to_usd_dof_indices[i] = j means: motion[i] -> usd[j]
        for motion_idx in range(dof_g1.shape[-1]):
            usd_idx = self._g1_to_usd_dof_indices[motion_idx]
            dof_usd[..., usd_idx] = dof_g1[..., motion_idx]
        
        return dof_usd
    
    def _ref_state_init(self, env_ids: torch.Tensor):
        """Initialize character state from reference motion.
        
        Overrides parent to use G1-specific DOF mapping.
        """
        # Set root state
        root_state = self.character.data.default_root_state[env_ids].clone()
        root_pos = self._ref_root_pos[env_ids]
        root_rot = self._ref_root_rot[env_ids]
        root_vel = self._ref_root_vel[env_ids]
        root_ang_vel = self._ref_root_ang_vel[env_ids]
        
        # Convert quaternion from MimicKit format [x,y,z,w] to Isaac Lab format [w,x,y,z]
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        
        root_state[:, :3] = root_pos
        # Add environment origins to maintain env_spacing during reset
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_rot_wxyz
        root_state[:, 7:10] = root_vel
        root_state[:, 10:13] = root_ang_vel
        
        # Convert joint positions from G1 retargeted DOF to USD order
        joint_pos_g1 = self._ref_dof_pos[env_ids]
        joint_pos_usd = self._map_dof_g1_to_usd(joint_pos_g1)
        
        # Convert velocities
        joint_vel_g1 = self._ref_dof_vel[env_ids]
        joint_vel_usd = self._map_dof_g1_to_usd(joint_vel_g1)
        
        # Write to simulation
        self.character.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.character.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.character.write_joint_state_to_sim(joint_pos_usd, joint_vel_usd, None, env_ids)
    
    def _reset_ref_char(self, env_ids: torch.Tensor):
        """Reset reference character visualization.
        
        Overrides parent to use G1-specific DOF mapping.
        """
        if not self._enable_ref_char():
            return
            
        # Set root state with offset
        root_state = self.ref_character.data.default_root_state[env_ids].clone()
        root_pos = self._ref_root_pos[env_ids] + self._ref_char_offset
        root_rot = self._ref_root_rot[env_ids]
        
        # Convert quaternion format
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        
        root_state[:, :3] = root_pos
        # Add environment origins to maintain env_spacing during reset
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_rot_wxyz
        root_state[:, 7:] = 0.0
        
        # Convert joint positions
        joint_pos_g1 = self._ref_dof_pos[env_ids]
        joint_pos_usd = self._map_dof_g1_to_usd(joint_pos_g1)
        joint_vel = torch.zeros_like(joint_pos_usd)
        
        # Write to simulation
        self.ref_character.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.ref_character.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.ref_character.write_joint_state_to_sim(joint_pos_usd, joint_vel, None, env_ids)
    
    def _update_ref_char_all(self):
        """Update reference character visualization for all environments.
        
        Overrides parent to use G1-specific DOF mapping.
        """
        if not self._enable_ref_char():
            return
            
        # Set root state with offset
        root_pos = self._ref_root_pos + self._ref_char_offset
        root_rot = self._ref_root_rot
        
        # Convert quaternion format
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        
        # Convert joint positions - G1 motion order to USD order
        joint_pos_motion = self._ref_dof_pos  # Motion data order
        joint_pos_usd = self._map_dof_g1_to_usd(joint_pos_motion)  # USD order
        
        # Write to simulation
        self.ref_character.write_root_pose_to_sim(
            torch.cat([root_pos, root_rot_wxyz], dim=-1), env_ids=None
        )
        self.ref_character.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=None
        )
        self.ref_character.write_joint_state_to_sim(
            joint_pos_usd,
            torch.zeros_like(joint_pos_usd),
            None,
            env_ids=None
        )

