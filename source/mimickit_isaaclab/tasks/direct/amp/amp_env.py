"""AMP (Adversarial Motion Priors) environment implementation.

AMP extends DeepMimic with a discriminator that learns to distinguish between
real motion data and policy-generated motions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..deepmimic.deepmimic_env import DeepMimicEnv
from ....utils.circular_buffer import CircularBuffer
from ....utils import torch_util

if TYPE_CHECKING:
    from .amp_env_cfg import AMPEnvCfg


class AMPEnv(DeepMimicEnv):
    """AMP environment for learning adversarial motion priors.
    
    This environment extends DeepMimic by adding discriminator observations
    that capture the style and characteristics of reference motions.
    """

    cfg: AMPEnvCfg

    def __init__(self, cfg: AMPEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize AMP environment.
        
        Args:
            cfg: Configuration for AMP environment
            render_mode: Render mode for the environment
            **kwargs: Additional keyword arguments
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # AMP-specific initialization
        self._num_disc_obs_steps = cfg.num_disc_obs_steps
        self._build_disc_obs_buffers()

    def _build_disc_obs_buffers(self):
        """Build circular buffers for discriminator observations."""
        num_envs = self.num_envs
        n = self._num_disc_obs_steps
        
        # Get sample data shapes
        root_pos_shape = (3,)
        root_rot_shape = (4,)
        root_vel_shape = (3,)
        root_ang_vel_shape = (3,)
        joint_rot_shape = (self.character.num_bodies - 1, 4)  # Exclude root
        dof_vel_shape = (self.character.num_joints,)
        
        # Create circular buffers for history
        self._disc_hist_root_pos = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_pos_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        self._disc_hist_root_rot = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_rot_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        self._disc_hist_root_vel = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_vel_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        self._disc_hist_root_ang_vel = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_ang_vel_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        self._disc_hist_joint_rot = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=joint_rot_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        self._disc_hist_dof_vel = CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=dof_vel_shape,
            dtype=torch.float32,
            device=self.device,
        )
        
        # Optional: key body positions
        if self._has_key_bodies():
            num_key_bodies = len(self._key_body_ids)
            self._disc_hist_key_pos = CircularBuffer(
                batch_size=num_envs,
                buffer_len=n,
                shape=(num_key_bodies, 3),
                dtype=torch.float32,
                device=self.device,
            )

    def _post_physics_step(self, actions: torch.Tensor):
        """Post-physics step processing.
        
        Args:
            actions: Actions to apply
        """
        super()._post_physics_step(actions)
        
        # Update discriminator observation history
        self._update_disc_obs()

    def _update_disc_obs(self):
        """Update discriminator observation buffers with current state."""
        # Get current state
        root_pos = self.character.data.root_pos_w
        root_rot = self.character.data.root_quat_w
        root_vel = self.character.data.root_lin_vel_w
        root_ang_vel = self.character.data.root_ang_vel_w
        
        # Get joint rotations (body rotations excluding root)
        body_rot = self.character.data.body_quat_w
        joint_rot = body_rot[:, 1:, :]  # Exclude root
        
        # Get DOF velocities
        dof_vel = self.character.data.joint_vel
        
        # Push to circular buffers
        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_joint_rot.push(joint_rot)
        self._disc_hist_dof_vel.push(dof_vel)
        
        # Optional: key body positions
        if self._has_key_bodies():
            body_pos = self.character.data.body_pos_w
            key_pos = body_pos[:, self._key_body_ids, :]
            self._disc_hist_key_pos.push(key_pos)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        super()._reset_idx(env_ids)
        
        # Reset discriminator history for these environments
        if len(env_ids) > 0:
            self._reset_disc_hist(env_ids)

    def _reset_disc_hist(self, env_ids: torch.Tensor):
        """Reset discriminator history buffers for specific environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        # Get reference motion data for these environments
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)
        
        # Fetch historical motion data
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = \
            self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        # Fill buffers with motion data
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_joint_rot.fill(env_ids, joint_rot)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)
        
        if self._has_key_bodies():
            self._disc_hist_key_pos.fill(env_ids, key_pos)

    def _fetch_disc_demo_data(self, motion_ids: torch.Tensor, motion_times0: torch.Tensor):
        """Fetch historical motion data for discriminator.
        
        Args:
            motion_ids: Motion IDs for each environment
            motion_times: Starting time for each environment
            
        Returns:
            Tuple of (root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos)
        """
        num_samples = motion_ids.shape[0]
        
        # Create time steps going backwards
        motion_ids_exp = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self.step_dt * torch.arange(0, self._num_disc_obs_steps, device=self.device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps
        
        # Flatten for batch processing
        motion_ids_flat = motion_ids_exp.view(-1)
        motion_times_flat = motion_times.view(-1)
        
        # Get motion frames
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(motion_ids_flat, motion_times_flat)
        
        # Compute key body positions if needed
        if self._has_key_bodies():
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self.device)
        
        # Reshape to [num_samples, num_disc_obs_steps, ...]
        root_pos = root_pos.reshape(num_samples, self._num_disc_obs_steps, -1)
        root_rot = root_rot.reshape(num_samples, self._num_disc_obs_steps, -1)
        root_vel = root_vel.reshape(num_samples, self._num_disc_obs_steps, -1)
        root_ang_vel = root_ang_vel.reshape(num_samples, self._num_disc_obs_steps, -1)
        joint_rot = joint_rot.reshape(num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], -1)
        dof_vel = dof_vel.reshape(num_samples, self._num_disc_obs_steps, -1)
        
        if self._has_key_bodies():
            key_pos = key_pos.reshape(num_samples, self._num_disc_obs_steps, key_pos.shape[-2], -1)
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos

    def fetch_disc_obs_demo(self, num_samples: int) -> torch.Tensor:
        """Fetch discriminator observations from demonstration data.
        
        Args:
            num_samples: Number of samples to fetch
            
        Returns:
            Discriminator observations from demo data
        """
        motion_ids, motion_times0 = self._sample_motion_times(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs

    def _compute_disc_obs_demo(self, motion_ids: torch.Tensor, motion_times0: torch.Tensor) -> torch.Tensor:
        """Compute discriminator observations from demonstration data.
        
        Args:
            motion_ids: Motion IDs
            motion_times0: Motion times
            
        Returns:
            Discriminator observations
        """
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = \
            self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        # Use the last frame as reference
        if self._track_global_root():
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1  # Identity quaternion [0,0,0,1]
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
        
        disc_obs = self._compute_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
        )
        return disc_obs

    def _compute_disc_obs(
        self,
        ref_root_pos: torch.Tensor,
        ref_root_rot: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_rot: torch.Tensor,
        dof_vel: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator observations.
        
        This creates a representation of the motion that the discriminator can use
        to distinguish between real and generated motions.
        
        Args:
            ref_root_pos: Reference root position
            ref_root_rot: Reference root rotation  
            root_pos: Root positions over time
            root_rot: Root rotations over time
            root_vel: Root velocities over time
            root_ang_vel: Root angular velocities over time
            joint_rot: Joint rotations over time
            dof_vel: DOF velocities over time
            key_pos: Key body positions over time
            
        Returns:
            Discriminator observations
        """
        # Compute position observations (same as target observations in DeepMimic)
        pos_obs = self._compute_tar_obs_with_time(
            ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos
        )
        
        # Compute velocity observations
        vel_obs = self._compute_disc_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel)
        
        # Concatenate position and velocity observations
        disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
        
        return disc_obs

    def _compute_tar_obs_with_time(
        self,
        ref_root_pos: torch.Tensor,
        ref_root_rot: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        joint_rot: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target observations over multiple time steps.
        
        Args:
            ref_root_pos: Reference root position [batch_size, 3]
            ref_root_rot: Reference root rotation [batch_size, 4]
            root_pos: Root positions [batch_size, num_steps, 3]
            root_rot: Root rotations [batch_size, num_steps, 4]
            joint_rot: Joint rotations [batch_size, num_steps, num_joints, 4]
            key_pos: Key body positions [batch_size, num_steps, num_key_bodies, 3]
            
        Returns:
            Target observations [batch_size, obs_dim]
        """
        # Flatten time dimension for processing
        batch_size = root_pos.shape[0]
        num_steps = root_pos.shape[1]
        
        root_pos_flat = root_pos.reshape(-1, 3)
        root_rot_flat = root_rot.reshape(-1, 4)
        joint_rot_flat = joint_rot.reshape(-1, joint_rot.shape[-2], 4)
        
        if self._has_key_bodies():
            key_pos_flat = key_pos.reshape(-1, key_pos.shape[-2], 3)
        else:
            key_pos_flat = torch.zeros([0], device=self.device)
        
        # Expand reference for each time step
        ref_root_pos_exp = ref_root_pos.unsqueeze(1).repeat(1, num_steps, 1).reshape(-1, 3)
        ref_root_rot_exp = ref_root_rot.unsqueeze(1).repeat(1, num_steps, 1).reshape(-1, 4)
        
        # Compute observations for each time step
        obs_flat = self._compute_tar_obs(
            ref_root_pos_exp, ref_root_rot_exp,
            root_pos_flat, root_rot_flat,
            joint_rot_flat, key_pos_flat
        )
        
        # Reshape and flatten time dimension
        obs = obs_flat.reshape(batch_size, num_steps, -1)
        obs = obs.reshape(batch_size, -1)  # Flatten time steps
        
        return obs

    def _compute_disc_vel_obs(
        self,
        ref_root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity observations for discriminator.
        
        Args:
            ref_root_rot: Reference root rotation [batch_size, 4]
            root_vel: Root velocities [batch_size, num_steps, 3]
            root_ang_vel: Root angular velocities [batch_size, num_steps, 3]
            dof_vel: DOF velocities [batch_size, num_steps, num_dof]
            
        Returns:
            Velocity observations [batch_size, vel_obs_dim]
        """
        if not self._global_obs:
            # Transform velocities to heading frame
            heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
            heading_inv_rot_exp = heading_inv_rot.unsqueeze(-2)
            heading_inv_rot_exp = heading_inv_rot_exp.repeat((1, root_vel.shape[1], 1))
            heading_inv_rot_flat = heading_inv_rot_exp.reshape(-1, 4)
            
            root_vel_flat = root_vel.reshape(-1, 3)
            root_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_vel_flat)
            root_vel_obs = root_vel_obs_flat.reshape(root_vel.shape)
            
            root_ang_vel_flat = root_ang_vel.reshape(-1, 3)
            root_ang_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_ang_vel_flat)
            root_ang_vel_obs = root_ang_vel_obs_flat.reshape(root_ang_vel.shape)
        else:
            root_vel_obs = root_vel
            root_ang_vel_obs = root_ang_vel
        
        # Concatenate and flatten
        obs = torch.cat([root_vel_obs, root_ang_vel_obs, dof_vel], dim=-1)
        obs = obs.reshape(obs.shape[0], -1)  # Flatten time steps
        
        return obs

    def get_disc_obs_size(self) -> int:
        """Get the size of discriminator observations.
        
        Returns:
            Size of discriminator observations
        """
        # Get a sample observation
        disc_obs = self.fetch_disc_obs_demo(1)
        return disc_obs.shape[-1]

