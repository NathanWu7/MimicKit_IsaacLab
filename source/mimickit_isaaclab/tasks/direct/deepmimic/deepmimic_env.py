# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# DeepMimic environment for Isaac Lab using Direct workflow.

from __future__ import annotations

import os
import torch
import numpy as np
import logging
from typing import Tuple
from collections.abc import Sequence

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/mimickit_deepmimic_env.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_mul, quat_conjugate

from .deepmimic_env_cfg import DeepMimicEnvCfg
from ....anim import MotionLib, KinCharModel
from ....anim.motion import LoopMode
from ....utils import torch_util


class DeepMimicEnv(DirectRLEnv):
    """DeepMimic environment for motion imitation using Isaac Lab Direct workflow.
    
    This environment implements the DeepMimic algorithm for physics-based character animation.
    It supports both humanoid and quadruped characters with motion tracking rewards.
    """

    cfg: DeepMimicEnvCfg

    def __init__(self, cfg: DeepMimicEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the DeepMimic environment.
        
        Args:
            cfg: Configuration for the environment
            render_mode: Rendering mode ("human", "rgb_array", or None)
            **kwargs: Additional arguments passed to parent class
        """
        # Store configuration
        self.cfg = cfg
        
        # Initialize parent class first (this will call _setup_scene and create robot)
        super().__init__(cfg, render_mode, **kwargs)
        
        # After robot is created, get joint information from USD
        self._device = cfg.sim.device
        
        # Get action size from the actual robot
        self._action_size = self.character.num_joints
        cfg.action_space = self._action_size
        
        # Create kinematic character model from XML
        self._kin_char_model = KinCharModel(self._device)
        char_xml_path = os.path.join("data", "assets", "smpl", "smpl.xml")
        self._kin_char_model.load_char_file(char_xml_path)
        
        # Build DOF mapping from XML to USD
        self._build_dof_mapping()
        
        # Load motions
        self._motion_lib = MotionLib(
            motion_file=cfg.motion_file,
            kin_char_model=self._kin_char_model,
            device=self.device
        )
        
        # Setup additional buffers
        self._setup_motion_buffers()
        self._setup_reward_buffers()
        
        # Parse configuration
        self._parse_config()
        
        print(f"[INFO] DeepMimic environment initialized with {self.num_envs} environments")
        print(f"[INFO] Loaded {self._motion_lib.get_num_motions()} motions")
        print(f"[INFO] Action space: {self._action_size} DoFs")

    def _setup_scene(self):
        """Setup the scene with character and ground plane.
        
        This is called during environment initialization.
        """
        # In replay mode, we only create reference character
        # In normal mode, we create both controllable and (optionally) reference character
        
        if not self.cfg.replay_mode:
            # Normal mode: create controllable character
            self.character = Articulation(self.cfg.char_asset)
        
        # Add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=self.cfg.sim.physics_material
            )
        )
        
        # Create reference character if needed
        if self._enable_ref_char() or self.cfg.replay_mode:
            # In replay mode, reference character is the main character
            if self.cfg.replay_mode:
                # Move reference character to center
                ref_char_cfg = self.cfg.char_asset.replace(prim_path="/World/envs/env_.*/RefCharacter")
                ref_char_cfg = ref_char_cfg.replace(
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, self.cfg.char_asset.init_state.pos[2]),  # Use same height
                        joint_pos=self.cfg.char_asset.init_state.joint_pos,
                        joint_vel=self.cfg.char_asset.init_state.joint_vel,
                    )
                )
                self.ref_character = Articulation(ref_char_cfg)
                # Also create a dummy character for env to work
                self.character = self.ref_character  # Use ref as main character
            else:
                # Normal mode: reference character with offset
                ref_char_cfg = self.cfg.char_asset.replace(prim_path="/World/envs/env_.*/RefCharacter")
                self.ref_character = Articulation(ref_char_cfg)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions to prevent inter-environment collisions
        # This ensures that objects in different environments don't collide with each other
        self.scene.filter_collisions(global_prim_paths=[])
        
        # Add articulations to scene
        self.scene.articulations["character"] = self.character
        if (self._enable_ref_char() or self.cfg.replay_mode) and not self.cfg.replay_mode:
            # Don't add ref_character separately in replay mode (it's the same as character)
            self.scene.articulations["ref_character"] = self.ref_character
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_character_cfg(self):
        """Create character articulation configuration from XML file."""
        # Note: In a full implementation, you would need to convert the XML
        # character file to USD or use a converter. For now, we assume
        # the character asset is already in USD format or we have a converter.
        
        # This is a simplified version - in production you'd load from USD
        self.cfg.char_asset = ArticulationCfg(
            prim_path="/World/envs/env_.*/Character",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self._convert_xml_to_usd(self.cfg.char_file),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                joint_pos={".*": 0.0},
            ),
            actuators={
                "joints": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self._get_joint_stiffness(),
                    damping=self._get_joint_damping(),
                ),
            },
        )

    def _create_ref_char_cfg(self) -> ArticulationCfg:
        """Create reference character configuration."""
        cfg = self.cfg.char_asset.__class__(
            prim_path="/World/envs/env_.*/RefCharacter",
            spawn=self.cfg.char_asset.spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(self.cfg.ref_char_offset[0], self.cfg.ref_char_offset[1], 1.0),
                joint_pos={".*": 0.0},
            ),
            actuators={}  # No actuators for reference character
        )
        return cfg

    def _parse_config(self):
        """Parse environment configuration."""
        # Key body IDs
        if len(self.cfg.key_bodies) > 0:
            self._key_body_ids = self._get_body_ids(self.cfg.key_bodies)
        else:
            self._key_body_ids = torch.tensor([], device=self.device, dtype=torch.long)
        
        # Contact body IDs
        if len(self.cfg.contact_bodies) > 0:
            self._contact_body_ids = self._get_body_ids(self.cfg.contact_bodies)
        else:
            self._contact_body_ids = torch.tensor([], device=self.device, dtype=torch.long)
        
        # Joint error weights
        self._parse_joint_err_weights()
        
        # Reference character offset (X, Y, Z) for visualization
        self._ref_char_offset = torch.tensor(
            self.cfg.ref_char_offset, device=self.device, dtype=torch.float32
        )
        
        # Add height offset to prevent ground collision for reference character
        # This is ONLY for visualization - rewards automatically compensate
        self._ref_char_height_offset = self.cfg.ref_char_height_offset
        self._ref_char_offset[2] += self._ref_char_height_offset  # Add to Z component
        
        logger.info(f"Reference character offset: {self._ref_char_offset.cpu().numpy()}")
        logger.info(f"Height offset for collision avoidance: {self._ref_char_height_offset}m")
        
        # Target observation steps
        self._tar_obs_steps = torch.tensor(
            self.cfg.tar_obs_steps, device=self.device, dtype=torch.int
        )

    def _setup_motion_buffers(self):
        """Setup buffers for motion tracking."""
        num_envs = self.num_envs
        
        # Motion IDs and time offsets
        self._motion_ids = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        
        # Reference state buffers (will be sized after motion lib is loaded)
        # These will be initialized in reset

    def _setup_reward_buffers(self):
        """Setup buffers for reward computation."""
        # These buffers will store reference motion data
        pass

    def _parse_joint_err_weights(self):
        """Parse joint error weights for reward computation."""
        num_joints = self._kin_char_model.get_num_joints()
        
        if self.cfg.joint_err_w is None:
            self._joint_err_w = torch.ones(num_joints - 1, device=self.device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(
                self.cfg.joint_err_w, device=self.device, dtype=torch.float32
            )
        
        assert self._joint_err_w.shape[-1] == num_joints - 1
        
        # Convert to DoF error weights
        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self.device, dtype=torch.float32)
        
        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if dof_dim > 0:
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before physics simulation.
        
        Args:
            actions: Actions from the policy, shape (num_envs, action_dim)
        """
        # Update reference motion for all environments
        self._update_ref_motion_all()
        
        # Update reference character visualization if enabled
        if self._enable_ref_char():
            self._update_ref_char_all()
        
        # Store actions (will be applied in _apply_action)
        self.actions = actions.clone()
        
        # Scale actions if needed
        self.actions *= self.cfg.action_scale

    def _apply_action(self):
        """Apply actions to the character.
        
        This is called at each physics timestep (decimated).
        """
        if self.cfg.control_mode == "pd":
            # PD control: actions are target joint positions
            self.character.set_joint_position_target(self.actions)
        elif self.cfg.control_mode == "pos":
            # Position control
            self.character.set_joint_position_target(self.actions)
        elif self.cfg.control_mode == "vel":
            # Velocity control
            self.character.set_joint_velocity_target(self.actions)
        elif self.cfg.control_mode == "torque":
            # Torque control
            self.character.set_joint_effort_target(self.actions)
        else:
            raise ValueError(f"Unsupported control mode: {self.cfg.control_mode}")

    def _get_observations(self) -> dict:
        """Compute observations for the policy.
        
        Returns:
            Dictionary with "policy" key containing observations
        """
        # Get current state
        root_pos = self.character.data.root_pos_w
        root_rot = self.character.data.root_quat_w
        root_vel = self.character.data.root_lin_vel_w
        root_ang_vel = self.character.data.root_ang_vel_w
        dof_pos = self.character.data.joint_pos
        dof_vel = self.character.data.joint_vel
        
        # Convert DoF to joint rotations
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        
        # Get motion phase
        if self.cfg.enable_phase_obs:
            motion_times = self._get_motion_times()
            motion_phase = self._motion_lib.calc_motion_phase(self._motion_ids, motion_times)
        else:
            motion_phase = torch.zeros(self.num_envs, 0, device=self.device)
        
        # Get key body positions if needed
        if len(self._key_body_ids) > 0:
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[:, self._key_body_ids, :]
        else:
            key_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
        
        # Get target observations if needed
        if self.cfg.enable_tar_obs:
            tar_root_pos, tar_root_rot, tar_joint_rot = self._fetch_tar_obs_data(
                self._motion_ids, motion_times
            )
            
            # Compute target key positions
            if len(self._key_body_ids) > 0:
                tar_root_pos_flat = tar_root_pos.reshape(-1, 3)
                tar_root_rot_flat = tar_root_rot.reshape(-1, 4)
                tar_joint_rot_flat = tar_joint_rot.reshape(-1, tar_joint_rot.shape[-2], 4)
                
                tar_body_pos_flat, _ = self._kin_char_model.forward_kinematics(
                    tar_root_pos_flat, tar_root_rot_flat, tar_joint_rot_flat
                )
                tar_body_pos = tar_body_pos_flat.reshape(
                    self.num_envs, -1, tar_body_pos_flat.shape[-2], 3
                )
                tar_key_pos = tar_body_pos[:, :, self._key_body_ids, :]
            else:
                tar_key_pos = torch.zeros(self.num_envs, 0, 0, 3, device=self.device)
        else:
            tar_root_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
            tar_root_rot = torch.zeros(self.num_envs, 0, 4, device=self.device)
            tar_joint_rot = torch.zeros(self.num_envs, 0, joint_rot.shape[1], 4, device=self.device)
            tar_key_pos = torch.zeros(self.num_envs, 0, 0, 3, device=self.device)
        
        # Compute observations
        obs = compute_deepmimic_obs(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_obs=self.cfg.global_obs,
            root_height_obs=self.cfg.root_height_obs,
            phase=motion_phase,
            num_phase_encoding=self.cfg.num_phase_encoding,
            enable_phase_obs=self.cfg.enable_phase_obs,
            enable_tar_obs=self.cfg.enable_tar_obs,
            tar_root_pos=tar_root_pos,
            tar_root_rot=tar_root_rot,
            tar_joint_rot=tar_joint_rot,
            tar_key_pos=tar_key_pos,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for motion tracking.
        
        Returns:
            Reward tensor of shape (num_envs,)
        """
        # Get current state
        root_pos = self.character.data.root_pos_w
        root_rot = self.character.data.root_quat_w
        root_vel = self.character.data.root_lin_vel_w
        root_ang_vel = self.character.data.root_ang_vel_w
        dof_pos = self.character.data.joint_pos
        dof_vel = self.character.data.joint_vel
        body_pos = self.character.data.body_pos_w
        
        # Convert DoF to joint rotations
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        
        # Get key positions
        if len(self._key_body_ids) > 0:
            key_pos = body_pos[:, self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[:, self._key_body_ids, :]
        else:
            key_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
            ref_key_pos = key_pos
        
        track_root_h = self.cfg.root_height_obs
        track_root = self.cfg.enable_tar_obs and self.cfg.global_obs
        
        # Compute reward
        reward = compute_reward(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            tar_root_pos=self._ref_root_pos,
            tar_root_rot=self._ref_root_rot,
            tar_root_vel=self._ref_root_vel,
            tar_root_ang_vel=self._ref_root_ang_vel,
            tar_joint_rot=self._ref_joint_rot,
            tar_dof_vel=self._ref_dof_vel,
            tar_key_pos=ref_key_pos,
            joint_rot_err_w=self._joint_err_w,
            dof_err_w=self._dof_err_w,
            track_root_h=track_root_h,
            track_root=track_root,
            pose_w=self.cfg.reward_pose_w,
            vel_w=self.cfg.reward_vel_w,
            root_pose_w=self.cfg.reward_root_pose_w,
            root_vel_w=self.cfg.reward_root_vel_w,
            key_pos_w=self.cfg.reward_key_pos_w,
            pose_scale=self.cfg.reward_pose_scale,
            vel_scale=self.cfg.reward_vel_scale,
            root_pose_scale=self.cfg.reward_root_pose_scale,
            root_vel_scale=self.cfg.reward_root_vel_scale,
            key_pos_scale=self.cfg.reward_key_pos_scale,
        )
        
        return reward

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags for termination and timeout.
        
        Returns:
            Tuple of (terminated, time_out) tensors, each of shape (num_envs,)
        """
        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(self._motion_ids)
        
        # Check motion end
        motion_end = motion_times >= motion_len
        motion_len_term = motion_loop_mode != LoopMode.WRAP.value
        motion_end = torch.logical_and(motion_end, motion_len_term)
        
        # Check timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check early termination
        if self.cfg.enable_early_termination:
            root_rot = self.character.data.root_quat_w
            body_pos = self.character.data.body_pos_w
            # Get contact forces - use body_link_contact_forces or similar
            # In Isaac Lab, contact forces might be accessed differently
            contact_forces = torch.zeros_like(body_pos)  # Placeholder for now
            
            terminated = compute_termination(
                root_rot=root_rot,
                body_pos=body_pos,
                tar_root_rot=self._ref_root_rot,
                tar_body_pos=self._ref_body_pos,
                contact_force=contact_forces,
                contact_body_ids=self._contact_body_ids,
                termination_height=self.cfg.termination_height,
                pose_termination=self.cfg.pose_termination,
                pose_termination_dist=self.cfg.pose_termination_dist,
                global_obs=self.cfg.global_obs,
                track_root=self.cfg.enable_tar_obs and self.cfg.global_obs,
                episode_length=self.episode_length_buf,
            )
            
            # Mark successful completions
            terminated = torch.logical_or(terminated, motion_end)
        else:
            terminated = motion_end
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset specified environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        super()._reset_idx(env_ids)
        
        if len(env_ids) == 0:
            return
        
        # Sample new motions
        self._reset_ref_motion(env_ids)
        
        # Initialize character state from reference motion
        self._ref_state_init(env_ids)
        
        # Initialize reference character if visualizing
        if self._enable_ref_char():
            self._reset_ref_char(env_ids)

    def _reset_ref_motion(self, env_ids: torch.Tensor):
        """Sample and initialize reference motions for reset environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        # Get reference state from motion
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        # Initialize reference buffers if needed
        if not hasattr(self, '_ref_root_pos'):
            self._ref_root_pos = torch.zeros_like(self.character.data.root_pos_w)
            self._ref_root_rot = torch.zeros_like(self.character.data.root_quat_w)
            self._ref_root_vel = torch.zeros_like(self.character.data.root_lin_vel_w)
            self._ref_root_ang_vel = torch.zeros_like(self.character.data.root_ang_vel_w)
            self._ref_joint_rot = torch.zeros(
                self.num_envs, joint_rot.shape[1], 4, device=self.device
            )
            self._ref_dof_vel = torch.zeros_like(self.character.data.joint_vel)
            self._ref_body_pos = torch.zeros_like(self.character.data.body_pos_w)
            self._ref_dof_pos = torch.zeros_like(self.character.data.joint_pos)
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        # Compute body positions
        ref_body_pos, _ = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot
        )
        self._ref_body_pos[:] = ref_body_pos
        
        # Convert joint rotations to DoF positions
        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos

    def _ref_state_init(self, env_ids: torch.Tensor):
        """Initialize character state from reference motion.
        
        Args:
            env_ids: Environment indices to reset
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
        
        # Set joint state (convert from XML DOF order to USD DOF order)
        joint_pos_xml = self._ref_dof_pos[env_ids]
        joint_pos_usd = self._map_dof_xml_to_usd(joint_pos_xml)
        joint_vel_xml = self._ref_dof_vel[env_ids]
        joint_vel_usd = self._map_dof_xml_to_usd(joint_vel_xml)
        
        # Write to simulation
        self.character.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.character.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.character.write_joint_state_to_sim(joint_pos_usd, joint_vel_usd, None, env_ids)

    def _reset_ref_char(self, env_ids: torch.Tensor):
        """Reset reference character visualization.
        
        Args:
            env_ids: Environment indices to reset
        """
        # Set root state with offset
        root_state = self.ref_character.data.default_root_state[env_ids].clone()
        root_pos = self._ref_root_pos[env_ids] + self._ref_char_offset
        root_rot = self._ref_root_rot[env_ids]
        
        # Convert quaternion from MimicKit format [x,y,z,w] to Isaac Lab format [w,x,y,z]
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        
        root_state[:, :3] = root_pos
        # Add environment origins to maintain env_spacing during reset
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_rot_wxyz
        root_state[:, 7:] = 0.0  # Zero velocity for visualization
        
        # Set joint state (convert from XML DOF order to USD DOF order)
        joint_pos_xml = self._ref_dof_pos[env_ids]
        joint_pos_usd = self._map_dof_xml_to_usd(joint_pos_xml)
        joint_vel = torch.zeros_like(joint_pos_usd)
        
        # Write to simulation
        self.ref_character.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.ref_character.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.ref_character.write_joint_state_to_sim(joint_pos_usd, joint_vel, None, env_ids)
    
    def _update_ref_motion_all(self):
        """Update reference motion for all environments (called every frame).
        
        This computes the reference motion state at the current time for tracking.
        """
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        
        # Get motion frame at current time
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(
            motion_ids, motion_times
        )
        
        # Update reference state buffers
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel
        
        # Compute body positions via forward kinematics
        ref_body_pos, _ = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot
        )
        self._ref_body_pos[:] = ref_body_pos
        
        # Convert joint rotations to DoF positions for visualization
        if self._enable_ref_char():
            dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
            self._ref_dof_pos[:] = dof_pos
    
    def _update_ref_char_all(self):
        """Update reference character visualization for all environments.
        
        This sets the reference character's pose to match the current reference motion.
        """
        # Set root state with offset
        root_pos = self._ref_root_pos + self._ref_char_offset
        root_rot = self._ref_root_rot
        
        # Convert quaternion from MimicKit format [x,y,z,w] to Isaac Lab format [w,x,y,z]
        root_rot_wxyz = torch.cat([root_rot[:, 3:4], root_rot[:, :3]], dim=-1)
        
        # Convert joint positions from XML to USD order
        joint_pos_xml = self._ref_dof_pos
        joint_pos_usd = self._map_dof_xml_to_usd(joint_pos_xml)
        
        # Write to simulation (use None for env_ids to update all)
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

    # Helper methods
    
    def _build_dof_mapping(self):
        """Build mapping from XML DOF indices to USD DOF indices.
        
        The PKL motion files store DOF values in XML order (as defined in smpl.xml).
        However, Isaac Lab's USD Articulation may have joints in a different order.
        This function creates a mapping to convert between the two orderings.
        """
        xml_body_names = self._kin_char_model.get_body_names()
        usd_joint_names = list(self.character.data.joint_names)
        
        xml_dof_size = self._kin_char_model.get_dof_size()
        usd_dof_size = self.character.num_joints
        
        if xml_dof_size != usd_dof_size:
            logger.warning(f"DOF size mismatch: XML={xml_dof_size}, USD={usd_dof_size}")
        
        # Build DOF mapping: XML DOF index → USD DOF index
        # Strategy: For each XML body, find the corresponding USD joints by name
        xml_to_usd_dof_map = []
        
        for xml_body_idx in range(1, len(xml_body_names)):  # Skip root (index 0)
            xml_body_name = xml_body_names[xml_body_idx]
            
            # Find the 3 USD joints for this body: {body_name}_x, {body_name}_y, {body_name}_z
            for axis_suffix in ['_x', '_y', '_z']:
                usd_joint_name = xml_body_name + axis_suffix
                
                if usd_joint_name in usd_joint_names:
                    usd_dof_idx = usd_joint_names.index(usd_joint_name)
                    xml_to_usd_dof_map.append(usd_dof_idx)
                else:
                    logger.warning(f"USD joint not found: {usd_joint_name}")
                    xml_to_usd_dof_map.append(0)  # Placeholder
        
        self._xml_to_usd_dof_indices = torch.tensor(
            xml_to_usd_dof_map, 
            dtype=torch.long, 
            device=self.device
        )
    
    def _map_dof_xml_to_usd(self, dof_xml: torch.Tensor) -> torch.Tensor:
        """Convert DOF values from XML order to USD order.
        
        Args:
            dof_xml: DOF values in XML order, shape (..., xml_dof_size)
            
        Returns:
            DOF values in USD order, shape (..., usd_dof_size)
        """
        # Create output tensor in USD order
        dof_usd = torch.zeros_like(dof_xml)
        
        # Map each XML DOF to its corresponding USD position
        # self._xml_to_usd_dof_indices[i] tells us: XML DOF i should go to USD position indices[i]
        for xml_idx in range(len(self._xml_to_usd_dof_indices)):
            usd_idx = self._xml_to_usd_dof_indices[xml_idx]
            dof_usd[..., usd_idx] = dof_xml[..., xml_idx]
        
        return dof_usd
    
    def _get_motion_times(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current motion times for environments."""
        if env_ids is None:
            motion_times = self.episode_length_buf * self.step_dt + self._motion_time_offsets
        else:
            motion_times = self.episode_length_buf[env_ids] * self.step_dt + self._motion_time_offsets[env_ids]
        return motion_times

    def _sample_motion_times(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample motion IDs and times for initialization."""
        motion_ids = self._motion_lib.sample_motions(n)
        
        if self.cfg.rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float32, device=self.device)
        
        return motion_ids, motion_times

    def _fetch_tar_obs_data(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch target observation data for future timesteps."""
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        
        motion_times = motion_times.unsqueeze(-1)
        time_steps = self.step_dt * self._tar_obs_steps
        motion_times = motion_times + time_steps
        motion_ids_tiled = motion_ids.unsqueeze(-1).expand(-1, num_steps)
        
        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        
        root_pos, root_rot, _, _, joint_rot, _ = self._motion_lib.calc_motion_frame(
            motion_ids_tiled, motion_times
        )
        
        root_pos = root_pos.reshape(n, num_steps, 3)
        root_rot = root_rot.reshape(n, num_steps, 4)
        joint_rot = joint_rot.reshape(n, num_steps, joint_rot.shape[-2], 4)
        
        return root_pos, root_rot, joint_rot

    def _get_body_ids(self, body_names: list[str]) -> torch.Tensor:
        """Get body IDs from body names."""
        body_ids = []
        for name in body_names:
            body_id = self._kin_char_model.get_body_id(name)
            body_ids.append(body_id)
        return torch.tensor(body_ids, device=self.device, dtype=torch.long)

    def _enable_ref_char(self) -> bool:
        """Check if reference character should be visualized."""
        return self.cfg.visualize_ref_char and self.sim.has_gui()
    
    def _has_key_bodies(self) -> bool:
        """Check if key bodies are defined for tracking."""
        return hasattr(self, '_key_body_ids') and len(self._key_body_ids) > 0

    def _convert_xml_to_usd(self, xml_file: str) -> str:
        """Convert XML character file to USD.
        
        Note: This is a placeholder. In production, you would need a proper
        converter from MJCF/XML to USD format.
        """
        # For now, assume USD file exists or return path to be converted
        import os
        base_name = os.path.splitext(xml_file)[0]
        usd_file = base_name + ".usd"
        
        # In production, check if file exists and convert if needed
        return usd_file

    def _get_joint_stiffness(self) -> dict:
        """Get joint stiffness values for actuators."""
        # Return default stiffness or load from character config
        return {".*": 100.0}

    def _get_joint_damping(self) -> dict:
        """Get joint damping values for actuators."""
        # Return default damping or load from character config
        return {".*": 10.0}


# JIT compiled reward and observation functions

@torch.jit.script
def compute_char_obs(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_rot: torch.Tensor,
    dof_vel: torch.Tensor,
    key_pos: torch.Tensor,
    global_obs: bool,
    root_height_obs: bool,
) -> torch.Tensor:
    """Compute character observations."""
    if global_obs:
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        heading_rot = torch_util.calc_heading_quat_inv(root_rot)
        local_root_rot = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)
    
    # Joint rotations
    joint_rot_flat = joint_rot.reshape(-1, 4)
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = joint_rot_obs_flat.reshape(joint_rot.shape[0], -1)
    
    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]
    
    # Key body positions
    if key_pos.shape[1] > 0:
        key_pos = key_pos - root_pos.unsqueeze(1)
        if not global_obs:
            heading_rot = torch_util.calc_heading_quat_inv(root_rot)
            heading_rot_exp = heading_rot.unsqueeze(1).expand(-1, key_pos.shape[1], -1)
            key_pos_flat = key_pos.reshape(-1, 3)
            heading_rot_flat = heading_rot_exp.reshape(-1, 4)
            key_pos_flat = torch_util.quat_rotate(heading_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)
        key_pos_flat = key_pos.reshape(key_pos.shape[0], -1)
        obs.append(key_pos_flat)
    
    if root_height_obs:
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs
    
    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_phase_obs(phase: torch.Tensor, num_phase_encoding: int) -> torch.Tensor:
    """Compute phase observations with positional encoding."""
    phase_obs = phase.unsqueeze(-1)
    
    if num_phase_encoding > 0:
        pe_exp = torch.arange(num_phase_encoding, device=phase.device, dtype=phase.dtype)
        pe_scale = 2.0 * np.pi * torch.pow(2.0, pe_exp)
        pe_val = phase.unsqueeze(-1) * pe_scale.unsqueeze(0)
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)
        phase_obs = torch.cat([phase_obs, pe_sin, pe_cos], dim=-1)
    
    return phase_obs


@torch.jit.script
def compute_deepmimic_obs(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_rot: torch.Tensor,
    dof_vel: torch.Tensor,
    key_pos: torch.Tensor,
    global_obs: bool,
    root_height_obs: bool,
    phase: torch.Tensor,
    num_phase_encoding: int,
    enable_phase_obs: bool,
    enable_tar_obs: bool,
    tar_root_pos: torch.Tensor,
    tar_root_rot: torch.Tensor,
    tar_joint_rot: torch.Tensor,
    tar_key_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute full DeepMimic observations."""
    char_obs = compute_char_obs(
        root_pos, root_rot, root_vel, root_ang_vel,
        joint_rot, dof_vel, key_pos,
        global_obs, root_height_obs
    )
    obs = [char_obs]
    
    if enable_phase_obs:
        phase_obs = compute_phase_obs(phase, num_phase_encoding)
        obs.append(phase_obs)
    
    # Target observations would be added here if enable_tar_obs is True
    # This is simplified for brevity
    
    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_reward(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_rot: torch.Tensor,
    dof_vel: torch.Tensor,
    key_pos: torch.Tensor,
    tar_root_pos: torch.Tensor,
    tar_root_rot: torch.Tensor,
    tar_root_vel: torch.Tensor,
    tar_root_ang_vel: torch.Tensor,
    tar_joint_rot: torch.Tensor,
    tar_dof_vel: torch.Tensor,
    tar_key_pos: torch.Tensor,
    joint_rot_err_w: torch.Tensor,
    dof_err_w: torch.Tensor,
    track_root_h: bool,
    track_root: bool,
    pose_w: float,
    vel_w: float,
    root_pose_w: float,
    root_vel_w: float,
    key_pos_w: float,
    pose_scale: float,
    vel_scale: float,
    root_pose_scale: float,
    root_vel_scale: float,
    key_pos_scale: float,
) -> torch.Tensor:
    """Compute motion tracking reward."""
    # Pose reward
    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_err = torch.sum(joint_rot_err_w * pose_diff * pose_diff, dim=-1)
    
    # Velocity reward
    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)
    
    # Root pose reward
    root_pos_diff = tar_root_pos - root_pos
    if not track_root:
        root_pos_diff[:, 0:2] = 0
    if not track_root_h:
        root_pos_diff[:, 2] = 0
    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = root_rot_err * root_rot_err
    
    # Root velocity reward
    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
    
    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
    
    # Key position reward
    if key_pos.shape[1] > 0:
        key_pos = key_pos - root_pos.unsqueeze(1)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(1)
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros_like(pose_err)
    
    # Compute reward components
    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)
    
    # Total reward
    reward = (
        pose_w * pose_r +
        vel_w * vel_r +
        root_pose_w * root_pose_r +
        root_vel_w * root_vel_r +
        key_pos_w * key_pos_r
    )
    
    return reward


@torch.jit.script
def compute_termination(
    root_rot: torch.Tensor,
    body_pos: torch.Tensor,
    tar_root_rot: torch.Tensor,
    tar_body_pos: torch.Tensor,
    contact_force: torch.Tensor,
    contact_body_ids: torch.Tensor,
    termination_height: float,
    pose_termination: bool,
    pose_termination_dist: float,
    global_obs: bool,
    track_root: bool,
    episode_length: torch.Tensor,
) -> torch.Tensor:
    """Compute early termination flags."""
    terminated = torch.zeros(root_rot.shape[0], dtype=torch.bool, device=root_rot.device)
    
    # Check for falls (non-contact bodies touching ground)
    if contact_body_ids.shape[0] > 0:
        masked_contact = contact_force.clone()
        masked_contact[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact) > 0.1, dim=-1)
        
        body_height = body_pos[:, :, 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        
        fall_contact = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.any(fall_contact, dim=-1)
        terminated = torch.logical_or(terminated, has_fallen)
    
    # Check pose termination
    if pose_termination:
        root_pos = body_pos[:, 0:1, :]
        tar_root_pos = tar_body_pos[:, 0:1, :]
        
        if not global_obs:
            body_pos = body_pos[:, 1:, :] - root_pos
            tar_body_pos = tar_body_pos[:, 1:, :] - tar_root_pos
            # Apply rotation to local frame
        elif not track_root:
            body_pos = body_pos[:, 1:, :] - root_pos
            tar_body_pos = tar_body_pos[:, 1:, :] - tar_root_pos
        
        body_pos_diff = tar_body_pos - body_pos
        body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
        body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
        pose_fail = body_pos_dist > pose_termination_dist * pose_termination_dist
        
        if track_root:
            root_pos_diff = tar_root_pos - root_pos
            root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
            root_pos_fail = root_pos_dist > pose_termination_dist * pose_termination_dist
            root_pos_fail = root_pos_fail.squeeze(-1)
            pose_fail = torch.logical_or(pose_fail, root_pos_fail)
        
        terminated = torch.logical_or(terminated, pose_fail)
    
    # Only fail after first timestep
    not_first_step = episode_length > 0
    terminated = torch.logical_and(terminated, not_first_step)
    
    return terminated

