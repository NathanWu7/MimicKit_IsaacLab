# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# G1-specific kinematic character model for G1 motion data.

import torch
from typing import List

from .kin_char_model import KinCharModel, Joint, JointType


class KinCharModelG1(KinCharModel):
    """G1 kinematic character model for G1 motion data (29 DOF).
    
    This class is used for G1 format motion data (such as g1_walk.pkl).
    It provides the kinematic model that matches G1's joint structure.
    
    G1 has 29 DOF in the following order:
    [0-5]   Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    [6-11]  Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll  
    [12-14] Waist: yaw, roll, pitch
    [15-21] Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, 
                      wrist_roll, wrist_pitch, wrist_yaw
    [22-28] Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                       wrist_roll, wrist_pitch, wrist_yaw
    """
    
    # G1 29DOF joint names in order
    G1_29DOF_JOINT_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    
    def __init__(self, device):
        """Initialize G1 kinematic model.
        
        Args:
            device: Device to store tensors on
        """
        super().__init__(device)
        self._device = device
        
        # Initialize G1 structure
        self._init_g1_structure()
        
        # IMPORTANT: Override _dof_size AFTER calling init() because 
        # the base class init() calculates dof_size from joints
        # But for G1, we want to use the fixed 29 DOF
        self._dof_size = 29
        
    def _init_g1_structure(self):
        """Initialize G1 robot kinematic structure.
        
        For G1, we create a structure with semantic body names that match
        the actual G1 robot structure for proper tracking.
        """
        # Create bodies with semantic names matching G1 structure
        body_names = [
            "pelvis",  # 0 - Root
            # Left leg (6 joints)
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", 
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            # Right leg (6 joints)
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
            # Waist (3 joints)
            "waist_yaw", "waist_roll", "waist_pitch",
            # Left arm (7 joints)
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
            "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
            # Right arm (7 joints)
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
            "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        ]
        
        # Parent structure: all joints are children of root (simplified)
        parent_indices = [-1]  # pelvis (root)
        parent_indices.extend([0] * 29)  # All joints parent to root for simplicity
        
        # Local translations (all zero for simplicity)
        local_translation = [[0.0, 0.0, 0.0] for _ in body_names]
        
        # Identity rotations
        local_rotation = [[0.0, 0.0, 0.0, 1.0] for _ in body_names]
        
        # Define rotation axes for each joint based on pitch/roll/yaw naming
        # pitch: Y-axis, roll: X-axis, yaw: Z-axis
        joint_axes = self._get_g1_joint_axes()
        
        # Create 29 hinge joints (one per DOF) with correct rotation axes
        joints = [self._build_root_joint()]
        for i in range(29):
            joint = Joint(
                name=self.G1_29DOF_JOINT_NAMES[i],
                joint_type=JointType.HINGE,
                axis=joint_axes[i]
            )
            joints.append(joint)
        
        # Initialize parent class
        self.init(
            body_names=body_names,
            parent_indices=parent_indices,
            local_translation=local_translation,
            local_rotation=local_rotation,
            joints=joints
        )
    
    def _get_g1_joint_axes(self) -> List[torch.Tensor]:
        """Get rotation axes for each G1 joint based on naming convention.
        
        pitch: Y-axis [0, 1, 0]
        roll:  X-axis [1, 0, 0]
        yaw:   Z-axis [0, 0, 1]
        
        Returns:
            List of 29 axis tensors
        """
        axes = []
        for joint_name in self.G1_29DOF_JOINT_NAMES:
            if "pitch" in joint_name or "elbow" in joint_name or "knee" in joint_name:
                # Pitch and elbow/knee joints rotate around Y-axis
                axis = torch.tensor([0.0, 1.0, 0.0], device=self._device)
            elif "roll" in joint_name:
                # Roll joints rotate around X-axis
                axis = torch.tensor([1.0, 0.0, 0.0], device=self._device)
            elif "yaw" in joint_name:
                # Yaw joints rotate around Z-axis
                axis = torch.tensor([0.0, 0.0, 1.0], device=self._device)
            else:
                # Default to Y-axis for unknown joints
                axis = torch.tensor([0.0, 1.0, 0.0], device=self._device)
            axes.append(axis)
        return axes
        
    def get_dof_size(self) -> int:
        """Get G1 DOF size (29)."""
        return self._dof_size
    
    def get_g1_joint_names(self) -> List[str]:
        """Get G1 joint names in order."""
        return self.G1_29DOF_JOINT_NAMES.copy()
    
    def dof_to_rot(self, dof: torch.Tensor) -> torch.Tensor:
        """Convert G1 DOF (29) to joint rotations.
        
        Uses parent class implementation which properly handles each joint's
        rotation axis (pitch/roll/yaw) via axis-angle to quaternion conversion.
        
        Args:
            dof: G1 DOF values, shape (..., 29)
            
        Returns:
            Joint rotations, shape (..., 29, 4) as quaternions
        """
        # Use parent class implementation which properly uses each joint's axis
        return super().dof_to_rot(dof)
    
    def rot_to_dof(self, rot: torch.Tensor) -> torch.Tensor:
        """Convert joint rotations back to G1 DOF.
        
        Uses parent class implementation which properly extracts the rotation
        angle around each joint's axis (pitch/roll/yaw) using quat_twist_angle.
        
        Args:
            rot: Joint rotations, shape (..., 29, 4) as quaternions
            
        Returns:
            G1 DOF values, shape (..., 29)
        """
        # Use parent class implementation which properly uses each joint's axis
        return super().rot_to_dof(rot)


