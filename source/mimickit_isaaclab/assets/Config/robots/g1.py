# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Configuration for Unitree G1 robot.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# Get asset directory
ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

##
# Configuration
##

UNITREE_G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/USD/Robots/g1_29dof/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness=40.18,  # Match MimicKit XML
            damping=2.56,     # Match MimicKit XML
            armature=0.0102,  # Match MimicKit XML
        ),
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness=99.10,  # Match MimicKit XML
            damping=6.31,     # Match MimicKit XML
            armature=0.0251,  # Match MimicKit XML
        ),
        "ankle_waist_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_shoulder_.*",
                ".*_elbow_.*",
            ],
            effort_limit_sim=50,  # Match ankle limit
            velocity_limit_sim=37,
            stiffness=28.50,  # Match MimicKit XML ankle
            damping=1.81,     # Match MimicKit XML ankle
            armature=0.0072,  # Match MimicKit XML
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=8.55,   # Match MimicKit XML wrist
            damping=0.54,     # Match MimicKit XML wrist
            armature=0.0022,  # Match MimicKit XML
        ),
    },
)

# Alias for compatibility
G1_CFG = UNITREE_G1_29DOF_CFG

# G1 29DOF joint names in SDK order (for reference)
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

