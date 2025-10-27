# MimicKit for Isaac Lab

Physics-based character animation using Isaac Lab, featuring motion imitation with DeepMimic, AMP, and ASE algorithms.

This is an Isaac Lab implementation of [MimicKit](https://github.com/xbpeng/MimicKit), adapted for NVIDIA Isaac Sim and Isaac Lab. The G1 robot configuration and USD assets are adapted from [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab).

> **Note**: This repository was developed with significant AI assistance. If you find areas for improvement or optimization, please open an issue or pull request. Contributions are welcome!

## Features

- **DeepMimic**: Learn locomotion skills from reference motion data
- **AMP**: Style-aware motion learning with discriminator
- **ASE**: Diverse skill discovery through exploration
- **SMPL & G1 Support**: Humanoid (SMPL) and robot (Unitree G1) character models
- **GPU-Accelerated**: Parallel simulation with NVIDIA Isaac Sim

### IsaacLab Demos (Simulation)

<table>
  <tr>
    <td align="center">
      <img src="docs/1.gif" alt="Policy Playback" width="400"/>
    </td>
    <td align="center">
      <img src="docs/2.gif" alt="Motion Replay" width="400"/>
    </td>
  </tr>
</table>
**I reconstruct the scene using 3D Gaussian Splatting combined with 3DGRUT**

## Installation

### Prerequisites
- Ubuntu 20.04/22.04/24.04
- NVIDIA GPU with CUDA support 
- Isaac Lab 2.1+
- Python 3.10+
- Isaac Sim 4.5+ (tested with 5.0)

### Setup

1. **Install Isaac Lab** following [official instructions](https://isaac-sim.github.io/IsaacLab/)

2. **Clone and install MimicKit for Isaac Lab**:
```bash
git clone https://github.com/nathanwu7/MimicKit_IsaacLab.git
cd MimicKit_IsaacLab
pip install -e source/mimickit_isaaclab
```

3. **Download motion data**:
```bash
# Motion data should be placed in:
# - data/motions/smpl/*.pkl  (SMPL motions)
# - data/motions/g1/*.pkl    (G1 motions)
# Character models in:
# - data/assets/smpl/        (SMPL models)
# - data/assets/g1/          (G1 models)
```

## Quick Start

### Training

#### DeepMimic
```bash
# SMPL humanoid
python scripts/train.py --task MimicKit-DeepMimic-SMPL-Direct-v0

# G1 robot
python scripts/train.py --task MimicKit-DeepMimic-G1-Direct-v0
```

#### AMP (Adversarial Motion Priors)
```bash
# SMPL humanoid
python scripts/train.py --task MimicKit-AMP-SMPL-Direct-v0

# G1 robot
python scripts/train.py --task MimicKit-AMP-G1-Direct-v0
```

#### ASE (Adversarial Skill Embeddings)
```bash
# SMPL humanoid
python scripts/train.py --task MimicKit-ASE-SMPL-Direct-v0

# G1 robot
python scripts/train.py --task MimicKit-ASE-G1-Direct-v0
```

#### Training Options
```bash
python scripts/train.py --task MimicKit-DeepMimic-G1-Direct-v0 \
    --num_envs 4096 \          # Number of parallel environments
    --max_iterations 10000 \   # Training iterations
    --headless                 # Run without GUI (faster)
```

### Inference & Visualization

#### 1. Play Trained Policy (Strategy Only)
```bash
python scripts/play.py --task MimicKit-DeepMimic-G1-Direct-v0 \
    --num_envs 1 \
    --checkpoint path/to/model.pt
```

#### 2. Replay Reference Motion (Data Only)
```bash
python scripts/replay.py --task MimicKit-DeepMimic-G1-Direct-v0 \
    --num_envs 1 \
    --motion_file data/motions/g1/g1_walk.pkl
```

#### 3. Compare Policy with Reference (Side-by-Side)
```bash
python scripts/compare.py --task MimicKit-DeepMimic-G1-Direct-v0 \
    --num_envs 1 \
    --checkpoint path/to/model.pt
```

## Available Environments

### SMPL Humanoid (69 DOF)

| Environment ID | Description |
|----------------|-------------|
| `MimicKit-DeepMimic-SMPL-Direct-v0` | DeepMimic with SMPL humanoid |
| `MimicKit-AMP-SMPL-Direct-v0` | AMP with SMPL humanoid |
| `MimicKit-ASE-SMPL-Direct-v0` | ASE with SMPL humanoid |

### Unitree G1 Robot (29 DOF)

| Environment ID | Description |
|----------------|-------------|
| `MimicKit-DeepMimic-G1-Direct-v0` | DeepMimic with G1 robot |
| `MimicKit-AMP-G1-Direct-v0` | AMP with G1 robot |
| `MimicKit-ASE-G1-Direct-v0` | ASE with G1 robot |

## Configuration

### Environment Configuration
Located in `source/mimickit_isaaclab/tasks/direct/{algorithm}/{algorithm}_{robot}_env_cfg.py`

**Key Parameters**:
```python
# Motion file
motion_file = "data/motions/g1/g1_walk.pkl"

# Termination conditions
enable_early_termination = True
termination_height = 0.1         # Min height before termination
pose_termination = True
pose_termination_dist = 5.0      # Max deviation from reference

# Visualization
visualize_ref_char = False       # Show reference character (training)
ref_char_offset = [2.0, 0.0, 0.0]  # Reference character offset
```

### Agent Configuration
Located in `source/mimickit_isaaclab/tasks/direct/{algorithm}/agents/`

## Motion Data Format

Motion files (`.pkl`) should contain a dictionary with:

```python
{
    "fps": 30,                    # Frames per second
    "loop_mode": "wrap",          # "wrap" or "clamp"
    "frames": {
        "root_pos": np.array,     # Shape: (num_frames, 3)
        "root_rot_exp_map": np.array,  # Shape: (num_frames, 3)
        "joint_dof": np.array,    # Shape: (num_frames, num_dofs)
    }
}
```

### DOF Configuration

- **SMPL**: 69 DOF (3 DOF per joint, spherical joints)
- **G1**: 29 DOF (1 DOF per joint, revolute joints)



## Reference

```bibtex
@misc{mimickit_peng2025,
  title={MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control}, 
  author={Xue Bin Peng},
  year={2025},
  eprint={2510.13794},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2510.13794}, 
}

@article{2018-TOG-deepMimic,
  author = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
  title = {DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills},
  journal = {ACM Trans. Graph.},
  volume = {37},
  number = {4},
  year = {2018},
  doi = {10.1145/3197517.3201311},
}

@article{2021-TOG-AMP,
  author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  journal = {ACM Trans. Graph.},
  volume = {40},
  number = {4},
  year = {2021},
  doi = {10.1145/3450626.3459670},
}

@article{2022-TOG-ASE,
  author = {Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
  title = {ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
  journal = {ACM Trans. Graph.},
  volume = {41},
  number = {4},
  year = {2022},
}
```

## License

This project is licensed under the BSD-3-Clause License. See LICENSE file for details.

## Acknowledgments

- Based on [MimicKit](https://github.com/xbpeng/MimicKit) by Xue Bin Peng
- Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) by NVIDIA
- G1 robot configuration and USD assets adapted from [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab) by Ziang Zheng
- Inspired by:
  - [DeepMimic](https://arxiv.org/abs/1804.02717) (Peng et al., 2018)
  - [AMP](https://arxiv.org/abs/2104.02180) (Peng et al., 2021)
  - [ASE](https://arxiv.org/abs/2205.01906) (Peng et al., 2022)

## Contact & Support

For issues, questions, or contributions:
- Open an issue on [GitHub](https://github.com/nathanwu7/MimicKit_IsaacLab)
- Refer to [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for simulation-specific questions
- See original [MimicKit](https://github.com/xbpeng/MimicKit) for Isaac Gym implementation
