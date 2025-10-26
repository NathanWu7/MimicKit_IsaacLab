#!/bin/bash
# Quick setup script for MimicKit IsaacLab

set -e

echo "========================================="
echo "MimicKit IsaacLab Quick Setup"
echo "========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Isaac Lab is installed
echo -e "${YELLOW}Checking Isaac Lab installation...${NC}"
if ! python -c "import isaaclab" &> /dev/null; then
    echo -e "${RED}Error: Isaac Lab not found!${NC}"
    echo "Please install Isaac Lab first:"
    echo "  cd /home/wqw/git_pkgs/IsaacLab"
    echo "  ./isaaclab.sh --install"
    exit 1
fi
echo -e "${GREEN}✓ Isaac Lab found${NC}"

# Check if Isaac Sim is installed
echo -e "${YELLOW}Checking Isaac Sim installation...${NC}"
if ! python -c "import isaacsim" &> /dev/null; then
    echo -e "${RED}Error: Isaac Sim not found!${NC}"
    echo "Please install Isaac Sim first:"
    echo "  pip install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com"
    exit 1
fi
echo -e "${GREEN}✓ Isaac Sim found${NC}"

# Install MimicKit IsaacLab
echo -e "${YELLOW}Installing MimicKit IsaacLab...${NC}"
cd source/mimickit_isaaclab
pip install -e .
cd ../..
echo -e "${GREEN}✓ MimicKit IsaacLab installed${NC}"

# Check data directory
echo -e "${YELLOW}Checking data directory...${NC}"
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Creating data symlink...${NC}"
    ln -s ../data ./data
fi
echo -e "${GREEN}✓ Data directory ready${NC}"

# Make scripts executable
echo -e "${YELLOW}Setting up scripts...${NC}"
chmod +x scripts/*.py
echo -e "${GREEN}✓ Scripts ready${NC}"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "You can now:"
echo "  1. Train a model:"
echo "     python scripts/train.py --task MimicKit-DeepMimic-Humanoid-Direct-v0 --num_envs 1024 --headless"
echo ""
echo "  2. Test a trained model:"
echo "     python scripts/play.py --task MimicKit-DeepMimic-Humanoid-Direct-v0 --use_last_checkpoint"
echo ""
echo "  3. See available environments:"
echo "     python -c \"import gymnasium as gym; import mimickit_isaaclab; print([e for e in gym.envs.registry if 'MimicKit' in e])\""
echo ""

