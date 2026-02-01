# Environment Setup Guide - RL Project

This guide will help you recreate the environment on any Linux system, regardless of hardware specs.

## System Requirements

### Minimum Requirements:
- **OS:** Ubuntu 20.04+ or similar Linux distribution
- **Python:** 3.8 - 3.10
- **RAM:** 8 GB minimum (16 GB recommended)
- **CPU:** 4 cores minimum (more cores = faster training)
- **GPU:** Optional but recommended (NVIDIA with CUDA support)
- **Disk Space:** 10 GB free

### Recommended for Faster Training:
- **CPU:** 32+ cores (for parallel environment training)
- **GPU:** NVIDIA RTX 2060 or better with 6+ GB VRAM
- **RAM:** 24 GB+

## Installation Steps

### 1. System Packages

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-venv git build-essential

# Install MuJoCo dependencies
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

### 2. MuJoCo Physics Engine

```bash
# Download MuJoCo 2.1.0
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz

# Set environment variables (add to ~/.bashrc)
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

### 3. CUDA Setup (Optional - For GPU)

**If you have NVIDIA GPU:**

```bash
# Check your NVIDIA driver
nvidia-smi

# If not installed, install NVIDIA drivers
sudo apt install nvidia-driver-535  # or latest version

# Install CUDA 11.8 (compatible with PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**If NO GPU available:** The code will automatically use CPU.

### 4. Project Setup

```bash
# Extract the project zip
unzip rl_mldl_25_backup.zip
cd rl_mldl_25

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Test MuJoCo
python -c "import mujoco_py; print('MuJoCo OK')"

# Test environment
python -c "import gym; from env.custom_hopper import *; env = gym.make('CustomHopper-target-v0'); print('Environment OK')"

# Check GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Key Dependencies

```
Python: 3.10
stable-baselines3 = 1.8.0
gym = 0.21.0
mujoco-py = 2.1.2.14
torch = 2.0.1+cu118
optuna = 3.1.1
numpy < 1.24
```

## Adapting for Different Hardware

### CPU-Only Systems:
- Reduce `--n_envs` parameter (use 4-8 instead of 16)
- Training will be slower but works fine
- Expect 3-4× longer training times

### Low-Memory Systems (< 16GB):
- Reduce `--n_envs` to 4-8
- Close other applications during training
- Use swap space if needed

### High-Core Systems (64+ cores):
Maximize parallel environments:
```bash
# For 64 cores
python train_simopt.py --n_envs 48 ...

# For 96 cores  
python train_simopt.py --n_envs 80 ...
```

### GPU Systems:
Training automatically uses GPU if available. Check with:
```bash
nvidia-smi  # Monitor GPU usage
```

## Running the Project

### Test Trained Models:
```bash
source venv/bin/activate

# Test on target environment
python test_model.py --model best_model/source_model.zip --episodes 50 --env CustomHopper-target-v0
```

### Train New Models:

**SAC with default params:**
```bash
python train_sb3.py --episodes 2000 --n_envs 16
```

**UDR Training:**
```bash
python train_sb3.py --episodes 2000 --n_envs 16 --use_udr --mass_variation 0.3
```

**SimOpt Training:**
```bash
python train_simopt.py --load_best_params best_sac_params.json \
    --n-initial-points 5 --n-iterations 20 \
    --episodes 2000 --n_envs 16 \
    --mass_variation 0.3 --eval-episodes 50
```

## Troubleshooting

### MuJoCo Issues:
```bash
# If you get "GLEW initialization error"
export MUJOCO_GL=egl

# If rendering issues
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### Out of Memory:
- Reduce `--n_envs` parameter
- Reduce `--eval-episodes` from 50 to 10
- Close other applications

### Slow Training:
- Increase `--n_envs` if you have more CPU cores
- Ensure GPU is being used (check with `nvidia-smi`)
- Training time scales with: `n_envs` (cores used)

### Import Errors:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or install individually
pip install stable-baselines3==1.8.0
pip install gym==0.21.0
pip install mujoco-py==2.1.2.14
```

## Performance Expectations

### Training Times (per trial, 1M timesteps):

| Hardware | Parallel Envs | Time/Trial |
|----------|---------------|------------|
| 4 cores, no GPU | 4 | ~180 min |
| 16 cores, no GPU | 16 | ~60 min |
| 16 cores + GPU | 16 | ~48 min |
| 64 cores + GPU | 48 | ~12 min |
| 96 cores + GPU | 80 | ~6 min |

### Adjust Training Parameters:

For **faster testing** (lower quality):
```bash
--episodes 1000  # Instead of 2000
--n-iterations 10  # Instead of 20
```

For **better results** (slower):
```bash
--episodes 3000  # More training
--eval-episodes 100  # Better evaluation
```

## File Structure

```
rl_mldl_25/
├── env/                    # Custom Hopper environment
├── best_model*/            # Trained models
├── train_simopt.py        # SimOpt training
├── train_sb3.py           # SAC/UDR training
├── test_model.py          # Model evaluation
├── requirements.txt       # Python dependencies
└── venv/                  # Virtual environment (create this)
```

## Notes

- **CPU cores:** More cores → faster training (near-linear scaling up to ~64-96 cores)
- **GPU:** Speeds up neural network updates (~20-30% faster overall)
- **RAM:** 16GB+ recommended for 16 parallel environments
- **Disk:** Models are ~3MB each, logs can grow large

## Support

For issues:
1. Check `train_*.log` files for errors
2. Verify all dependencies installed correctly
3. Ensure MuJoCo environment variables are set
4. Test with fewer parallel environments first (`--n_envs 4`)
