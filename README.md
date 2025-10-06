# Modular_diffusion Installation Guide

## Project Overview
This project is **based on CleanDiffuser**, with some modifications and extensions on top of their excellent work.  

Big thanks to the CleanDiffuser team for their great contribution to the community.  

Our goal is to refine the algorithm and enhance its capabilities.

---

## Installation

It is recommended to create an isolated environment using **conda**:

```bash
git clone https://github.com/modulardiffusion-design/Modular_diffusion.git
conda create -n modular_diff python=3.10 -y
conda activate modular_diff
cd Modular_diffusion
pip install -r requirements.txt
```

---

## MuJoCo Installation (Critical Step)
```
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210 ~/.mujoco/mujoco210
```

### 2. Set Environment Variables
Add the lines proposed by your computer to your `~/.bashrc` or `~/.zshrc`, for example:

```bash
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Then reload:
```bash
source ~/.bashrc
conda activate modular_diff
```

### 3. Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential libglew-dev libgl1-mesa-dev libosmesa6-dev patchelf libegl1 libgles2 libglx0 libopengl0
```

### 4. Install Python Dependencies
```bash
cd Modular_diffusion
pip install -r requirements.txt

# If mujoco-py compilation fails, try this specific installation order:
pip install "cython<3.0"
pip install --no-cache-dir mujoco-py==2.1.2.14
```

---



## Usage

To run the main program:
```bash
python pipelines/dql_d4rl_mujoco_GFDT.py
```
--
## Usage

To run the main program:
```bash
python pipelines/dql_d4rl_mujoco_GFDT.py
```
--
## Practical Techniques

Be wise in guidance selection. Check and select the guidance that offered a good normalized score in the inference stage. Do not blindly select the last .pt file, as many training trials (if not every trial) have shown visible and significant performance degradation after longer training steps, a clear sign of overfitting. The quality of guidance is vitally important for the performance of GFDT. 

---
# Project README

<details>
  <summary>📌 Common Issues & Solutions</summary>

If you encounter compilation errors with `mujoco-py`, try these solutions:

### Solution 1: Clean reinstall
```bash
pip uninstall mujoco-py
pip cache purge
pip install --force-reinstall --no-cache-dir mujoco-py==2.1.2.14
```

### Solution 2: Alternative mujoco-py version
```bash
pip install mujoco-py==2.3.3
```

### Solution 3: Manual compilation from source
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
```

---

**Q: Compilation fails with "GL/glew.h: No such file or directory"**  
**A:** Install the missing system dependencies:  
```bash
sudo apt install libglew-dev libgl1-mesa-dev
```

**Q: "command 'gcc' failed with exit status 1"**  
**A:** Ensure you have build tools:  
```bash
sudo apt install build-essential
```

**Q: Still having issues?**  
**A:** Please open an Issue with your error log and system information.

</details>

---

<details>
  <summary>🛠 Verification</summary>

Test your installation:
```bash
python -c "import mujoco_py; print('MuJoCo installation successful!')"
```

</details>

---

<details>
  <summary>🙌 Acknowledgements</summary>

This project is built upon CleanDiffuser.  
All credits for the dataset and benchmark go to the original authors.

</details>

---

<details>
  <summary>📜 License</summary>

This repository follows the original CleanDiffuser license.  
Please make sure to comply with the corresponding terms when using or distributing this project.

</details>
