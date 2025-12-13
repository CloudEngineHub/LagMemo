
# **LagMemo**: **La**nguage 3D **G**aussian Splatting **Me**mory for **M**ulti-modal **O**pen-vocabulary Multi-goal Visual Navigation

> **Current Scope:** This repository currently focuses on the **Visual Navigation** implementation of LagMemo. It provides the codebase for reproducing navigation experiments using pre-computed representations and goal lists.

## 📅 News
- **[2025/12/13]** 🚀 **Open Source Release:** The navigation evaluation code for LagMemo is now available! Includes environment setup, `glue_agent` implementation, and baseline configurations.

---

## 📝 Project Status & Roadmap

### 🚀 Core Features (Navigation)
- [x] **LagMemo Agent:** Implementation of the glue-based navigation agent (`glue_agent.py`).
- [x] **Goal Verification:** Multi-goal consistency check and termination logic.
- [x] **Environment:** Habitat-Lab integration with custom sensors.

### 🧪 Baselines & Comparisons
- [ ] **Frontier Exploration:** Integrated for exploration baselines.
- [ ] **Vlmaps:** Baseline implementation for comparison.

### 🛠 Code & Data
- [x] **Refactoring:** Fixed similarity calculation bugs and improved stability.
- [ ] **Data Recording:** Validated on new 3-episode scene datasets.
- [ ] **Mapping Module:** (Planned) Code for generating 3DGS maps from raw trajectories.

---

## Data

please download goat episode dataset from [here](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view?usp=sharing), and put it as ***/data/datasets/goat/hm3d/...***

```bash
# scene datasets
ln -s hm3d_path /path/to/data/scene_datasets/hm3d

```

This repository focuses on multi-goal visual navigation and goal verification mechanism using pre-computed navigation waypoints.
Please download the required data files from [Link](https://disk.pku.edu.cn/link/AA6BD829693D7E4987B6870878EE5C57F8) and place them in the `data/` directory:
1.  **`lagmemo_goal.json`**: Contains waypoints calculated by the mapping and memory module.
2.  **`goal_list.json`**: A list of goals required to ensure the program runs without errors.

The file structure should be as follow:

```bash
.
├── data
│   ├── goal_list.json
│   ├── lagmemo_goal.json
│   ├── datasets
│   │   └── goat
│   │   |   └── hm3d
|   |   |   |   └── v1
|   │   │   │   │   ├── train
|   │   │   │   │   ├── val_seen
|   │   │   │   │   ├── val_unseen
|   │   │   │   │   └── val_seen_synonyms
│   ├── scene_datasets
│   │   └── hm3d
│   │   │   ├── 00800-TEEsavR23oF
│   │   │   ├── ...
```



<!-- You can find predownloaded 2D-map for two-stage 3dgs navigation in [global_map_seem](https://disk.pku.edu.cn/link/AA96EFEAD6141C43CE88B2ECD6487E0534), put it on root directory. -->

## Installation

<!-- **If you get an error when following this tutorial, please read the `Problem` section first before taking next action** -->

### 1. Clone codes and create Environments
```bash
# git clone our codes and switch branch to 3dgs
# git clone https://github.com/happywangmakeit/lagmemo.git
# cd lagmemo
# git checkout 3dgs
# git clone & cd lagmemo

# create conda env
conda create -n lagmemo python=3.9
conda activate lagmemo
```

### 2. CUDA Configuration
This project requires CUDA 11.8. Please verify your setup before proceeding.

1. **Check Version**: Run nvcc -V to check your current compiler version.

2. **Install/Switch (If needed)**:

    * Option A: install the CUDA Toolkit 11.8 directly within your Conda environment
    
        ```bash
        conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
        ```

    * Option B (System-level): If you have CUDA 11.8 installed at /usr/local/cuda-11.8, switch using environment variables:

        ```bash
        export CUDA_HOME=/usr/local/cuda-11.8
        export PATH=${CUDA_HOME}/bin:${PATH}
        export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
        ```

### 3. Install GCC/G++ 11 Compiler
```bash
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
```

check version, make sure GCC 11
```bash
x86_64-conda-linux-gnu-cc --version
x86_64-conda-linux-gnu-c++ --version
```


### 4. install dependencies
```bash
conda env update --name lagmemo --file environment.yml
pip install -r requirements.txt
conda install -c conda-forge pynput
```

### 5. install navigation agents
```bash
# Install the core package
python -m pip install -e src/lagmemo
```

### 6. install submodules
```bash
# initialize submodules
git submodule update --init --recursive 
# src/lagmemo/lagmemo/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork src/third_party/frontier_exploration

# install specific pytorch 
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# detectron2 module
cd src/third_party
python -m pip install -e detectron2 --no-build-isolation 
# # Detic module (not used)
# cd ../..
# cd src/lagmemo/lagmemo/perception/detection/detic/Detic/
# pip install -r requirements.txt
# mkdir models
# wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate
# # you should run demo if env correctly
# wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
# python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
# mkdir -p data/checkpoints
# cd data/checkpoints
# wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
# unzip ovmm_baseline_home_robot_challenge_2023.zip
cd LAGMEMO_ROOT # return to repo's root, should be changed to your own path
# simulation environment
conda env update -f src/environment.yml
# habitat environment
git submodule update --init --recursive src/third_party/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-lab
# don't be panic if get pip conflicts
python -m pip install -e src/third_party/habitat-lab/habitat-baselines # if pip conflict, ensure numpy==1.23.5 moviepy==1.0.3
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git" # this is not neccessary if you have pytorch3d in your pip list
# install frontier_exploration module
pip install -e src/third_party/frontier_exploration
```

### 7.CLIP and SEEM Installation

### Mobile-CLIP
```bash
cd src/third_party/ml-mobileclip
```
Open requirements.txt and modify it as:

```bash
clip-benchmark>=1.4.0
datasets>=2.21.0
open-clip-torch>=2.20.0
timm>=0.9.5
# torch>=2.1.0
# torchvision>=0.14.1
```

To install it:

```bash
pip install -e . # don't be panic if pip conflicts, see problems to use specific version
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.

cd ../../..

```

<!-- Try ```python project/habitat_lagmemo/eval_episode.py```, if get an error, reinstall habitat:

```bash
cd src/third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
``` -->


### SEEM

```bash
cd src/third_party/seem
# if empty, try to relink submodules: git submodule add -f https://github.com/happywangmakeit/seem.git src/third_party/seem
```

```bash
conda install -c conda-forge mpi4py mpich
# maybe python version changed
# pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# don't be panic if pip conflicts, see problems to use specific version
pip install -r requirements_our.txt
pip install -e .
```

<!-- Please follow the recommand at [seem_for_lagmemo](https://github.com/fflahm/seem_for_lgs) `Download model checkpoints` chapter. -->
- Download https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt to `src/third_party/seem/checkpoints/`

- **Optional**: Since huggingface connection may be unstable, it is recommended to use local checkpoints of CLIP tokenizer. 
    
    - To do this, download CLIP tokenizer with git

        ```sh
        cd ./checkpoints
        git lfs install
        git clone https://huggingface.co/openai/clip-vit-base-patch32
        ```

    - After this, modify `src/third_party/seem/src/seem/modeling/language/LangEncoder/__init__.py`

    - from

      ```python
          if config_encoder['TOKENIZER'] == 'clip':
              pretrained_tokenizer = config_encoder.get(
                  'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
              )
              tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
              tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
      ```

    - to

      ```python
          if config_encoder['TOKENIZER'] == 'clip':
              pretrained_tokenizer = config_encoder.get(
                  'PRETRAINED_TOKENIZER', 'src/third_party/seem/checkpoints/clip-vit-base-patch32'
              )
              tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
              tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
      ```

- **After install them, please change the relative args in `eval_episode.py`**

# ready to start!
```bash
python project/habitat_lagmemo/eval_episode_glue.py
# output path is written in /project/config/agent/hm3d_eval.yaml DUMP_LOCATION
```

## Problems

#### If you have problem with numpy 2.0.2 when installing habitat_lab, and habitat_lab is installed successfully

```bash
pip install numpy==1.23.5 # and continue next step
```

#### When having pip conflicts when installing **webdataset**, **huggingface-hub**, **pyarrow**, and **timm** libraries

Don't panic if you encounter pip installation conflicts, this is a normal occurrence and won't affect the program's execution. Just use the following library versions: 

* webdataset: 0.1.40
* huggingface-hub: 0.17.3
* pyarrow: 13.0.0
* timm: 0.4.12

#### If the `SuperGluePretrainedNetwork` directory is empty.
```bash
# Synchronize submodule configuration URLs
git submodule sync src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork

# Force update and initialization of the submodule
git submodule update --init --force src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork
```