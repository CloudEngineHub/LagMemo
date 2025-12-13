
## This is a project for Lagmemo

- **Update 2025/5/23:** Provide 2D-map download path for two-stage 3dgs-nav, and the installation method of the CLIP and SEEM is given.

- **Usage of latest code please refer to glue_agent.py**

## Progress Status 2025.8.11

### Code
- [x] Fixing similarity bug and re-test

### Experiment
- [x] Lagmemo  
- [x] Vlmaps* 
- [x] Frontier Exploration*  

### Others
- [x] Data record for new 3-episodes scenes

## Data

please download goat episode dataset from [here](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view?usp=sharing), and put it as ***/data/datasets/goat/hm3d/...***

```bash
# scene datasets
ln -s hm3d_path /path/to/data/scene_datasets/hm3d

```

The file structure should be as follow:

```bash
.
├── data
│   ├── goal_list.json
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

You can find predownloaded 2D-map for two-stage 3dgs navigation in [global_map_seem](https://disk.pku.edu.cn/link/AA96EFEAD6141C43CE88B2ECD6487E0534), put it on root directory.

## Installation

**If you get an error when following this tutorial, please read the `Problem` section first before taking next action**

```bash

# create conda env
conda env create -n lagmemo -f environment.yml

conda activate lagmemo

# Install the core package
python -m pip install -e src/lagmemo

# initialize submodules
git submodule update --init --recursive 
# src/lagmemo/lagmemo/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork src/third_party/frontier_exploration

# dection module
cd src/third_party
python -m pip install -e detectron2 # torch2.1.2+cu118 is available if get error here, and some mistake maybe caused by cpu version torch, please pay attention, refer to Problem section
cd ../..

cd src/lagmemo/lagmemo/perception/detection/detic/Detic/
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate

# you should run demo if env correctly
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

mkdir -p data/checkpoints
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd LAGMEMO_ROOT # return to repo's root, should be changed to your own path

# simulation environment
conda env update -f src/environment.yml

git submodule update --init --recursive src/third_party/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git" # this is not neccessary if you have pytorch3d in your pip list

# switch to goat branch
cd src/third_party/habitat-lab
git checkout home-robot_goat_support
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../../..

# really to start!
python project/habitat_lagmemo/eval_episode.py

```

## CLIP and SEEM Installation

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
pip install -e .
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.

cd ../../..

```

Try ```python project/habitat_lagmemo/eval_episode.py```, if get an error, reinstall habitat:

```bash
cd src/third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```


### SEEM

```bash
cd src/third_party/seem
```

```bash
conda install -c conda-forge mpi4py mpich
# maybe python version changed
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# maybe there are some empty package because of cloning env, please handle it
pip install -r requirements_our.txt

```

```bash
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

## Problem

#### When having problem with installing detectron2:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch # install torch2.5.1
python -m pip install -e detectron2
conda uninstall libtorch # downgrade torch version
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 # install torch2.1.2 cu118 which is confirmed to be correct

```

#### When having problem with "AttributeError:'dict'object has no attribute 'env_specs'"

Change all registry.env_specs to registry.keys()

#### If you have problem with numpy 2.0.2 when installing habitat_lab, and habitat_lab is installed successfully

```bash
pip install numpy==1.23.5 # and continue next step
```

#### When having problem with command "python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git""

It's not neccessary if you have pytorch3d in your pip list

#### When having problem with "import sophus"

Change "sophus" to "sophuspy"

#### When having pip conflicts when installing **webdataset**, **huggingface-hub**, **pyarrow**, and **timm** libraries

Don't panic if you encounter pip installation conflicts, this is a normal occurrence and won't affect the program's execution. Just use the following library versions: 

* webdataset: 0.1.40
* huggingface-hub: 0.17.3
* pyarrow: 13.0.0
* timm: 0.4.12