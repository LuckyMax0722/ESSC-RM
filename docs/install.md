# Step-by-step installation instructions

CGFormer is developed based on the official OccFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**
```shell
conda create -n ESSC python=3.8 -y
conda activate ESSC
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

**c. clone**
```bash
git clone
```
**c. Install other dependencies, like timm, einops, torchmetrics, spconv, pytorch-lightning, etc.**

```shell
cd docs
pip install -r requirements.txt
```

**d. Install Natten**
```shell
pip3 install natten==0.17.4+torch200cu117 -f https://shi-labs.com/natten/wheels
```

**d. OpenMMLab**
```shell
pip install mmengine==0.10.6
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
#pip install mmdet==3.3.0
#pip install mmdet3d==1.4.0
```

**e. Install ESSC-RM**
```bash
cd ..
pip install -e .
```

**e. Fix bugs (known now)**
