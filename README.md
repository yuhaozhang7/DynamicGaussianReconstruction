# Improved Dynamic Gaussian Reconstruction with Multi-Scene Training

This project is forked from [Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://github.com/ingra14m/Deformable-3D-Gaussians).

![Our Teaser image](assets/teaser.gif)
![Our Teaser image](assets/teaser2.gif)

- [ ] Change sub-folder name
- [ ] Change the whole repo name (check git clone command below)
- [ ] Check if all modes of rendering supported

This repository contains the implementation for the project **Improved Dynamic Gaussian Reconstruction with Multi-Scene Training**. In this project, we aim to develop a model to predict Gaussians for a dynamic monocular scene to allow novel view synthesis.


## Dataset

In our project, we mainly use:

- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html).
- real-world dataset from [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) and [Hyper-NeRF](https://hypernerf.github.io/).

You could organize the datasets as follows, recommended by the authors of Deformable 3D Gaussians:

```shell
├── data
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```

## Run

### Environment

This project is an incomplete clone of the original one. To set up the environment, please set up an environment using the original repository:
```shell
git clone https://github.com/ingra14m/Deformable-3D-Gaussians --recursive
cd Deformable-3D-Gaussians

conda create -n deformable_gaussian_env python=3.7
conda activate deformable_gaussian_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```

Then, you could clone our repository:
```shell
git clone git@github.com:Andy-Zhou2/MVP_Course_Project.git
```


### Train

**D-NeRF:**

```shell
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender
```


### Render & Evaluation

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```

We have modified the code to support several modes for rendering:

- `render`: render all the test images
- `time`: time interpolation tasks for D-NeRF dataset
- `all`: time and view synthesis tasks for D-NeRF dataset
- `view`: view synthesis tasks for D-NeRF dataset
- `original`: time and view synthesis tasks for real-world dataset


