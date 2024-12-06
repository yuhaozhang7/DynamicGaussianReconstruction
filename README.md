### INSTALLATION INSTRUCTIONS

- create an environment using python 3.10 (also works with python 3.7 but requires some changes to some packages)
- for the next two steps, install everything to the environment you created in the last step
- go to [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) codebase and follow the installation instructions (except the python version)
- go to [MVSplat](https://github.com/donydchen/mvsplat) codebase and follow the installation instructions

### Train

```shell
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender
```
There are several important parameters:
 - num_views : the number of context views used in the Multi-view transformer, default is 2
 - warmup : number of iterations the model will be trained without deformation at the start, after that the deformation network will start training, default=4000
 - pretrained_backbone_path : the path to the pretrained multi-view transformer backbone provided at the MVSplat codebase, if not provided, the model will be created from scratch
 - use_depth : if used, the depth predictor of the MVSplat will be used instead of the MLP we implemented. usage : --use_depth
 - -s : path to the dataset
 - -m : path to the output

### Render & Evaluation

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```

We provide several modes for rendering:

- `render`: render all the test images
- `time`: time interpolation tasks for D-NeRF dataset
- `all`: time and view synthesis tasks for D-NeRF dataset
- `view`: view synthesis tasks for D-NeRF dataset
- `original`: time and view synthesis tasks for real-world dataset

There are couple of important arguments for the render.py:

- use_depth : if used, the depth predictor will be used
- num_views : number of context views of he Multi-view Transformer, must be the same with the one used in training, REQUIRED
- path_model : path to the model that will be used for evaluation, REQUIRED 


