# DiffRend: An Experimental Differentiable Renderer

## TODO list
* [ ] Check weight initialization (Florian)
* [x] Remove rendering from Shapenet dataset leader (David)
* [ ] Add conditional BatchNorm on Camera position and orientation to the generator (Florian)
* [ ] Improve Resnet generator/discriminator (Florian)
* [ ] Implement different generators/discriminators from DCGAN/WasserstainGAN/WasserstainGAN-GP (David)
* [ ] Play with Discriminator parameters (David)
* [x] Use hyperdash.io (david)
* [ ] Try plane bias (David)
* [ ] Implement Sphere bias (David)
* [ ] Implement Gaussian bias (Florian)
* [ ] Init Generator to produce a sphere (Fahim)

## Setup

```bash
python setup.py develop --user
```

## Usage

[//]: # (### OpenGL/Qt based rendering)

Command line:
```bash
python diffrend/render.py --use [gl|np|tf|tch] --scene <scene-description-filename>
```

### Using the TF based optimizer

```python
from diffrend.tensorflow.renderer import render

# generate splats

splat = ...

# setup the scene here
scene = splat_to_scene(splat)

# render
res = render(scene)
image = res['image']
im_depth = res['depth']

# use
# ...

```
For other versions use:

```python
from diffrend.numpy.renderer import render
```

## Code Organization

Numpy, Tensorflow and PyTorch versions of the renderer are in
```bash
diffrend/numpy
diffrend/tensorflow
diffrend/torch
```
folders respectively.

`model.py`: For loading and transforming models

## 3D Model and Geometric Primitives
Geometric primitives:
* Plane
* Disk
* Triangle
* Sphere


Mesh files: Currently OBJ and OFF only for triangle meshes.

## Lighting/Shading/Material


## Scene Description File
[Scene Description File Format](./docs/scene_description.md)

## Rendering

### Forward Pass


### Gradient-based Optimization


## Memory/Speed
In most cases there will be more pixels than geometric primitives.
Pixels are rendered independently of each other. So if the GPU runs out
of memory limit then tiled rendering might fix the memory issue.

