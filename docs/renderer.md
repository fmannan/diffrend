# Renderer

## How it works

The main idea is that, for every pixel, get the nearest
intersection point, normal at that point, and the material properties.
In the current version, this is done by performing
vectorized ray-object intersection (i.e., batched operation but the
batch size being the size of the full dataset), and then masking out
invalid intersection.

This results in 4 (distance along ray, point, normal and material),
`M x N x D` matrices, where `M` is the number of objects, `N` the
number of pixels, and `D = 1` for distance along ray, and `3` for the
rest. The nearest object is then selected by taking the `argmin` along
`axis=0` of the ray-distance matrix.

The resulting indices are then used for constructing fragments,
by gathering the appropriate entries from the intersection point,
normal and material matrices. The rest is vectorized fragment shading.

The Tensorflow version is basically the numpy one with the numpy
operations replaced by Tensorflow functions (e.g., `np.sum` becomes
`tf.reduce_sum`, etc...but had to replace TF's cross prod with a
custom implementation, because TF's version can't operate on Tensors of
different shape like numpy!)

## Limitations and Improvements

The current version should work fine with a handful of geometric
primitives, but will run out of memory for thousands of triangle meshes.
There are two main improvements that need to be made. They are,

1. Batched rendering. The rays/pixels are independent of other rays,
so I'll render a batch of pixels at a time.

2. Ray and geometry culling. Not all geometries are visible from the
camera and not all rays intersect some geometry. Here I basically
have to use/modify some existing idea that doesn't break gradient based
optimization.

