# FIT_Mesh

### Fit a mesh with Sign Distance Field represented by a MLP

```sh
    CUDA_VISIBLE_DEVICES=2 python fit_mesh.py --in_mesh example_data/bunny_rot.ply --expname bunny
```

### About the input mesh
!!! Please normalize the mesh within [-0.92, 0.92]^3, and center the mesh into origin !!!
Reference function `_normalize_mesh()` could be found in `data_utils.py`