# TopoDiT-3D
![TopoDiT-3D](https://github.com/Zechao-Guan/TopoDiT-3D/blob/main/figures/TopoDiT-3D-architecture.jpg)

The illustration of the proposed Topology-Aware Diffusion Transformer (TopoDiT-3D) for 3D point cloud generation. TopoDiT-3D initially voxelizes the point clouds, employing the patch operator to generate tokens related to the local point-voxel feature, and the persistence images to generate tokens related to the global topological feature. The persistence images are generated by the pretrained VAE during inference. Subsequently, TopoDiT-3D uses a fixed minor number of learned queries and the Perceiver Resampler to downsample and learn the topological and geometric information. After $N$ DiT-3D blocks, it uses the Perceiver Resampler to achieve upsampling, which recovers the same number of patch tokens to devoxelize.

## Result
| class  | 1-NNA | | COV | |
| -------|------ | -------|------ |
| class  | CD | EMD | CD | EMD |
| -------|------ | -------|------ |
| Chair  | Content Cell  | Chair  | Content Cell  |
| Airplane  | Content Cell  | Airplane  | Content Cell  |
| Airplane  | Content Cell  | Airplane  | Content Cell  |

## Useage
### install

### Train


### Test
```python
cd Partsegment
# train pointMLP
python train_TopoPointMLP_tailversion.py
# train pointNet++
python train_TopoPointNet2_headversion.py
```
