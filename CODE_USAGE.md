# Source code usage

The `surfsurg6d/` package contains the correspondence model, data loaders,
training, inference, pose refinement, and supporting tools. The training
entry point included in this source snapshot is `train_blend.py`.

## Environment

`environment.yml` contains the environment specification from the source
project. A checked inference environment used Python 3.10, PyTorch 2.3.1,
torchvision 0.18.1, PyTorch Lightning 2.4.0, and torch-scatter 2.1.2.
Install CUDA-compatible versions of PyTorch, torchvision, and torch-scatter
for your machine. The renderer uses ModernGL with the EGL backend.

Run module commands from the repository root so that the `surfsurg6d`
package and relative `data/` paths are available.

## Data layout

Prepare a BOP-style dataset and its mesh and surface samples:

```text
data/
  bop/<dataset>/
    models/
      models_info.json
      obj_000000.ply
      ...
    train/<scene_id>/
      rgb/
      mask_visib/
      scene_camera.json
      scene_gt.json
      scene_gt_info.json
    train_pbr/<scene_id>/
      rgb/
      depth/
      mask_visib/
      scene_camera.json
      scene_gt.json
      scene_gt_info.json
    test/<scene_id>/
      rgb/
      scene_camera.json
      scene_gt.json
      scene_gt_info.json
  detection_results/<dataset>/
    bboxes.npy
    obj_ids.npy
    scene_ids.npy
    view_ids.npy
    scores.npy
    times.npy
  surface_samples/<dataset>/obj_000000.ply
  surface_samples_normals/<dataset>/obj_000000.ply
  models/
  results/
```

Mesh metadata must contain each object's diameter. Detection arrays must
have aligned first dimensions; bounding boxes use `[x_min, y_min, x_max,
y_max]`. The checkpoint filename prefix determines the dataset name:
`<dataset>-<run_id>.ckpt`.

Datasets, trained checkpoints, cached predictions, and machine-specific data
links are separate runtime inputs.

## Training

For PBR training, replace `DATASET` with your dataset name:

```bash
CUDA_VISIBLE_DEVICES=0 python -m surfsurg6d.scripts.train_blend DATASET
```

Add `--real` to combine `train_pbr/` with the real `train/` split. The script
balances the dataset sizes by repeating the smaller dataset. It uses
`surface_embedding_blend.SurfaceEmbeddingModel`, with `--alpha` and
`--lambda_consistency` both defaulting to 1. Training logs use Weights & Biases.

## RGB inference

```bash
python -m surfsurg6d.scripts.infer_new \
  data/models/DATASET-RUN_ID.ckpt --device cuda:0
```

This entry point writes pose arrays, scores, and metrics to `data/results/`.
Its metric section currently expects the two reference test scenes
`000001` and `000002` with 238 and 373 detections in that order, object 0,
and a root-level `joint.npy` point cloud. It uses a reference diameter of
16.24 mm. Adapt the metric section when evaluating another dataset.

For pose inference without that reference metric section:

```bash
python -m surfsurg6d.scripts.infer \
  data/models/DATASET-RUN_ID.ckpt --device cuda:0
```

`infer.py` writes poses, scores, and timings to `data/new_results/`.

## Tools

The `surfsurg6d.scripts.misc` modules provide surface sampling, normal
recovery, remeshing, pose rendering, checkpoint compaction, detection
conversion, and BOP CSV export. The depth-refinement entry point is
`surfsurg6d.scripts.infer_refine_depth`; it additionally needs depth images
and existing pose and timing arrays under `data/results/`.

`load_detection_results` is a legacy CosyPose conversion helper and needs
an external CosyPose installation and a matching local result path.

## Third-party attribution

The source package includes code adapted from SurfEmb. Its original MIT
license notice is retained in `THIRD_PARTY_LICENSES/SurfEmb-LICENSE.txt`.
