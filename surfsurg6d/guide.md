
## Setup

```bash
git clone https://github.com/jinlab-imvr/SurfSurg6D.git
cd SurfSurg6D
conda env create -f environment.yml
conda activate surfsurg6d
```


## Dataset

Obtain synthetic data from the [dataset page](https://huggingface.co/datasets/Daiyun/SurfSurg6D); the repository README describes the upload as partial. For real data access(SurgRIPE), go to [Synapse](https://www.synapse.org/Synapse:syn51471789).

Organize the data in BOP style. Synthetic training uses `train_pbr/`, real training uses `train/`, and inference uses `test/`. Dataset assets and checkpoints are supplied separately from the source.

```text
SurfSurg6D/
├── joint.npy                         # Reference evaluation points
└── data/
    ├── bop/ycbv_ori/s
    │   ├── models/
    │   │   ├── models_info.json
    │   │   └── obj_XXXXXX.ply
    │   ├── train_pbr/SCENE/
    │   │   ├── rgb/000000.png
    │   │   ├── mask_visib/000000_000000.png
    │   │   ├── scene_camera.json
    │   │   ├── scene_gt.json
    │   │   └── scene_gt_info.json
    │   ├── train/SCENE/               # Same layout as train_pbr
    │   └── test/SCENE/
    │       ├── rgb/000000.png
    │       ├── scene_camera.json
    │       └── scene_gt.json         # Required for evaluation
    ├── detection_results/ycbv_ori/    # Detection .npy arrays
    ├── surface_samples/ycbv_ori/     # Surface-point PLY files
    ├── surface_samples_normals/ycbv_ori/
    ├── models/                       # Training checkpoint destination
    ├── new_results/                  # infer.py outputs
    └── results/                      # infer_new.py outputs
```

Use six-digit scene/image IDs, such as `000001`. Each camera entry contains row-major `cam_K`; each pose contains `obj_id`, `cam_R_m2c`, and `cam_t_m2c`. Training additionally reads visible masks and `bbox_obj`, `bbox_visib`, `visib_fract`, and `px_count_visib` from `scene_gt_info.json`. Mask filenames use the image ID and pose-list index.

Keep object IDs consistent across meshes, annotations, detections, and the checkpoint. `models_info.json` must include each object's diameter. Meshes, translations, and evaluation points must use the same coordinates and units; the reference setup uses millimeters. Check BOP for reference.

## Surface Samples and Detections

```bash
export DATASET=ycbv_ori
mkdir -p data/surface_samples data/surface_samples_normals
python -m surfsurg6d.scripts.misc.surface_samples_remesh_visible "$DATASET"
python -m surfsurg6d.scripts.misc.surface_samples_sample_even "$DATASET" --n-samples 50000
python -m surfsurg6d.scripts.misc.surface_samples_recover_normals "$DATASET"
```

Prepare aligned `bboxes.npy` `(N, 4)`, `obj_ids.npy`, `scene_ids.npy`, and `view_ids.npy` `(N,)`. Boxes use original-image pixels in `[x_min, y_min, x_max, y_max]` order. Visualization and CSV export also require `scores.npy` and `times.npy`. The release does not include a detector runner.

## Training

```bash
export DATASET=ycbv_ori # can be replaced with your dataset name
export WANDB_MODE=offline  # optional            
CUDA_VISIBLE_DEVICES=0 python -m surfsurg6d.scripts.train_blend "$DATASET" \
  --real --batch-size 20 --num-workers 4 --max-epochs 200
```

`--real` to introduce real data; `--no-synth` to forbid using the synthetic data.  

Checkpoints are written to `data/models/DATASET-RUN_ID.ckpt`. 
## Inference

Set the checkpoint path. Its filename prefix determines the dataset directory:

```bash
export MODEL_DIR=YOUR_MODEL_DIR
export CHECKPOINT="$MODEL_DIR/ycbv_ori-YOUR_RUN_ID.ckpt"
mkdir -p data/new_results data/results

python -m surfsurg6d.scripts.infer "$CHECKPOINT" --device cuda:0
```

`infer.py` writes poses, scores, and timings to `data/new_results/`. Poses have shape `(2, N, 3, 4)`: index `0` is the initial estimate, index `1` is refined, and row `i` corresponds to detection row `i`.
We have uploaded a model trained with syn+real, check https://huggingface.co/Daiyun/SurfSurg6D.

## Reference ADD-10 Evaluation

For the existing reference split, run:

```bash
python -m surfsurg6d.scripts.infer_new "$CHECKPOINT" \
  --device cuda:0 --max-poses 0
```

This requires object-0 model-frame points in root-level `joint.npy` and the following detection order:

| Subset | Scene | Rows | View IDs |
| --- | --- | --- | --- |
| OCC | `000001` | 0–237 | 0–237 |
| NON-OCC | `000002` | 238–610 | 0–372 |

The dataset split is according to the SurgRIPE dataset. Results are saved in `data/results/` as `MODEL_NAME-poses.npy`, `MODEL_NAME-poses-scores.npy`, and `MODEL_NAME-metrics.json`. `add_10_accuracy` is the combined accuracy as a fraction. Separate OCC/NON-OCC scores require evaluating refined poses against their matched ground truth and grouping by scene.

