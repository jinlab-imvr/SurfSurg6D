import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json

from .. import utils
from ..data import detector_crops
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose

# ---- Original inference logic unchanged ----
parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-poses', type=int, default=10000)
parser.add_argument('--max-pose-evaluations', type=int, default=1000)
parser.add_argument('--no-rotation-ensemble', dest='rotation_ensemble', action='store_false')
args = parser.parse_args()

device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()

model_name = model_path.stem
dataset = model_name.split('-')[0]

# Load model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(device)
model.freeze()


# Load data
root = Path('data/bop') / dataset
gt_folder = root /'test'

cfg = config[dataset]
objs, obj_ids = load_objs(root / cfg.model_folder)
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
data = detector_crops.DetectorCropDataset(
    dataset_root=root, cfg=cfg, obj_ids=obj_ids,
    detection_folder=Path(f'data/detection_results/{dataset}'),
    auxs=model.get_infer_auxs(objs=objs, crop_res=args.res_crop, from_detections=True)
)
renderer = ObjCoordRenderer(objs, w=args.res_crop, h=args.res_crop)

all_poses = np.empty((2, len(data), 3, 4))
all_scores = np.ones(len(data)) * -np.inf
time_forward, time_pnpransac, time_refine = [], [], []

for i, d in enumerate(tqdm(data, desc='running pose est.', smoothing=0)):
    if args.max_poses and i >= args.max_poses:
        break

    obj_idx = d['obj_idx']
    img = d['rgb_crop']
    K_crop = d['K_crop']

    # CNN + key extraction
    mask_lgts, query_img = model.infer_cnn(img, obj_idx, rotation_ensemble=args.rotation_ensemble)
    mask_lgts[0, 0].item()

    # Keys
    verts = surface_samples[obj_idx]
    verts_norm = (verts - objs[obj_idx].offset) / objs[obj_idx].scale
    obj_keys = model.infer_mlp(torch.from_numpy(verts_norm).float().to(device), obj_idx)
    verts = torch.from_numpy(verts).float().to(device)

    # PnP + RANSAC
    R_est, t_est, scores, *_ = estimate_pose(
        mask_lgts=mask_lgts, query_img=query_img,
        obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
        obj_diameter=objs[obj_idx].diameter, K=K_crop,
    )
    if len(scores) > 0:
        best_idx = torch.argmax(scores).item()
        R_p = R_est[best_idx].cpu().numpy()
        t_p=t_est[best_idx].cpu().numpy()[:,None]
        all_scores[i] = scores[best_idx].item()
    else:
        R_p, t_p = np.eye(3), np.zeros((3,1))

    # Refine
    if len(scores) > 0:
        R_r, t_r, _ = refine_pose(
            R=R_p, t=t_p, query_img=query_img, K_crop=K_crop,
            renderer=renderer, obj_idx=obj_idx, obj_=objs[obj_idx], model=model, keys_verts=obj_keys,
        )
        R_pred, t_pred = R_r, t_r
    else:
        R_pred, t_pred = R_p, t_p

    all_poses[0, i] = np.hstack((R_p, t_p.reshape(3,1)))
    all_poses[1, i] = np.hstack((R_pred, t_pred.reshape(3,1)))

    # Timing placeholders (not altered)

# Save outputs (unchanged)
results_dir = Path('data/results')
results_dir.mkdir(exist_ok=True)
np.save(results_dir / f'{model_name}-poses.npy', all_poses)
np.save(results_dir / f'{model_name}-poses-scores.npy', all_scores)
# timings saving omitted for brevity


def rotation_error(R_pred, R_gt):
    R_diff = R_pred.T @ R_gt
    cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def translation_error(t_pred, t_gt):
    return np.linalg.norm(t_pred.flatten() - t_gt.flatten())

def compute_add(R_pred, t_pred, R_gt, t_gt, verts, diameter, thr_ratio=0.1):
    pts_p = (R_pred @ verts.T + t_pred).T
    pts_g = (R_gt @ verts.T + t_gt).T
    dists = np.linalg.norm(pts_p - pts_g, axis=1)
    mean_add = dists.mean()
    return mean_add < thr_ratio * diameter, mean_add

# Load ground-truth
model_path='./joint.npy'
verts_np=np.load(model_path)
errors = []
for i, d in enumerate(data):
    scene_id = d['scene_id']
    obj_idx = d['obj_idx']
    # parse scene file
    with open(gt_folder / f"{scene_id:06d}/scene_gt.json") as f:
        scene = json.load(f)

    start_num=0
    if scene_id==2:
        start_num=238
    R_gt=np.array(scene[str(i-start_num)][0]['cam_R_m2c']).reshape(3,3)
    t_gt =np.array(scene[str(i-start_num)][0]['cam_t_m2c']).reshape(3,1)
    R_pred = all_poses[1, i, :3, :3]
    t_pred = all_poses[1, i, :3, 3:].reshape(3,1)
    rot_e = rotation_error(R_pred, R_gt)
    trans_e = translation_error(t_pred, t_gt)
    diam=16.24
    add_succ, add_val = compute_add(R_pred, t_pred, R_gt, t_gt, verts_np, diam)
    errors.append((rot_e, trans_e, add_val, add_succ))

# Summarize and dump
rots, trans, adds, succs = zip(*errors)
summary = {
    'mean_rot_error': float(np.mean(rots)),
    'mean_trans_error': float(np.mean(trans)),
    'mean_add': float(np.mean(adds)),
    'add_10_accuracy': float(np.mean(succs))
}
with open(results_dir / f'{model_name}-metrics.json', 'w') as mf:
    json.dump(summary, mf, indent=2)

print('Metrics:', summary)
'''
poses_fp = Path('data/results') / f'{model_name}-poses.npy'
scores_fp = Path('data/results') / f'{model_name}-poses-scores.npy'
timings_fp = Path('data/results') / f'{model_name}-poses-timings.npy'

np.save(str(poses_fp), None)  # placeholder, or retain previous save logic
np.save(str(scores_fp), None)
np.save(str(timings_fp), None)
'''
