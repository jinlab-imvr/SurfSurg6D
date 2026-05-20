# **[IEEE Robotics and Automation Letters 2026] SurfSurg6D: Geometry Consistent Dense Correspondence for Textureless Surgical Instrument Pose Estimation**

_Daiyun Shen, Shuojue Yang, Chang Han Low, Qian Li, Mengya Xu, Qi Dou, Yueming Jin_

[![IEEE RA-L 2026](https://img.shields.io/badge/IEEE-RA--L%202026-blue)]()

 **Accepted by IEEE Robotics and Automation Letters (RA-L), 2026**  
 Project page / code: https://github.com/StackingDataYeti/SurfSurg6D  
 Code and dataset will be released at the project repository.

---

##  Overview

![SurfSurg6D architecture](Figures/main_fig_v7.pdf)

Surgical instrument pose estimation is important for robotic minimally invasive surgery, including autonomous robotic surgery, skill assessment, motion control, and surgical workflow standardization. However, reliable 6-DoF pose estimation remains challenging because surgical instruments are often textureless, reflective, partially occluded, and difficult to annotate with accurate 3D pose labels in real operating scenes.

**SurfSurg6D** addresses these challenges with a dense correspondence-based pose estimation framework tailored for surgical instruments. The method learns pixel-to-surface correspondences between RGB observations and instrument CAD surfaces, then estimates 6-DoF pose through correspondence matching and geometric pose solving.

The work also introduces **SynSurg6D**, a synthetic surgical instrument dataset generated from reconstructed surgical scenes and realistic instrument configurations.

### Key Features

- **Dense Surface Correspondence** – Learns correspondences between image pixels and 3D points on surgical instrument surfaces.
- **Instrument Latent Field** – Encodes normalized 3D surface coordinates into geometry-aware embeddings.
- **Image Recognition Network** – Predicts dense per-pixel embeddings from RGB image crops.
- **Distance-Aware Hard Negative Mining** – Emphasizes confusing but geometrically different surface points during contrastive learning.
- **Localized Correspondence Consistency** – Regularizes the implicit embedding field to respect local 3D geometry and reduce surface folding.
- **Synthetic-to-Real Training** – Uses SynSurg6D together with real data to improve pose estimation robustness and generalization.
- **RGB-Only Inference** – Estimates pose from RGB images without requiring depth at test time.

---

##  SynSurg6D Dataset

![SynSurg6D examples](Figures/fig1_4modal.png)

We introduce **SynSurg6D**, a synthetic dataset designed to alleviate the shortage of accurately annotated surgical instrument pose data.

### Dataset Properties

- **60K synthetic RGB-D images**
- **6 surgical instruments**
- **6-DoF pose annotations**
- **Segmentation maps**
- **Depth maps**
- **Normal maps**
- **Realistic reconstructed surgical backgrounds**
- **Randomized lighting and scene variation**

### Instrument Library

SynSurg6D contains six robotic surgical instruments:

1. Large Needle Driver
2. Scissors
3. Small Clip Applier
4. Cadiere Forceps
5. Biopsy Forceps
6. Permanent Cautery Spatula

### Realistic Surgical Configuration Constraints

To make the synthetic data closer to real robotic surgery, the rendering pipeline applies surgical configuration constraints, including:

- **Remote-center-of-motion (RCM) constraint**, requiring the instrument shaft to pass through a fixed trocar point.
- **Tool-tip depth constraint**, keeping the tip within a 30–100 mm depth range from the local tissue surface.
- **Articulated wrist modeling**, where distal wrist and clip angles are sampled according to surgical instrument kinematic structure.
- **Visibility filtering**, rejecting samples where the instrument leaves the field of view or the wrist visibility is too low.
- **Collision checking**, avoiding physically implausible instrument-scene configurations.
- **Lighting and background randomization**, increasing visual diversity and reducing overfitting.

![SynSurg6D variation](Figures/ral_addition.pdf)

---

## 🧠 Method

![Method overview](Figures/main_fig_v7.pdf)

SurfSurg6D follows a correspondence-based formulation. Given an RGB crop of a surgical instrument, the framework predicts dense image embeddings and matches them with 3D surface embeddings from the instrument model.

### Framework Components

1. **Instrument Latent Field**  
   A SIREN-based implicit MLP maps normalized 3D surface points to latent embeddings.

2. **Image Recognition Network**  
   A ResNet18 + U-Net image network predicts dense per-pixel embeddings and an instrument mask score.

3. **Contrastive Correspondence Learning**  
   Positive pixel-surface pairs are pulled together in the embedding space, while negative pairs are pushed apart using InfoNCE-style supervision.

4. **Hard Negative Pair Mining**  
   Surface points that are locally similar but geometrically distinct are given stronger penalties to improve discrimination on textureless and symmetric-like regions.

5. **Localized Embedding Consistency**  
   Neighboring surface points are regularized to have locally consistent embeddings, making the learned surface representation smoother and more geometry-aware.

6. **Pose Estimation**  
   During inference, 2D-3D correspondences are constructed by matching image embeddings to surface embeddings. The final 6-DoF pose is estimated from these correspondences.

---

##  Experimental Evaluation

SurfSurg6D is evaluated on three surgical benchmark datasets:

- **SurgRIPE** – Real surgical instrument pose estimation with 6-DoF pose annotations.
- **EndoVis2018** – Real surgical scenes with instrument segmentation masks.
- **SurgPose** – Surgical instrument keypoint annotations for 2D projection evaluation.

### Metrics

For 6-DoF pose estimation on SurgRIPE:

- **Rotation Error (RE)**
- **Translation Error (TE)**
- **ADD-10 Accuracy**
- **Average Accuracy from 0–5 mm**

For 2D projection evaluation:

- **Dice score** on EndoVis2018
- **mAP over OKS** on SurgPose

### Main Findings

- SynSurg6D consistently improves multiple pose estimation methods when combined with real training data.
- SurfSurg6D achieves robust pose estimation under occlusion and reflection.
- Compared with Surfemb, SurfSurg6D improves ADD-10 accuracy on SurgRIPE by introducing hard negative mining and consistency regularization.
- On EndoVis2018 and SurgPose, SynSurg6D improves generalization to real surgical scenes.

### 2D Projection Results

| Setting | Method | Dice on EndoVis2018 ↑ | mAP on SurgPose ↑ |
|---|---:|---:|---:|
| Zero-shot | MegaPose | 0.5463 | 0.0030 |
| Zero-shot | FoundPose | 0.7110 | 0.0976 |
| Real | MRC-Net | 0.7168 | 0.2678 |
| Real | GDR-Net | 0.7322 | 0.4693 |
| Real | Surfemb | 0.7566 | 0.5797 |
| Real | **SurfSurg6D** | 0.7555 | **0.6756** |
| Syn + Real | MRC-Net | 0.7550 | 0.6006 |
| Syn + Real | GDR-Net | 0.7653 | 0.5201 |
| Syn + Real | Surfemb | 0.7753 | 0.6385 |
| Syn + Real | **SurfSurg6D** | **0.7952** | **0.6897** |

### Ablation Study

| Method | RE ↓ | TE ↓ | ADD-10 Acc. ↑ |
|---|---:|---:|---:|
| Surfemb | 7.22° | 3.38 mm | 41.56% |
| + Hard Negative Mining | 7.03° | 3.20 mm | 42.06% |
| + Consistency Loss | 6.71° | 3.30 mm | 42.38% |
| **SurfSurg6D** | **6.42°** | **3.20 mm** | **42.89%** |

---

## Case Studies

![Pose estimation examples](Figures/pose_pic.pdf)

The supplementary video demonstrates SurfSurg6D under challenging real surgical conditions, including serious occlusion and strong reflection. Compared with Surfemb and FoundPose, SurfSurg6D shows more stable and robust pose estimation in these difficult frames.

![SurgPose keypoint projection](Figures/surgpose_2.pdf)

---

## 🎥 Video Demo

We provide a supplementary video demo for the SurfSurg6D pipeline.

<p align="center">
  <video width="90%" controls>
    <source src="Supplementary/demo.mp4" type="video/mp4">
  </video>
</p>

<p align="center">
  <em>Demo of SurfSurg6D for surgical instrument pose estimation, including motivation and qualitative comparison with Surfemb and FoundPose in real surgical scenes.</em>
</p>

> Minimum requirement: the video can be played by any MP4 player, such as VLC Media Player on Windows or macOS.

---

##  Expected Dataset Format

SurfSurg6D / SynSurg6D can be organized in a BOP-style or project-specific format containing:

```text
dataset_root/
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── mask/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── normal/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── scene_camera.json
├── scene_gt.json
├── scene_gt_info.json
└── models/
    ├── obj_000001.ply
    ├── obj_000002.ply
    └── ...
```

The essential inputs for pose training and evaluation are:

- RGB images
- Camera intrinsics
- Instrument CAD models
- 6-DoF pose annotations
- Object masks or bounding boxes
- Optional depth / normal / segmentation modalities for synthetic data analysis

---

##  Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{SurfSurg6D2026,
  author={Shen, Daiyun and Yang, Shuojue and Low, Chang Han and Li, Qian and Xu, Mengya and Dou, Qi and Jin, Yueming},
  journal={IEEE Robotics and Automation Letters},
  title={SurfSurg6D: Geometry Consistent Dense Correspondence for Textureless Surgical Instrument Pose Estimation},
  year={2026},
  pages={1-8},
  keywords={Pose estimation; Surgical scene synthesis; Surgical robotics; Contrastive learning; Data generation},
}
```

---

## Acknowledgement

This work was supported by Ministry of Education Tier 2 grant, Singapore (T2EP20224-0028), and Ministry of Education Tier 1 grant, Singapore (23-0651-P0001).

