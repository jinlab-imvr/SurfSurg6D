import argparse
from typing import Sequence, Union
from pathlib import Path
import torch
import subprocess
from torch.nn import functional as F
import numpy as np
import wandb
import albumentations as A
import cv2
import pytorch_lightning as pl
import json
from . import data
from .dep.unet import ResNetUNet
from .dep.siren import Siren
from .data.obj import Obj
from .data.tfms import denormalize
from . import utils

# could be extended to allow other mlp architectures
mlp_class_dict = dict(
    siren=Siren
)
def weighted_distance_consistency_loss(coords_pos: torch.Tensor, keys_pos: torch.Tensor, 
                                    distance_scale: float = 10.0, min_weight: float = 0.01):
    """
    Vectorized batch loss: embedding distance should follow coord distance, weighted by proximity.

    Args:
        coords_pos: (B, n_pos, 3)
        keys_pos: (B, n_pos, D)
    Returns:
        Scalar loss
    """
    B, n_pos, _ = coords_pos.shape

    # 1. 距离矩阵
    pdist_coord = torch.cdist(coords_pos, coords_pos, p=2)  # (B, n_pos, n_pos)
    pdist_embed = torch.cdist(keys_pos, keys_pos, p=2)      # (B, n_pos, n_pos)

    # 2. 提取上三角 (排除对角线)
    triu_mask = torch.triu(torch.ones(n_pos, n_pos, device=coords_pos.device), diagonal=1).bool()
    coord_dist = pdist_coord[:, triu_mask]  # (B, n_pairs)
    embed_dist = pdist_embed[:, triu_mask]  # (B, n_pairs)

    # 3. 归一化
    coord_dist_norm = coord_dist / (coord_dist.max(dim=1, keepdim=True).values + 1e-8)
    embed_dist_norm = embed_dist / (embed_dist.max(dim=1, keepdim=True).values + 1e-8)

    # 4. 权重 = exp(-scale * dist)
    weights = torch.exp(-distance_scale * coord_dist_norm)
    weights = torch.clamp(weights, min=min_weight)

    # 5. loss
    loss_per_batch = weights * (coord_dist_norm - embed_dist_norm).pow(2)  # (B, n_pairs)
    loss = loss_per_batch.mean()
    return loss


class SurfaceEmbeddingModel(pl.LightningModule):
    def __init__(self, n_objs: int, emb_dim=12, n_pos=1024, n_neg=1024, lr_cnn=3e-4, lr_mlp=3e-5,alpha=1,lambda_consistency=1,
                 mlp_name='siren', mlp_hidden_features=256, mlp_hidden_layers=2,
                 key_noise=1e-3, warmup_steps=2000, separate_decoders=True,dataset:str=None,run_id:str=None,
                 **kwargs):
        """
        :param emb_dim: number of embedding dimensions
        :param n_pos: number of positive (q, k) pairs from the object mask
        :param n_neg: number of negative keys, k-, from the object surface
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset=dataset
        self.run_id=run_id
        self.n_objs, self.emb_dim = n_objs, emb_dim
        self.n_pos, self.n_neg = n_pos, n_neg
        self.lr_cnn, self.lr_mlp = lr_cnn, lr_mlp
        self.warmup_steps = warmup_steps
        self.key_noise = key_noise
        self.separate_decoders = separate_decoders
        self.lambda_consistency=lambda_consistency
        self.alpha=alpha
        # query model
        self.cnn = ResNetUNet(
            n_class=(emb_dim + 1) if separate_decoders else n_objs * (emb_dim + 1),
            n_decoders=n_objs if separate_decoders else 1,
        )
        # key models
        mlp_class = mlp_class_dict[mlp_name]
        mlp_args = dict(in_features=3, out_features=emb_dim,
                        hidden_features=mlp_hidden_features, hidden_layers=mlp_hidden_layers)
        self.mlps = torch.nn.Sequential(*[mlp_class(**mlp_args) for _ in range(n_objs)])

    @staticmethod
    def model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group(SurfaceEmbeddingModel.__name__)
        parser.add_argument('--emb-dim', type=int, default=12)
        parser.add_argument('--single-decoder', dest='separate_decoders', action='store_false')
        return parent_parser

    def get_auxs(self, objs: Sequence[Obj], crop_res: int):
        random_crop_aux = data.std_auxs.RandomRotatedMaskCrop(crop_res)
        return (
            data.std_auxs.RgbLoader(),
            data.std_auxs.MaskLoader(),
            random_crop_aux.definition_aux,
            # Some image augmentations probably make most sense in the original image, before rotation / rescaling
            # by cropping. 'definition_aux' registers 'AABB_crop' such that the "expensive" image augmentation is only
            # performed where the crop is going to be taken from.
            data.std_auxs.TransformsAux(key='rgb', crop_key='AABB_crop', tfms=A.Compose([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.ISONoise(),
                A.GaussNoise(),
                data.tfms.DebayerArtefacts(),
                data.tfms.Unsharpen(),
                A.CLAHE(),  # could probably be moved to the post-crop augmentations
                A.GaussianBlur(blur_limit=(1, 3)),
            ])),
            random_crop_aux.apply_aux,
            data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=True),
            data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
            data.pose_auxs.MaskSamplesAux(self.n_pos),
            data.std_auxs.TransformsAux(tfms=A.Compose([
                A.CoarseDropout(max_height=16, max_width=16, min_width=8, min_height=8),
                A.ColorJitter(hue=0.1),
            ])),
            data.std_auxs.NormalizeAux(),
            data.std_auxs.KeyFilterAux({'rgb_crop', 'obj_coord', 'obj_idx', 'surface_samples', 'mask_samples'})
        )

    def get_infer_auxs(self, objs: Sequence[Obj], crop_res: int, from_detections=True):
        auxs = [data.std_auxs.RgbLoader()]
        if not from_detections:
            auxs.append(data.std_auxs.MaskLoader())
        auxs.append(data.std_auxs.RandomRotatedMaskCrop(
            crop_res, max_angle=0,
            offset_scale=0 if from_detections else 1,
            use_bbox=from_detections,
            rgb_interpolation=(cv2.INTER_LINEAR,),
        ))
        if not from_detections:
            auxs += [
                data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=True),
                data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
                data.pose_auxs.MaskSamplesAux(self.n_pos),
            ]
        return auxs

    def configure_optimizers(self):
        opt = torch.optim.Adam([
            dict(params=self.cnn.parameters(), lr=1e-4),
            dict(params=self.mlps.parameters(), lr=3e-5),
        ])
        sched = dict(
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / self.warmup_steps)),
            interval='step'
        )
        return [opt], [sched]

    def step(self, batch, log_prefix):
        img = batch['rgb_crop']  # (B, 3, H, W)
        coord_img = batch['obj_coord']  # (B, H, W, 4) [-1, 1]
        obj_idx = batch['obj_idx']  # (B,)
        coords_neg = batch['surface_samples']  # (B, n_neg, 3) [-1, 1]
        mask_samples = batch['mask_samples']  # (B, n_pos, 2)

        device = img.device
        B, _, H, W = img.shape
        assert coords_neg.shape[1] == self.n_neg
        mask = coord_img[..., 3] == 1.  # (B, H, W)
        y, x = mask_samples.permute(2, 0, 1)  # 2 x (B, n_pos)
        coords_pos = coord_img[torch.arange(B).view(B, 1), y, x, :3]  # (B, n_pos, 3)


        coords_pos_expand = coords_pos.unsqueeze(2)  # (B, n_pos, 1, 3)
        coords_neg_expand = coords_neg.unsqueeze(1)  # (B, 1, n_neg, 3)
        dist = torch.norm(coords_pos_expand - coords_neg_expand, dim=-1)  # (B, n_pos, n_neg)
        threshold = 1e-3
        too_close = (dist < threshold).any(dim=1)  # (B, n_neg) — 
        valid_neg_mask = ~too_close  # (B, n_neg)
        


        large_value = 1e6
        rand_vals = torch.rand((B, self.n_neg), device=device)  # (B, n_neg)
        rand_vals = rand_vals.masked_fill(~valid_neg_mask, large_value)  


        sorted_rand_vals, sorted_idx = torch.sort(rand_vals, dim=1)  # (B, n_neg）
        sampled_idx = sorted_idx[:, :self.n_pos]  # (B, n_pos)

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, self.n_pos)  # (B, n_pos)
        coords_neg = coords_neg[batch_idx, sampled_idx]  # (B, n_pos, 3)
        '''

        mean_dist= dist.mean(dim=1)            # (B, n_neg)

        mean_dist = mean_dist.masked_fill(~valid_neg_mask, 0.0)  # (B, n_neg)

        eps = 1e-6
        weights =1/(mean_dist + eps)                # (B, n_neg)


        probs = weights / weights.sum(dim=1, keepdim=True)  # (B, n_neg)

        sampled_idx = torch.multinomial(probs, self.n_pos, replacement=False)  # (B, n_pos)
        
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.n_pos)  # (B, n_pos)
        coords_neg = coords_neg[batch_idx, sampled_idx]  # (B, n_pos, 3)
        '''



        if self.separate_decoders:
            cnn_out = self.cnn(img, obj_idx)  # (B, 1 + emb_dim, H, W)
            mask_lgts = cnn_out[:, 0]  # (B, H, W)
            queries = cnn_out[:, 1:]  # (B, emb_dim, H, W)
        else:
            cnn_out = self.cnn(img)  # (B, n_objs + n_objs * emb_dim, H, W)
            mask_lgts = cnn_out[torch.arange(B), obj_idx]  # (B, H, W)
            queries = cnn_out[:, self.n_objs:].view(B, self.n_objs, self.emb_dim, H, W)
            queries = queries[torch.arange(B), obj_idx]  # (B, emb_dim, H, W)

        mask_prob = torch.sigmoid(mask_lgts)  # (B, H, W)
        mask_loss = F.binary_cross_entropy(mask_prob, mask.type_as(mask_prob))

        queries = queries[torch.arange(B).view(B, 1), :, y, x]  # (B, n_pos, emb_dim)


        coords_pos += torch.randn_like(coords_pos) * self.key_noise
        keys_pos = torch.stack([self.mlps[i](c) for i, c in zip(obj_idx, coords_pos)])  # (B, n_pos, emb_dim)
        sim_pos = (queries * keys_pos).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)

        # compute similarities for negative pairs
        coords_neg += torch.randn_like(coords_neg) * self.key_noise
        keys_neg = torch.stack([self.mlps[i](v) for i, v in zip(obj_idx, coords_neg)])  # (B, n_neg, n_dim)
        sim_neg = queries @ keys_neg.permute(0, 2, 1)  # (B, n_pos, n_neg)
        coord_dist = torch.cdist(coords_pos, coords_neg, p=2)
        alpha = self.alpha
        #weights = torch.exp(-coord_dist.pow(2) / (2 * alpha**2))  # (B, n_pos, n_neg)
        #sim_neg = sim_neg * weights
    
        penalty = torch.log1p(coord_dist)
        sim_neg = sim_neg - alpha * penalty  
        
        lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
        target = torch.zeros(B, self.n_pos, device=device, dtype=torch.long)
        nce_loss = F.cross_entropy(lgts, target)
        consistency_loss = weighted_distance_consistency_loss(coords_pos, keys_pos)
        lambda_consistency = self.lambda_consistency
        loss = mask_loss +  lambda_consistency * consistency_loss+ nce_loss
        #self.log(f'{log_prefix}/sim_neg_mean', sim_neg.mean())
        self.log(f'{log_prefix}/penalty_mean', penalty.mean())

        self.log(f'{log_prefix}/loss', loss)
        self.log(f"{log_prefix}/consistency_loss", consistency_loss)

        self.log(f'{log_prefix}/mask_loss', mask_loss)
        self.log(f'{log_prefix}/nce_loss', nce_loss)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        self.log_image_sample(batch)
        return self.step(batch, 'valid')



    def get_emb_vis(self, emb_img: torch.Tensor, mask: torch.Tensor = None, demean: torch.tensor = False):
        if demean is True:
            demean = emb_img[mask].view(-1, self.emb_dim).mean(dim=0)
        if demean is not False:
            emb_img = emb_img - demean
        shape = emb_img.shape[:-1]
        emb_img = emb_img.view(*shape, 3, -1).mean(dim=-1)
        if mask is not None:
            emb_img[~mask] = 0.
        emb_img /= torch.abs(emb_img).max() + 1e-9
        emb_img.mul_(0.5).add_(0.5)
        return emb_img

    def log_image_sample(self, batch, i=0):
        img = batch['rgb_crop'][i]
        obj_idx = batch['obj_idx'][i]
        coord_img = batch['obj_coord'][i]
        coord_mask = coord_img[..., 3] != 0

        mask_lgts, query_img = self.infer_cnn(img, obj_idx)
        query_img = self.get_emb_vis(query_img)
        mask_est = torch.tile(torch.sigmoid(mask_lgts)[..., None], (1, 1, 3))

        key_img = self.infer_mlp(coord_img[..., :3], obj_idx)
        key_img = self.get_emb_vis(key_img, mask=coord_mask, demean=True)

        log_img = torch.cat((
            denormalize(img).permute(1, 2, 0), mask_est, query_img, key_img,
        ), dim=1).cpu().numpy()
        self.trainer.logger.experiment.log(dict(
            embeddings=wandb.Image(log_img),
            global_step=self.trainer.global_step
        ))



    @torch.no_grad()
    def infer_cnn(self, img: Union[np.ndarray, torch.Tensor], obj_idx, rotation_ensemble=True):
        assert not self.training
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = data.tfms.normalize(img)
            img = torch.from_numpy(img).to(self.device)
        _, h, w = img.shape

        if rotation_ensemble:
            img = utils.rotate_batch(img)  # (4, 3, h, h)
        else:
            img = img[None]  # (1, 3, h, w)
        cnn_out = self.cnn(img, [obj_idx] * len(img) if self.separate_decoders else None)
        if not self.separate_decoders:
            channel_idxs = [obj_idx] + list(self.n_objs + obj_idx * self.emb_dim + np.arange(self.emb_dim))
            cnn_out = cnn_out[:, channel_idxs]
        # cnn_out: (B, 1+emb_dim, h, w)
        if rotation_ensemble:
            cnn_out = utils.rotate_batch_back(cnn_out).mean(dim=0)
        else:
            cnn_out = cnn_out[0]
        mask_lgts, query_img = cnn_out[0], cnn_out[1:]
        query_img = query_img.permute(1, 2, 0)  # (h, w, emb_dim)
        return mask_lgts, query_img

    @torch.no_grad()
    def infer_mlp(self, pts_norm: Union[np.ndarray, torch.Tensor], obj_idx):
        assert not self.training
        if isinstance(pts_norm, np.ndarray):
            pts_norm = torch.from_numpy(pts_norm).to(self.device).float()
        return self.mlps[obj_idx](pts_norm)  # (..., emb_dim)
    @torch.no_grad()

    def on_validation_epoch_end(self):
        """
        After validation, run external inference and log ADD-10 accuracy.
        """
        # ensure we know device
        if (self.current_epoch+1)%10==0:
            device_name = str(self.device)
            # checkpoint path matches training naming: dataset-runid.ckpt
            ckpt_file = f"{self.dataset}-{self.run_id}.ckpt"
            ckpt_path = Path('data/models') / ckpt_file
            cmd = [
                "python", "-m", "surfsurg6d.scripts.infer_new",
                str(ckpt_path),
                "--device", device_name
            ]
            print(f"[Model] Running inference: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # load metrics
            metrics_fp = Path('data/results') / f"{self.dataset}-{self.run_id}-metrics.json"
            if not metrics_fp.exists():
                print(f"[Model] Metrics file not found: {metrics_fp}")
                return
            with open(metrics_fp, 'r') as mf:
                metrics = json.load(mf)

            add10 = metrics.get('add_10_accuracy', None)
            if add10 is None:
                print("[Model] add_10_accuracy not in metrics")
                return

            # log into Lightning so checkpoint callback can see it
            self.log('valid/add_10_accuracy', add10, prog_bar=True, on_epoch=True)

            # also push to WandB image
            wandb.log({'add_10_accuracy': add10, 'epoch': self.current_epoch})
