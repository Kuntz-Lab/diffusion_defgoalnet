import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

# from utils.dataset import *
from utils.misc import *
# from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from utils.misc import pcd_ize
import open3d

# def normalize_point_clouds(pcs, mode, logger):
#     if mode is None:
#         logger.info('Will not normalize point clouds.')
#         return pcs
#     logger.info('Normalization mode: %s' % mode)
#     for i in tqdm(range(pcs.size(0)), desc='Normalize'):
#         pc = pcs[i]
#         if mode == 'shape_unit':
#             shift = pc.mean(dim=0).reshape(1, 3)
#             scale = pc.flatten().std().reshape(1, 1)
#         elif mode == 'shape_bbox':
#             pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
#             pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
#             shift = ((pc_min + pc_max) / 2).view(1, 3)
#             scale = (pc_max - pc_min).max().reshape(1, 1) / 2
#         pc = (pc - shift) / scale
#         pcs[i] = pc
#     return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/null.pt')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=128)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()


# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(args.seed)


# Model
print('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])


# Generate Point Clouds
gen_pcs = []
for i in tqdm(range(0, 20), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
        gen_pcs.append(x.detach().cpu())

        # pcd_ize(x[0].detach().cpu().numpy(), color=[0,0,1], vis=True)
        pcd = pcd_ize(x[0].detach().cpu().numpy(), color=[0,0,1])
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        open3d.visualization.draw_geometries([pcd, coor])


