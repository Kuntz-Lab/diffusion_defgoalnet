import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm
from copy import deepcopy

# from utils.dataset import *
from utils.misc import *
# from utils.data import *
from models.vae_gaussian import *
from models.vae_flow_3 import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from utils.misc import pcd_ize, read_pickle_data, spherify_point_cloud_open3d
import open3d
# from dataset_loader import DefGoalNetDataset
# from utils.data import *
# from torch.utils.data import DataLoader

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/null.pt')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=1)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

### Load data
# dataset_path = "/home/baothach/shape_servo_data/tanner/processed_data_slow"
# dataset_path = "/home/baothach/shape_servo_data/tanner/processed_data_tuned"
dataset_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/processed_data/retraction_cutting"


# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(args.seed)


# Model
print('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = BaoFlowVAE(ckpt['args']).to(args.device)
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])
model.eval()



# Generate Point Clouds
gen_pcs = []
for i in tqdm(range(0, 20), 'Generate'):
    with torch.no_grad():
        
        

        idx = 0#np.random.randint(200,239)
        data = read_pickle_data(os.path.join(dataset_path, f"processed_sample_{idx}.pickle"))
        goal_pcs = data["partial_goal_pcs"] # shape (2, N, 3)
        init_pc = data["partial_init_pc"]  # shape (N, 3)
        context_pc = data["context"] # shape (M, 3)

        shift = init_pc.mean(axis=0)
        scale = (init_pc - shift).flatten().std()       
        
    

        init_pc_tensor = torch.from_numpy((init_pc - shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)
        context_pc_tensor = torch.from_numpy((context_pc - shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)
        
        pcd_init = pcd_ize(init_pc, color=[0,0,0])
        # pcd_context = pcd_ize(context_pc, color=[0,0,1])
        pcd_context = spherify_point_cloud_open3d(context_pc, color=[0, 1, 0], radius=0.004)
        # pcd_goal_gt_1 = pcd_ize(goal_pcs[0], color=[0,1,0])
        # pcd_goal_gt_2 = pcd_ize(goal_pcs[1], color=[0,1,0])

        pcd_predicteds = []
        for i in range(20):
            print(f"Sample {i}")
            z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
            x = model.sample(z, context_pc_tensor, init_pc_tensor, args.sample_num_points, flexibility=ckpt['args'].flexibility)
            pcd_predicted = pcd_ize(x[0].detach().cpu().numpy()*scale + shift, color=[1,0,0])   # *scale + shift
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            # open3d.visualization.draw_geometries([pcd])
            open3d.visualization.draw_geometries([pcd_predicted, pcd_init, pcd_context]) # , pcd_goal_gt
            # pcd_predicteds.append(pcd_predicted)
            # if i == 2:
            #     break

        # translation = (0.3, 0, 0)
        # open3d.visualization.draw_geometries([pcd, pcd_init, pcd_context,
        #                                       pcd_gt.translate((translation)), deepcopy(pcd_init).translate((translation)), deepcopy(pcd_context).translate((translation))])
        
        # pcd_predicteds[2].paint_uniform_color([0, 0, 1]) 
        # open3d.visualization.draw_geometries([pcd_init, pcd_context, pcd_predicteds[0], pcd_predicteds[2]])



