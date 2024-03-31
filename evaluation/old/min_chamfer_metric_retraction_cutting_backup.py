import os
import argparse
import torch
from tqdm.auto import tqdm
from copy import deepcopy
import sys
sys.path.append("../../")

from utils.misc import *
from models.vae_flow_3 import *
from evaluation import chamfer_dist_normalized_torch, chamfer_dist_normalized_numpy, chamfer_dist_normalized_open3d, ChamferLoss
from utils.misc import pcd_ize, read_pickle_data, spherify_point_cloud_open3d
import open3d


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='../../pretrained/test.pt')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=1)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=512)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

### Load data
dataset_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/processed_data/retraction_cutting"


# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(args.seed)


# Model
print('Loading model...')
model = BaoFlowVAE(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])
model.eval()


# Specific parameters
total_eval_samples = 100
vis = False


# Generate Point Clouds
min_chamfer_dists = []

with torch.no_grad():
    for idx in range(total_eval_samples):
        print_color(f"\n======= Sample {idx} ...")
    
        data = read_pickle_data(os.path.join(dataset_path, f"processed_sample_{idx}.pickle"))
        goal_pcs = data["partial_goal_pcs"] # shape (2, N, 3)
        init_pc = data["partial_init_pc"]  # shape (N, 3)
        context_pc = data["context"] # shape (M, 3)

        shift = init_pc.mean(axis=0)
        scale = (init_pc - shift).flatten().std()    
   
        
        init_pc_tensor = torch.from_numpy((init_pc - shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)
        context_pc_tensor = torch.from_numpy((context_pc - shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)

        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        predicted_goal_pc_tensor = model.sample(z, context_pc_tensor, init_pc_tensor, args.sample_num_points, flexibility=ckpt['args'].flexibility)
        # predicted_goal_pc_tensor = predicted_goal_pc_tensor * scale + shift
       
        predicted_goal_pc = predicted_goal_pc_tensor[0].detach().numpy() * scale + shift if args.device=="cpu" \
                            else predicted_goal_pc_tensor[0].detach().cpu().numpy() * scale + shift  
        min_chamfer_dist = min(chamfer_dist_normalized_numpy(predicted_goal_pc, goal_pcs[0]), 
                               chamfer_dist_normalized_numpy(predicted_goal_pc, goal_pcs[1]))
        # print(min_chamfer_dist)
        print(f"Min chamfer distance: {np.sqrt(min_chamfer_dist[0]):.4f}")
        min_chamfer_dists.append(min_chamfer_dist[0])

        min_chamfer_dist_open3d = min(chamfer_dist_normalized_open3d(predicted_goal_pc, goal_pcs[0]), 
                               chamfer_dist_normalized_open3d(predicted_goal_pc, goal_pcs[1]))
        print(f"open3d comparison: {min_chamfer_dist_open3d[0]:.4f}")
        
        ### Test DefGoalNet chamfer distance
        chamf = ChamferLoss()
        test_tensor = torch.from_numpy(predicted_goal_pc).unsqueeze(0).float().to(args.device)
        min_chamfer_dist_defgoalnet = min(chamf(test_tensor, 
                                                torch.from_numpy(goal_pcs[0]).unsqueeze(0).float().to(args.device)),
                                        chamf(test_tensor, 
                                              torch.from_numpy(goal_pcs[1]).unsqueeze(0).float().to(args.device)))
        min_chamfer_dist_defgoalnet = min_chamfer_dist_defgoalnet.detach().numpy()if args.device=="cpu" \
                            else min_chamfer_dist_defgoalnet.detach().cpu().numpy()
        print(f"defgoalnet comparison: {np.sqrt(min_chamfer_dist_defgoalnet/512):.4f}")

        if vis:
            pcd_init = pcd_ize(init_pc, color=[0,0,0])
            pcd_context = spherify_point_cloud_open3d(context_pc, color=[0, 1, 0], radius=0.004)
            
            pcd_goals_gt = []
            translation = (0.3, 0, 0)
            for i in range(goal_pcs.shape[0]):
                pcd_goals_gt.append(pcd_ize(goal_pcs[i], color=[0,0,1]).translate((translation)))
            
            pcd_goal_predicted = pcd_ize(predicted_goal_pc, color=[1,0,0])   # *scale + shift
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)

            open3d.visualization.draw_geometries([pcd_goal_predicted, pcd_init, pcd_context] + pcd_goals_gt) 





