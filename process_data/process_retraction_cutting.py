import sys
sys.path.append("../")
from utils.misc import pcd_ize, read_pickle_data, write_pickle_data, down_sampling, print_color, spherify_point_cloud_open3d
import os
import open3d
import numpy as np
import timeit


raw_data_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/data/retraction_cutting"
processed_data_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/processed_data/retraction_cutting"
os.makedirs(processed_data_path, exist_ok=True)

categoies = ["cylinder", "ellipsoid"]    #["cylinder", "ellipsoid"]
num_object_per_category = 50     #50cc vxc v
processed_data_count = 0
start_time = timeit.default_timer()

num_pts = 512
vis = False

for category in categoies:
    for object_idx in range(0, num_object_per_category):
        for context_idx in range(0,100): 
            obj_name = f"{category}_{object_idx}"
            data_path = f"{raw_data_path}/{obj_name}_{context_idx}.pickle"
            if not os.path.exists(data_path):
                break            
            
            print("\n")
            print_color(f"Processing {obj_name}, data idx {context_idx} ...")
            print("Processed data count:", processed_data_count)
            print(f"Time elapsed (mins): {(timeit.default_timer() - start_time)/60:.2f}")


            data = read_pickle_data(data_path)
            goal_pcs_partial = data["partial_goal_pcs"]
            init_pc_partial = data["partial_init_pc"]
            goal_pcs_full = data["full_goal_pcs"]
            init_pc_full = data["full_init_pc"]
            context = data["context"]

            down_sampled_goal_pcs_partial = np.concatenate((down_sampling(goal_pcs_partial[0], num_pts)[None], 
                                                            down_sampling(goal_pcs_partial[1], num_pts)[None]), axis=0)
            down_sampled_goal_pcs_full = np.concatenate((down_sampling(goal_pcs_full[0], num_pts)[None], 
                                                            down_sampling(goal_pcs_full[1], num_pts)[None]), axis=0)
            
            down_sampled_init_pc_partial = down_sampling(init_pc_partial, num_pts)
            down_sampled_init_pc_full = down_sampling(init_pc_full, num_pts)
            down_sampled_context = down_sampling(context, num_pts)
            # print(down_sampled_goal_pcs_partial.shape, down_sampled_goal_pcs_full.shape, 
            #       down_sampled_init_pc_partial.shape, down_sampled_init_pc_full.shape, down_sampled_context.shape)

            if vis:
                pcd_goal_1 = pcd_ize(down_sampled_goal_pcs_partial[0], color=[0, 0, 1])
                pcd_goal_2 = pcd_ize(down_sampled_goal_pcs_partial[1], color=[1, 0, 0])
                pcd_init = pcd_ize(down_sampled_init_pc_partial, color=[0, 0, 0])
                # context_pcd = pcd_ize(data["context"], color=[0, 1, 0])
                context_pcd = spherify_point_cloud_open3d(down_sampled_context, color=[0, 1, 0])
                open3d.visualization.draw_geometries([pcd_goal_1, pcd_goal_2, pcd_init, context_pcd])
                # open3d.visualization.draw_geometries([pcd_goal_1, context_pcd])

            processed_data = {
                "partial_goal_pcs": down_sampled_goal_pcs_partial,
                "partial_init_pc": down_sampled_init_pc_partial,
                "full_goal_pcs": down_sampled_goal_pcs_full,
                "full_init_pc": down_sampled_init_pc_full,
                "context": down_sampled_context
            }
            write_pickle_data(processed_data, processed_data_path + f"/processed_sample_{processed_data_count}.pickle")
            processed_data_count += 1
