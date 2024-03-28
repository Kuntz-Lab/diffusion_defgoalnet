import torch
import os
from torch.utils.data import Dataset
import pickle5 as pickle
import numpy as np
import random
      

class DefGoalNetDataset(Dataset):
    """predict mani point using segmentation"""


    def __init__(self, dataset_path):
        """
        Args:

        """ 
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        # random.shuffle(self.filenames)  
        
        # pointclouds = []
        # for filename in self.filenames[:1000]:
        #     sample = self.load_pickle_data(filename)
        #     pointclouds.append(sample["full pcs"][1].transpose(1,0))
            
        # all_points = np.concatenate(tuple(pointclouds), axis=0).reshape(-1, 3) # (B, 3, N)
        # mean = all_points.mean(axis=0)
        # all_points -= mean
        # self.global_std = all_points.flatten().std()
            
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):    
            
        sample = self.load_pickle_data(f"processed example {idx}.pickle")
        # sample = self.load_pickle_data(f"processed sample {idx}.pickle")
 
        # pc = torch.from_numpy(sample["partial pcs"][0]).float()  # shape (B, 3, num_pts)  
        # pc_goal = torch.from_numpy(sample["partial pcs"][1]).float()         
        # kidney_pc = torch.from_numpy(sample["kidney_pc"]).permute(1,0).float()
        # sample = {"pcs": (pc, pc_goal), "kidney_pc": kidney_pc}
        
        # pc = torch.from_numpy(sample["full pcs"][1]).permute(1,0).float()  # shape (num_pts, 3)
        # shift = pc.mean(dim=0)
        # pc_centered = pc - shift

        # scale = pc_centered.flatten().std()
        # pc = pc_centered / scale
        
        # kidney_angle = torch.tensor(sample["angles"][0]).unsqueeze(0).float()  # shape (1,)
        
        # return {'pointcloud': pc, 'kidney_angle': kidney_angle}
        
        pc = torch.from_numpy(sample["full pcs"][1]).permute(1,0).float()  # shape (num_pts, 3)
        init_pc = torch.from_numpy(sample["full pcs"][0]).permute(1,0).float()  # shape (num_pts, 3)
        context_pc = torch.from_numpy(sample["kidney_pc"]).float()  # shape (num_pts, 3)

        
        shift = init_pc.mean(dim=0)
        scale = (init_pc - shift).flatten().std()
        
        init_pc = (init_pc - shift) / scale
        pc = (pc - shift) / scale
        context_pc = (context_pc - shift) / scale

        
        return {'pointcloud': pc, 'init_pc': init_pc, 'context_pc': context_pc}
    
    
    
    



        
         

        
        return sample   



