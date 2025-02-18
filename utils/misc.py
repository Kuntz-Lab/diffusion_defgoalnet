import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
import pickle5 as pickle
from copy import deepcopy
import open3d

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

def pcd_ize(pc, color=None, vis=False):
    """ 
    Convert point cloud numpy array to an open3d object (usually for visualization purpose).
    """
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)
        
def read_pickle_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)      


def write_pickle_data(data, data_path, protocol=3):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=protocol)

def down_sampling(pc, num_pts=1024, return_indices=False):
    # farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    # pc = pc[farthest_indices.squeeze()]  
    # return pc

    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled pointcloud index, [num_pts, D]
        pc: down_sampled point cloud, [num_pts, D]
    """

    if pc.ndim == 2:
        # insert batch_size axis
        pc = deepcopy(pc)[None, ...]

    B, N, D = pc.shape
    xyz = pc[:, :,:3]
    centroids = np.zeros((B, num_pts))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)

    pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]

    if return_indices:
        return pc.squeeze(), centroids.astype(np.int32)

    return pc.squeeze()

def down_sampling_torch(pc, num_pts=1024, return_indices=False):
    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled point cloud index, [num_pts, D]
        pc: down-sampled point cloud, [num_pts, D]
    """
    import torch
    if pc.ndim == 2:
        # Insert batch_size axis
        pc = pc.unsqueeze(0)

    B, N, D = pc.shape
    xyz = pc[:, :, :3]
    centroids = torch.zeros((B, num_pts), dtype=torch.long, device=pc.device)
    distance = torch.ones((B, N), dtype=pc.dtype, device=pc.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=pc.device)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(0, B), farthest, :]  # (B, D)
        centroid = centroid.unsqueeze(1)  # (B, 1, D)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)  # (B,)

    pc = pc[torch.arange(0, B).view(-1, 1), centroids, :]

    if return_indices:
        return pc.squeeze(), centroids

    return pc.squeeze()

def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)


def spherify_point_cloud_open3d(point_cloud, radius=0.002, color=None, vis=False):
    """
    Use Open3D to visualize a point cloud where each point is represented by a sphere.
    """
    """
    Visualize a point cloud where each point is represented by a sphere.
    
    Parameters:
    - point_cloud: NumPy array of shape (N, 3), representing the point cloud.
    - radius: float, the radius of each sphere used to represent a point.
    """
    # Create an empty list to hold the sphere meshes
    sphere_meshes = []
    
    # Iterate over the points in the point cloud
    for point in point_cloud:
        # Create a mesh sphere for the current point
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(tuple(point))  # Move the sphere to the point's location
        if color is not None:
            sphere.paint_uniform_color(color) 
        sphere_meshes.append(sphere)
    
    # Combine all spheres into one mesh
    combined_mesh = open3d.geometry.TriangleMesh()
    for sphere in sphere_meshes:
        combined_mesh += sphere
    if vis:
        open3d.visualization.draw_geometries([combined_mesh])
    return combined_mesh