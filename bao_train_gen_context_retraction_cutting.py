import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
# from models.vae_flow_2 import *
from models.vae_flow_3 import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
import timeit
from dataset_loader import RetractionCuttingDataset


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=512)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = os.path.join(args.log_root, "retraction_cutting")
    os.makedirs(log_dir, exist_ok=True)
    # log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')

# train_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='train',
#     scale_mode=args.scale_mode,
# )
# val_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='val',
#     scale_mode=args.scale_mode,
# )


dataset_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet/processed_data/retraction_cutting"

dataset = RetractionCuttingDataset(dataset_path)
dataset_size = 200  #4000 

train_len = round(dataset_size*0.95)
val_len = dataset_size - train_len
total_len = train_len + val_len

train_dset = torch.utils.data.Subset(dataset, range(0, train_len))
val_dset = torch.utils.data.Subset(dataset, range(train_len, total_len))

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))

print("***********************")
print("train_dset:", len(train_dset))
print("val_dset:", len(val_dset))

# Model
logger.info('Building model...')
if args.model == 'gaussian':
    model = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model = BaoFlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['goal_pcs'].to(args.device)
    x = x.view(-1, x.shape[-2], x.shape[-1])


    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight    

    context_pc = batch['context_pc'].to(args.device)
    init_pc = batch['init_pc'].to(args.device)
    context_pc = context_pc.view(-1, context_pc.shape[-2], context_pc.shape[-1])
    init_pc = init_pc.view(-1, init_pc.shape[-2], init_pc.shape[-1])

    # print("x.shape, context_pc.shape, init_pc.shape:", x.shape, context_pc.shape, init_pc.shape)
    loss = model.get_loss(x, context_pc, init_pc, kl_weight=kl_weight, writer=writer, it=it)
    


    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    # print(it)
    if it % 100 == 0:
        # logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        #     it, loss.item(), orig_grad_norm, kl_weight
        # ))
        # logger.info(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")

        print_color(f"================ Epoch {it}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        print('Loss %.6f | Grad %.4f | KLWeight %.4f' % (
             loss.item(), orig_grad_norm, kl_weight
        ))        
        print("\n")

    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()



# Main loop
start_time = timeit.default_timer()
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            # validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if it % 10000 == 0:
                ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
                
        # if it % args.test_freq == 0 or it == args.max_iters:
        #     test(it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
