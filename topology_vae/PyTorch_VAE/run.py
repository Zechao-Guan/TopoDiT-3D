import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, PIDataSet
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--save_dir', default="../../")
# PI dataset
parser.add_argument('--use_pi', default=True)
parser.add_argument('--root_dir', default="data/ShapeNetCore.v2.PC15k")
parser.add_argument('--categories', default='chair')
parser.add_argument('--tr_sample_size', type=int, default=2048)
parser.add_argument('--te_sample_size', type=int, default=2048)
parser.add_argument('--scale', type=float, default=1.)
parser.add_argument('--normalize_per_shape', default=False)
parser.add_argument('--normalize_std_per_axis', default=False)
parser.add_argument('--box_per_shape', default=False)
parser.add_argument('--random_subsample', default=True)
parser.add_argument('--all_points_mean', default=None)
parser.add_argument('--all_points_std', default=None)
parser.add_argument('--use_mask', default=False)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
if args.use_pi:
    config['logging_params']['save_dir'] = args.save_dir
    config['model_params']['in_channels'] = 2


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params']) #封装model, training_step, validation_step, on_validation_end(即sample), 以及opitimizer
if not args.use_pi:
    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
else:
    data = PIDataSet(root_dir = args.root_dir, 
                     categories=args.categories.split(','),
                     tr_sample_size=args.tr_sample_size,
                     te_sample_size=args.te_sample_size,
                     scale=args.scale,
                     normalize_per_shape=args.normalize_per_shape,
                     normalize_std_per_axis=args.normalize_std_per_axis,
                     box_per_shape=args.box_per_shape,
                     random_subsample=args.random_subsample,
                     all_points_mean=args.all_points_mean,
                     all_points_std=args.all_points_std,
                     use_mask=args.use_mask,
                     train_batch_size=args.train_batch_size,
                     val_batch_size=args.val_batch_size,
                     patch_size=args.patch_size,
                     num_workers=args.num_workers,
    )

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params']) #GPU，epoch


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)