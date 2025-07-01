"""Processors for training and evaluation"""


import datetime
import os  
from argparse import ArgumentParser  
import wandb  
from pytorch_lightning import Trainer  
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  
from pytorch_lightning.callbacks.progress import TQDMProgressBar  
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning.utilities import rank_zero_info  

from pl_gst_model import GSTModel
from pl_gcn_model import GcnModel

# Command line argument parser
def arg_parser():
  parser = ArgumentParser(description='Training PyTorch Lightning diffusion model on GST dataset.')
  
  # required parameter
  parser.add_argument('--task', type=str, default='gst')
  parser.add_argument('--model', type=str, default=None)
  parser.add_argument('--storage_path', type=str,default=None)
  # /home/sunyahui/DIFUSCO/data/result/dblp-g3-epsilon1-delta0.05/dblp-g3-cosine.txt
  parser.add_argument('--cost_path', type=str,default=None)

  parser.add_argument('--training_split', type=str, default=None)
  parser.add_argument('--validation_split', type=str, default=None)
  parser.add_argument('--test_split', type=str, default=None)
  parser.add_argument('--validation_examples', type=int, default=64) 

  # training parameter
  parser.add_argument('--batch_size', type=int, default=3)  
  parser.add_argument('--num_epochs', type=int, default=6)  
  parser.add_argument('--learning_rate', type=float, default=1e-5)  
  parser.add_argument('--weight_decay', type=float, default=0.0)  
  parser.add_argument('--lr_scheduler', type=str, default='constant')  

  # other parameters
  parser.add_argument('--num_workers', type=int, default=16) 
  parser.add_argument('--fp16', action='store_true')  
  parser.add_argument('--use_activation_checkpoint', action='store_true')  

  # parameters related to diffusion modeling
  parser.add_argument('--diffusion_type', type=str, default=None)  
  parser.add_argument('--diffusion_schedule', type=str, default='cosine')  
  parser.add_argument('--diffusion_steps', type=int, default=1000)  
  parser.add_argument('--inference_diffusion_steps', type=int, default=10)  
  parser.add_argument('--inference_schedule', type=str, default='cosine') 
  parser.add_argument('--inference_trick', type=str, default="ddim")  
  parser.add_argument('--sequential_sampling', type=int, default=8)  

  # diffusion model backbone network model parameters
  parser.add_argument('--n_layers', type=int, default=8)  
  parser.add_argument('--hidden_dim', type=int, default=256)  
  parser.add_argument('--sparse', type=int, default=True)  
  parser.add_argument('--aggregation', type=str, default='sum')  

  # GCN model parameters
  parser.add_argument('--node_dim_GCN', type=int, default=1)  
  parser.add_argument('--n_layers_GCN', type=int, default=8)  
  parser.add_argument('--hidden_dim_GCN', type=int, default=256)  
  parser.add_argument('--mlp_layers_GCN', type=int, default=1)
  parser.add_argument('--loss_type_GCN', type=str, default="FL")
  parser.add_argument('--aggregation_GCN', type=str, default='mean')  
  parser.add_argument('--learning_rate_GCN', type=float, default=1e-5)  

  # log and model retention
  parser.add_argument('--able_wandb', action='store_true')  
  parser.add_argument('--project_name', type=str, default='gst_diffusion')  
  parser.add_argument('--wandb_entity', type=str, default=None)  
  parser.add_argument('--wandb_logger_name', type=str, default=None)  
  parser.add_argument("--resume_id", type=str, default=None)  
  parser.add_argument('--ckpt_path', type=str, default=None)  
  parser.add_argument('--resume_weight_only', action='store_true')  

  # training and testing
  parser.add_argument('--do_train', action='store_true') 
  parser.add_argument('--do_test', action='store_true')  

  args = parser.parse_args()  
  return args  


def main(args):
  epochs = args.num_epochs  
  project_name = args.project_name  
 
  if args.task == 'gst':
    if args.model == 'diff':
      model_class = GSTModel
    elif args.model == 'gcn':
      model_class = GcnModel
    else:
      raise NotImplementedError
    saving_mode = 'min'  
  else:
    raise NotImplementedError  

  model = model_class(param_args=args)
  if args.able_wandb:
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()

    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,  
        project=project_name,  
        entity=args.wandb_entity,  
        save_dir=os.path.join(args.storage_path, f'model'),  
        id=args.resume_id or wandb_id,  
    )
    
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")
    checkpoint_dir = os.path.join(wandb_logger.save_dir, args.wandb_logger_name, wandb_logger._id, 'checkpoints')
  else:
    wandb_logger = None
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_dir = os.path.join(args.storage_path, 'checkpoints', run_id)
  
  checkpoint_callback = ModelCheckpoint(
      monitor='val_solved_cost_mean',
      filename='GST-{epoch:02d}-{val_solved_cost_mean:.4f}',
      mode=saving_mode,  
      save_top_k=3, save_last=True,  
      dirpath=checkpoint_dir,
      save_on_train_epoch_end=False  
  )
  lr_callback = LearningRateMonitor(logging_interval='step')  

  trainer = Trainer(
      accelerator="auto", 
      max_epochs=epochs,  
      callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, lr_callback],  
      logger=wandb_logger if args.able_wandb else None,  
      check_val_every_n_epoch=1,  
      strategy="ddp",  
      # gpus=[1, 2, 3],
      gpus=[0, 3],
      precision=32,  
  )

  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path  

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)  
      trainer.fit(model)  
    else:
      trainer.fit(model, ckpt_path=ckpt_path)  
  elif args.do_test:
    model = model_class.load_from_checkpoint(ckpt_path, param_args=args)  
    trainer.test(model,ckpt_path=ckpt_path)  
  if args.able_wandb:  
    trainer.logger.finalize("success")  

# 程序入口
if __name__ == '__main__':
  args = arg_parser()  
  main(args) 