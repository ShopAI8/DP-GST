"""用于训练和评估的处理器"""

# 引入必要的库和模块
import datetime
import os  # 操作系统相关的函数
from argparse import ArgumentParser  # 用于处理命令行参数
import time

import torch  # PyTorch 深度学习框架
import wandb  # 用于实验跟踪和模型可视化的工具
from pytorch_lightning import Trainer  # PyTorch Lightning 训练器，简化训练代码
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # 回调函数：监控学习率、保存模型
from pytorch_lightning.callbacks.progress import TQDMProgressBar  # 进度条显示工具
from pytorch_lightning.loggers import WandbLogger  # 日志记录器，用于与 Weights and Biases（wandb）集成
from pytorch_lightning.strategies.ddp import DDPStrategy  # 分布式数据并行策略，用于多GPU训练
from pytorch_lightning.utilities import rank_zero_info  # 用于在rank为0（主进程）时打印信息

#from pl_tsp_model import TSPModel  # 导入旅行商问题(TSP)模型
#from pl_mis_model import MISModel  # 导入最大独立集问题(MIS)模型
from pl_gst_model import GSTModel
from pl_gcn_model import GcnModel
gamma = 5#读取的数据中拓补结构数 musae g5训练
name = 'musae'
model_class = 'diff' #diff，gcn
diffusion_type = 'categorical' #gaussian,categorical
sparse = 1 
schedule = 'cosine'  #cosine,linear
epsilon = 1 #0.5,1
delta = 0.05 #1e-05,0.05
T=5
sample = 1
# 命令行参数解析器
def arg_parser():
  # 创建解析器，并为各个参数添加说明
  parser = ArgumentParser(description='在GST数据集上训练PyTorch Lightning扩散模型。')
  
  # 必要参数
  parser.add_argument('--task', type=str, default='gst')  # 任务类型（tsp 或 mis）
  parser.add_argument('--model', type=str, default=model_class)  
  parser.add_argument('--storage_path', type=str,default='/home/sunyahui/DIFUSCO/difusco/')  # 存储路径
  parser.add_argument('--cost_path', type=str,default=f'/home/sunyahui/DIFUSCO/data/result/{name}-g{gamma}-{diffusion_type}-epsilon{epsilon}-delta{delta}/{name}-g{gamma}-{diffusion_type}-{schedule}-epsilon{epsilon}-delta{delta}-step{T}-sample{sample}.txt')
  # parser.add_argument('--cost_path', type=str,default=f'/home/sunyahui/DIFUSCO/data/result/{name}-g{gamma}-{model_class}-epsilon{epsilon}-delta{delta}/{name}-g{gamma}-{model_class}-epsilon{epsilon}-delta{delta}.txt')

  # 数据集参数
  parser.add_argument('--training_split', type=str, default=f'/home/sunyahui/DIFUSCO/data/tasks_30-num_2000-epsilon_{epsilon}-delta_{delta}/{name}_GST1k-gamma_{gamma}-noise/train_split.txt')  # 训练集文件路径
  parser.add_argument('--training_split_label_dir', type=str, default=None, help="Directory containing labels for training split (used for MIS).")  # MIS任务的标签路径
  parser.add_argument('--validation_split', type=str, default=f'/home/sunyahui/DIFUSCO/data/tasks_30-num_2000-epsilon_{epsilon}-delta_{delta}/{name}_GST1k-gamma_{gamma}-noise/valid_split.txt')  # 验证集文件路径
  parser.add_argument('--test_split', type=str, default=f'/home/sunyahui/DIFUSCO/data/tasks_30-num_2000-epsilon_{epsilon}-delta_{delta}/{name}_GST1k-gamma_{gamma}-noise/test_split.txt')  # 测试集文件路径
  parser.add_argument('--validation_examples', type=int, default=64)  # 验证集示例数量

  # 训练参数
  parser.add_argument('--batch_size', type=int, default=3)  # 批次大小
  parser.add_argument('--num_epochs', type=int, default=9)  # 训练轮次
  parser.add_argument('--learning_rate', type=float, default=1e-5)  # 学习率
  parser.add_argument('--weight_decay', type=float, default=0.0)  # 权重衰减
  parser.add_argument('--lr_scheduler', type=str, default='constant')  # 学习率调度器类型

  # 其他参数
  parser.add_argument('--num_workers', type=int, default=16)  # 数据加载的工作线程数
  parser.add_argument('--fp16', action='store_true')  # 是否使用半精度浮点数训练
  parser.add_argument('--use_activation_checkpoint', action='store_true')  # 是否使用激活检查点

  # 扩散模型相关参数
  parser.add_argument('--diffusion_type', type=str, default=f'{diffusion_type}')  # 扩散模型类型
  parser.add_argument('--diffusion_schedule', type=str, default=f'{schedule}')  # 扩散过程的时间表
  parser.add_argument('--diffusion_steps', type=int, default=1000)  # 扩散步骤数
  parser.add_argument('--inference_diffusion_steps', type=int, default=T)  # 推理时的扩散步骤数
  parser.add_argument('--inference_schedule', type=str, default=f'{schedule}')  # 推理时的时间表
  parser.add_argument('--inference_trick', type=str, default="ddim")  # 推理时使用的技巧
  parser.add_argument('--sequential_sampling', type=int, default=sample)  # 顺序采样
  parser.add_argument('--parallel_sampling', type=int, default=1)  # 并行采样

  # 扩散模型主干网络模型参数
  parser.add_argument('--n_layers', type=int, default=8)  # 网络层数
  parser.add_argument('--hidden_dim', type=int, default=256)  # 隐藏层维度
  parser.add_argument('--sparse', type=int, default=sparse)  # 稀疏因子
  parser.add_argument('--aggregation', type=str, default='sum')  # 聚合方式
  parser.add_argument('--eps', type=float, default=0.01)
  parser.add_argument('--gamma', type=float, default=0.1)
  parser.add_argument('--add_noisy', type=int, default=0)
  # parser.add_argument('--two_opt_iterations', type=int, default=1000)  # Two-opt 算法迭代次数
  # parser.add_argument('--save_numpy_heatmap', action='store_true')  # 是否保存numpy热力图

  # GCN模型参数
  parser.add_argument('--node_dim_GCN', type=int, default=1)  
  parser.add_argument('--n_layers_GCN', type=int, default=9)  # 网络层数
  parser.add_argument('--hidden_dim_GCN', type=int, default=256)  # 隐藏层维度
  parser.add_argument('--mlp_layers_GCN', type=int, default=2)
  parser.add_argument('--loss_type_GCN', type=str, default="FL")
  parser.add_argument('--aggregation_GCN', type=str, default='mean')  # 聚合方式
  parser.add_argument('--learning_rate_GCN', type=float, default=1e-5)  # 学习率

  # 日志和模型保存
  parser.add_argument('--able_wandb', action='store_true')  # 是否使用wandb
  parser.add_argument('--project_name', type=str, default='gst_diffusion')  # 项目名称
  parser.add_argument('--wandb_entity', type=str, default=None)  # wandb的实体名称
  parser.add_argument('--wandb_logger_name', type=str, default=f'{diffusion_type}-{schedule}-{name}-g{gamma}-epsilon{epsilon}-delta{delta}-task30')  # wandb日志名称
  # parser.add_argument('--wandb_logger_name', type=str, default=f'{model_class}-{name}-g{gamma}-epsilon{epsilon}-delta{delta}-task30')  # wandb日志名称
  parser.add_argument("--resume_id", type=str, default=None, help="在wandb上恢复训练的ID。")  # 恢复训练的ID
  parser.add_argument('--ckpt_path', type=str, default=None)  # 模型检查点路径
  parser.add_argument('--resume_weight_only', action='store_true')  # 仅恢复权重

  # 训练和测试控制
  parser.add_argument('--do_train', action='store_true')  # 执行训练
  parser.add_argument('--do_test', action='store_true')  # 执行测试
  parser.add_argument('--do_valid_only', action='store_true')  # 仅进行验证

  args = parser.parse_args()  # 解析命令行参数
  return args  # 返回解析后的参数

# 主函数
def main(args):
  epochs = args.num_epochs  # 训练轮次
  project_name = args.project_name  # 项目名称
 
  # 根据任务类型选择模型
  if args.task == 'tsp':  # 如果任务是TSP
    # model_class = TSPModel  # 使用TSP模型
    saving_mode = 'min'  # 最小化目标（例如，旅行商问题目标是最小化路径）
  elif args.task == 'mis':  # 如果任务是MIS
    # model_class = MISModel  # 使用MIS模型
    saving_mode = 'max'  # 最大化目标（例如，最大独立集问题目标是最大化独立集）
  elif args.task == 'gst':
    if args.model == 'diff':
      model_class = GSTModel
    elif args.model == 'gcn':
      model_class = GcnModel
    else:
      raise NotImplementedError
    saving_mode = 'min'  
  else:
    raise NotImplementedError  # 如果任务类型不支持，则抛出未实现的异常

  # 实例化模型
  model = model_class(param_args=args)
  if args.able_wandb:
    # 从环境变量获取wandb运行ID，或生成一个新的ID
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()

    # 配置wandb日志
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,  # 日志名称
        project=project_name,  # 项目名称
        entity=args.wandb_entity,  # wandb实体
        save_dir=os.path.join(args.storage_path, f'models'),  # 模型保存路径
        id=args.resume_id or wandb_id,  # 恢复训练时的ID
    )
    # 记录日志信息
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")
    checkpoint_dir = os.path.join(wandb_logger.save_dir, args.wandb_logger_name, wandb_logger._id, 'checkpoints')
  else:
    wandb_logger = None
    # 生成独立运行ID（替代 WandB ID）
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用自定义路径（例如基于时间戳）
    checkpoint_dir = os.path.join(args.storage_path, 'checkpoints', run_id)
  # 设置模型检查点回调，保存最好的模型
  checkpoint_callback = ModelCheckpoint(
      monitor='val_solved_cost_mean',
      filename='GST-{epoch:02d}-{val_solved_cost_mean:.4f}',
      mode=saving_mode,  # 监控的指标和保存模式
      save_top_k=3, save_last=True,  # 保存最好的3个模型，保存最后一次训练模型
      dirpath=checkpoint_dir,# 检查点保存路径
      save_on_train_epoch_end=False  # 添加此参数
  )
  lr_callback = LearningRateMonitor(logging_interval='step')  # 监控学习率

  # 配置训练器
  trainer = Trainer(
      accelerator="auto",  # 根据硬件自动选择加速器
      # devices=torch.cuda.device_count() if torch.cuda.is_available() else None,  # 检测GPU设备数量
      # devices = 1,
      max_epochs=epochs,  # 最大训练轮次
      callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, lr_callback],  # 回调函数，包括进度条、检查点和学习率监控
      logger=wandb_logger if args.able_wandb else None,  # 使用wandb日志
      check_val_every_n_epoch=1,  # 每1个epoch进行验证
      # strategy=DDPStrategy(static_graph=False),  # 使用分布式数据并行策略
      strategy="ddp",  # 分布式策略（自动处理多卡同步）
      # gpus=[1, 2, 3],
      gpus=[0,2,3],
      precision=32,  # 使用16位精度（如果指定了fp16），否则使用32位精度
  )

  # 输出模型信息
  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path  # 检查点路径

  # 如果选择训练
  if args.do_train:
    # 如果仅恢复权重
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)  # 从检查点加载模型
      trainer.fit(model)  # 继续训练
    else:
      trainer.fit(model, ckpt_path=ckpt_path)  # 正常训练

    

  # 如果仅进行验证或测试
  elif args.do_test:
    
    model = model_class.load_from_checkpoint(ckpt_path, param_args=args)  # 从检查点加载模型
    trainer.test(model,ckpt_path=ckpt_path)  # 使用最好的模型进行测试
  if args.able_wandb:  
    trainer.logger.finalize("success")  # 结束训练并记录成功状态

# 程序入口
if __name__ == '__main__':
  os.environ["WANDB_API_KEY"] = '5710c42cf3f6f600cd091c4276c541476fe7b058'
  os.environ["WANDB_MODE"] = "offline"  # 离线
  # nohup python train_diff.py >> output.out 2>&1 &     ps aux | grep "[p]ython"来查看后台运行的进程  tail -f output.out来查看输出日志
  args = arg_parser()  # 解析命令行参数
  # args.able_wandb = True
  # args.do_train = True
  # args.resume_weight_only = True
  args.ckpt_path = "/home/sunyahui/DIFUSCO/difusco/models/categorical-cosine-musae-g5-epsilon1-delta0.05-task30/g3tjaubi/checkpoints/GST-epoch=01-val_solved_cost_mean=0.0000.ckpt"
  path = f'/home/sunyahui/DIFUSCO/data/result/{name}-g{gamma}-{diffusion_type}-epsilon{epsilon}-delta{delta}'#结果存在文件夹中
  # path = f'/home/sunyahui/DIFUSCO/data/result/{name}-g{gamma}-{model_class}-epsilon{epsilon}-delta{delta}'#结果存在文件夹中
  os.makedirs(path) if not os.path.exists(path) else print(f"目录{path}已存在")
  args.do_test = True
  args.able_wandb = False
  step_list = [1, 5, 10, 20, 50]
  sample_list = [32,16,8,4,1]
  for i in range(5):
    args.inference_diffusion_steps = step_list[i]
    args.sequential_sampling = sample_list[i]
    args.cost_path = f'/home/sunyahui/DIFUSCO/data/result/{name}-g{gamma}-{diffusion_type}-epsilon{epsilon}-delta{delta}/{name}-g{gamma}-{diffusion_type}-{schedule}-epsilon{epsilon}-delta{delta}-step{step_list[i]}-sample{sample_list[i]}.txt'
    main(args)  # 执行主函数
  # main(args)  # 执行主函数