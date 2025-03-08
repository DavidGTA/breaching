import os
import sys

os.chdir('/root/nas/ZhuZhu/breaching')
# print(os.getcwd())
# 将 breaching 文件夹的路径添加到 sys.path 中
sys.path.append(os.getcwd())
# python examples/Inverting\ Gradients\ -\ Optimization-based\ Attack\ -\ ResNet18\ on\ ImageNet\ -\ Federated\ Averaging.py
import torch
import logging
import json
from omegaconf import OmegaConf
from datetime import datetime
# 确保包存在，如果不存在则回退到父目录并尝试再次导入
try:
    import breaching
except ModuleNotFoundError:
    os.chdir("..")
    import breaching

# 设置日志文件路径和文件名
log_dir = "logs"  # 指定日志文件夹
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"Inverting Gradients - Optimization-based Attack - ResNet18 on ImageNet - Federated Averaging - {current_time}.log"  # 日志文件名
log_path = os.path.join(log_dir, log_file)

# 如果日志目录不存在，创建它
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 自定义日志输出格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler(log_path)  # 输出到日志文件
    ],
    encoding='utf-8'
)

logger = logging.getLogger()

# 示例日志
logger.info(f"This is Inverting Gradients - Optimization-based Attack - ResNet18 on ImageNet - Federated Averaging - {current_time} log.")



# 配置加载
cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])

# 设置设备
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
logger.info(f"Setup: {setup}")

# 配置参数设置
cfg.case.data.partition = "random"
cfg.case.user.user_idx = 1
cfg.case.model = 'resnet18'

cfg.case.user.provide_labels = True
cfg.case.user.num_data_points = 4
cfg.case.user.num_local_updates = 4
cfg.case.user.num_data_per_local_update_step = 2
cfg.attack.regularization.total_variation.scale = 1e-3

# 将 OmegaConf 配置转换为字典并输出为 JSON 格式
cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # 转换为 dict，并解析占位符
json_output = json.dumps(cfg_dict, indent=4, ensure_ascii=False)  # 格式化 JSON，使用 UTF-8 编码
logger.info(f"Config: {json_output}")

# 构建并准备案例
user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)

# 概览
logger.info(f"Overview: {breaching.utils.overview(server, user, attacker)}")

# 分配有效载荷并计算本地更新
server_payload = server.distribute_payload()

# server_payload有两个属性：parameters, buffers, metadata
# len(server_payload['parameters']) = 62
# 取其中一个：server_payload['parameters'][0].size() = torch.Size([64, 3, 3, 3])
# len(server_payload['buffers']) = 60
# 取其中一个：server_payload['buffers'][0].size() = torch.Size([64])
# server_payload['metadata'] = {'db': {'name': None}, 'name': 'CIFAR10', 'modality': 'vision', 'task': 'classification', 'path': '/root/nas/ZhuZhu/data', 'size': 50000, 'classes': 10, 'shape': [3, 32, 32], 'normalize': True, 'mean': [0.4914672374725342, 0.4822617471218109, 0.4467701315879822], 'std': [0.24703224003314972, 0.24348513782024384, 0.26158785820007324], 'augmentations_train': {'RandomCrop': [32, 4], 'RandomHorizontalFlip': 0.5}, 'augmentations_val': None, 'default_clients': 10, 'partition': 'random', 'examples_from_split': 'validation', 'batch_size': 128, 'caching': False}

shared_data, true_user_data = user.compute_local_updates(server_payload)

# shared_data有三个属性：gradients, buffers, metadata
# len(shared_data['gradients']) = 62
# 取其中一个：shared_data['gradients'][0].size() = torch.Size([64, 3, 3, 3])
# buffers为None
# shared_data['metadata'] = {'num_data_points': 4, 'labels': tensor([1, 1, 6, 3], device='cuda:0'), 'local_hyperparams': {'lr': 0.001, 'steps': 4, 'data_per_step': 2, 'labels': [tensor([1, 1], device='cuda:0'), tensor([3, 6], device='cuda:0'), tensor([1, 1], device='cuda:0'), tensor([3, 6], device='cuda:0')]}, 'data_key': 'inputs'}

# true_user_data有三个属性：data, labels, buffers
# true_user_data['data'].size() = torch.Size([4, 3, 32, 32])
# true_user_data['labels'].size() = torch.Size([4])
# len(true_user_data['buffers']) = 60
# 取其中一个：true_user_data['buffers'][0].size() = torch.Size([64])

# 绘制真实用户数据
user.plot(true_user_data)

# 攻击者重建数据并计算指标
reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# 打印攻击结果并绘制重建数据
metrics = breaching.analysis.report(
    reconstructed_user_data, true_user_data, [server_payload], server.model,
    order_batch=True, compute_full_iip=False, cfg_case=cfg.case, setup=setup
)
logger.info(f"Metrics: {metrics}")

user.plot(reconstructed_user_data)
