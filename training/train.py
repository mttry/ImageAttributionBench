import os  
import argparse  
import datetime  
import yaml  
import torch  
from torch.backends import cudnn  
from trainer.trainer import Trainer  
from attributors import ATTRIBUTOR  
from metrics.utils import parse_metric_for_print  
from logger import create_logger  

from dataset.ImageAttributionDataset.dataloader import get_dataloader
from dataset.ImageAttributionDataset.semantic_split import get_semantic

parser = argparse.ArgumentParser()  
parser.add_argument('--config', type=str, default='config/model/resnet50.yaml',help='path to attributor YAML config file')  
parser.add_argument('--use_semantic_split', action='store_true', default=False, help='whether to use semantic dataloader')  
# parser.add_argument('--root_dir', type=str, default='/home/final_dataset', help='root directory for dataset')  
parser.add_argument('--root_dir', type=str, default='/remote-home/share/gzy/attribution/final_dataset_thats_real', help='root directory for dataset')  
parser.add_argument('--batch_size', type=int, default=32)  
parser.add_argument('--n_epoch', type=int)  
parser.add_argument('--num_workers', type=int, default=2)  
parser.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)  
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)  
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)  
parser.add_argument('--task_target', type=str, default="")  
parser.add_argument('--resume_checkpoint', type=str, default=None)  
parser.add_argument('--log_dir', type=str, default="./logs")  
parser.add_argument('--task_id', type=int, default=1, help='任务编号，用于选择不同的语义划分，例如1、2、3等' )
parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint saving frequency (epochs)')  
parser.add_argument('--do_test', action='store_false', default=True, help='Whether to run test evaluation after training')  
args = parser.parse_args()  


def init_seed(seed, use_cuda=True):  
    import random  
    import numpy as np  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if use_cuda:  
        torch.cuda.manual_seed_all(seed)  


def create_training_logger(config, use_semantic_split=False):  
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  
    task_str = f"_{config['task_target']}" if config.get('task_target') else ""  
    base_folder = config['log_dir']  

    # 根据use_semantic_split选择不同的日志子目录  
    sub_folder = f"semantic_split_{args.task_id}" if use_semantic_split else "default_split"  
    log_dir = os.path.join(base_folder, sub_folder, config['model_name'], task_str + '_' + now)  
    os.makedirs(log_dir, exist_ok=True)  
    logger = create_logger(os.path.join(log_dir, 'training.log'))  
    logger.info(f"Training logs saved to {log_dir}")  
    return logger, log_dir, now   


def choose_optimizer(model, config):  
    opt_name = config['optimizer']['type']  
    if opt_name == 'adam':  
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer'][opt_name]['lr'])  
    elif opt_name == 'sgd':  
        optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer'][opt_name]['lr'], momentum=0.9)  
    else:  
        raise NotImplementedError(f"Optimizer {opt_name} not supported")  
    return optimizer  


def choose_scheduler(config, optimizer):  
    if config.get('lr_scheduler', None) == 'step':  
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])  
    elif config.get('lr_scheduler', None) == 'cosine':  
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])  
    else:  
        return None  


def choose_metric(config):  
    metric = config.get('metric_scoring', 'auc')  
    if metric not in ['auc', 'acc', 'ap']:  
        raise NotImplementedError(f"Metric {metric} not implemented")  
    return metric  


def main():  
    # loading config  
    with open(args.config, 'r') as f:  
        config = yaml.safe_load(f)  

    config['save_ckpt'] = args.save_ckpt  
    config['save_feat'] = args.save_feat  
    # config['task_target'] = args.task_target  
    config['log_dir'] = args.log_dir  

    use_cuda = torch.cuda.is_available()  
    init_seed(config.get('manualSeed', 42), use_cuda)  

    if use_cuda and config.get('cudnn', True):  
        cudnn.benchmark = True  

    # loading logger  
    logger, log_dir, now = create_training_logger(config, use_semantic_split=args.use_semantic_split)  
    logger.info("Config:\n" + "\n".join(f"{k}: {v}" for k, v in config.items()))  

    # loading dataloader  
    model_name = config['model_name']  
    if args.use_semantic_split:  
        print("using semantic split...")  
        train_semantics, test_semantics = get_semantic(args.task_id)  
        print("training semantic:", train_semantics)  
        print("testing semantic:", test_semantics)  
        train_loader, val_loader, test_loader = get_dataloader(  
            root_dir=args.root_dir,  
            model_name=model_name,  
            num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
            train_semantics=train_semantics,  
            test_semantics=test_semantics,  
            batch_size=args.batch_size,  
            num_workers=args.num_workers,
            config=config,
            use_semantic_split=True,  
        )  
    else:  
        print("using normal split...")  
        train_loader, val_loader, test_loader = get_dataloader(  
            root_dir=args.root_dir,  
            model_name=model_name,  
            num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
            batch_size=args.batch_size,  
            num_workers=args.num_workers,
            config=config,
        )  
    logger.info(f"train_loader samples: {len(train_loader)}")  
    logger.info(f"val_loader samples: {len(val_loader)}")  
    logger.info(f"test_loader samples: {len(test_loader)}")  

    model_class = ATTRIBUTOR[config['model_name']]  
    model = model_class(config)  
    logger.info(model.parameters())
    optimizer = choose_optimizer(model, config)  
    scheduler = choose_scheduler(config, optimizer)  
    metric_scoring = choose_metric(config)  

    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=now, log_dir=log_dir)  

    # ----------------- 新增 恢复训练部分 -----------------  
    start_epoch = config.get('start_epoch', 1)  
    best_val_metric = None  
    best_epoch = 0  

    resume_ckpt = args.resume_checkpoint
    if resume_ckpt:  
        try:  
            best_metrics, ckpt_epoch = trainer.load_checkpoint(resume_ckpt)  
            logger.info(f"Resumed checkpoint from {resume_ckpt}")  
            if ckpt_epoch is not None:  
                start_epoch = ckpt_epoch + 1  # 从checkpoint最后一个epoch后开始训练  
                logger.info(f"Resuming from epoch {start_epoch}")  
            if best_metrics is not None:  
                # 根据你的训练指标取合适的字段，假设你的best_val_metric就保存在best_metrics字典中  
                # 这里做示例，假设最优指标字段叫 'val_metric'  
                if 'val_metric' in best_metrics:  
                    best_val_metric = best_metrics['val_metric']  
                    logger.info(f"Restored best_val_metric: {best_val_metric}")  
        except Exception as e:  
            logger.error(f"Failed to load checkpoint {resume_ckpt}: {e}")  
            # 这里你可以选择退出或者继续从头训练  

    n_epochs = config.get('nEpochs', 10)  
    # 若命令行参数里有n_epoch，则用命令行参数覆盖默认值  
    if hasattr(args, 'n_epoch') and args.n_epoch is not None:  
        n_epochs = args.n_epoch  
          
    save_freq = config.get('save_freq', 5)  

    for epoch in range(start_epoch, n_epochs + 1):  
        logger.info(f"#### Epoch {epoch} ####")  
        val_metric, val_metrics = trainer.train_epoch(epoch, train_loader, val_loader={'test': val_loader})  

        if val_metric is not None:  
            if (best_val_metric is None) or (val_metric > best_val_metric):  
                best_val_metric = val_metric  
                best_epoch = epoch  
                logger.info(f"Validation metric improved, saving best checkpoint at epoch {epoch}")  
                trainer.save_checkpoint(filename="ckpt_best.pth", best_metrics={'val_metric': best_val_metric}, epoch=epoch)  

        if epoch % save_freq == 0 or epoch == n_epochs:  
            ckpt_name = f"ckpt_epoch_{epoch}.pth"  
            logger.info(f"Saving checkpoint at epoch {epoch}: {ckpt_name}")  
            trainer.save_checkpoint(filename=ckpt_name, best_metrics={'val_metric': best_val_metric}, epoch=epoch)  

    logger.info(f"Training complete. Best val at epoch {best_epoch}: {best_val_metric}")  

    for writer in trainer.writers.values():  
        writer.close()  

    if args.do_test:  
        degraded_levels = list(range(7))  # 0~6 的等级  

        for degraded in degraded_levels:  
            print(f"Testing with degraded level {degraded}...")  

            if args.use_semantic_split:  
                train_loader, val_loader, test_loader = get_dataloader(  
                    root_dir=args.root_dir,  
                    model_name=model_name,  
                    num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
                    train_semantics=train_semantics,  
                    test_semantics=test_semantics,  
                    batch_size=args.batch_size,  
                    degraded=degraded,  # 传入降级等级  
                    config=config,
                    num_workers=args.num_workers,
                    use_semantic_split=True, 
                )  
            else:  
                train_loader, val_loader, test_loader = get_dataloader(  
                    root_dir=args.root_dir,  
                    model_name=model_name,  
                    num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,  
                    batch_size=args.batch_size,  
                    degraded=degraded,  # 这里也传  
                    config=config,
                    num_workers=args.num_workers
                )  

            # 只针对 test_loader 进行测试，train_loader/val_loader不变或忽略  
            test_metrics = trainer.test(test_loader)  

            # 保存结果  
            result_txt_path = os.path.join(log_dir, f"test_results_degraded_{degraded}.txt")  
            with open(result_txt_path, 'w') as f:  
                f.write(f"Test metrics for degraded={degraded} ({datetime.datetime.now()}):\n")  
                for metric_name, value in test_metrics.items():  
                    if metric_name == "conf_matrix":  
                        f.write(f"{metric_name}:\n{value}\n")  
                    else:  
                        f.write(f"{metric_name}: {value}\n")  

            print(f"Saved test results for degraded={degraded} to {result_txt_path}")  
if __name__ == '__main__':  
    main()  