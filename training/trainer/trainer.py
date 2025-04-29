import os  
import time  
import datetime  
import torch  
import numpy as np  
from collections import defaultdict  
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter  
from torch.nn.parallel import DistributedDataParallel as DDP  

class Trainer(object):  
    def __init__(self,  
                 config,  
                 model,  
                 optimizer,  
                 scheduler,  
                 logger,  
                 metric_scoring='auc',  
                 time_now=None,
                 log_dir=None):  
        """  
        Trainer初始化，需传入模型、优化器、调度器、logger等。  
        """  
        if config is None or model is None or optimizer is None or logger is None:  
            raise ValueError("config, model, optimizer, logger must be provided")  

        self.config = config  
        self.model = model  
        self.optimizer = optimizer  
        self.scheduler = scheduler  
        self.logger = logger  
        self.metric_scoring = metric_scoring  
        self.writers = {}  
        self.best_metrics_all_time = defaultdict(  
            lambda: defaultdict(lambda: float('-inf') if self.metric_scoring!='eer' else float('inf'))  
        )  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


        # 创建日志目录  
        self.time_now = time_now if time_now else datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  
        self.log_dir = log_dir  
        self._setup_model()  
        self.logger.info(f"Trainer initialized, log_dir: {self.log_dir}")  

        # traindataset name
        self.dataset_name = self._get_dataset_name()  
    
    def _get_dataset_name(self):  
        use_semantic = self.config.get('use_semantic_split', False)  
        if use_semantic:  
            task_id = self.config.get('task_id', 0)  
            return f'semantic_split_{task_id}'  
        else:  
            return 'default_split'  

    def _setup_model(self):  
        self.model.to(self.device)  
        self.model.device = self.device  
        if self.config.get('ddp', False):  
            local_rank = self.config.get('local_rank', 0)  
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)  

    def get_writer(self, phase, dataset_key, metric_key):  
        key = f"{phase}-{dataset_key}-{metric_key}"  
        if key not in self.writers:  
            writer_path = os.path.join(self.log_dir, phase, dataset_key, metric_key, "metric_board")  
            os.makedirs(writer_path, exist_ok=True)  
            self.writers[key] = SummaryWriter(writer_path)  
        return self.writers[key]  

    def set_train(self):  
        self.model.train()  

    def set_eval(self):  
        self.model.eval()  

    def train_step(self, data_dict):  
        """  
        执行单个训练步骤。  
        返回 loss 字典 和 预测结果。  
        """  
        # print(data_dict['label'])
        preds = self.model(data_dict)  
        if isinstance(self.model, DDP):  
            losses = self.model.module.compute_losses(data_dict, preds)  
        else:  
            losses = self.model.compute_losses(data_dict, preds)  
        self.optimizer.zero_grad()  
        losses['overall'].backward()  
        self.optimizer.step()  
        return losses, preds  

    def train_epoch(self, epoch, train_loader, val_loader=None):  
        """  
        完成一个训练epoch逻辑，处理日志、指标更新和可选验证。  
        """  
        self.logger.info(f"===> Epoch[{epoch}] start!")  
        self.set_train()  

        train_loss_recorder = defaultdict(float)  
        train_metric_recorder = defaultdict(float)  
        num_batches = len(train_loader)  

        pbar = tqdm(enumerate(train_loader), total=num_batches)  
        global_step = epoch * num_batches  

        for i, data_dict in pbar:  
            # 将数据移到GPU  
            for k, v in data_dict.items():  
                if v is not None and k != 'name':  
                    if isinstance(v, torch.Tensor):  
                        data_dict[k] = v.to(self.device)  
                    else:  
                        # 其他类型原样保留  
                        data_dict[k] = v   
            # print("\n" + "*" * 40 + "\n", data_dict['label'], "\n" + "*" * 40 + "\n")  
            losses, preds = self.train_step(data_dict)  

            # 统计loss和metric  
            if isinstance(self.model, DDP):  
                batch_metrics = self.model.module.compute_metrics(data_dict, preds)  
            else:  
                batch_metrics = self.model.compute_metrics(data_dict, preds)  

            # 累积更新  
            for k, v in losses.items():  
                train_loss_recorder[k] += v.item()  
            for k, v in batch_metrics.items():  
                train_metric_recorder[k] += v  

            # 每隔一定step写tensorboard和日志  
            if i % 300 == 0:  
                for k, v in train_loss_recorder.items():  
                    avg_v = v / (i+1)  
                    writer = self.get_writer('train', self.dataset_name, k)    
                    writer.add_scalar(f'train_loss/{k}', avg_v, global_step)  
                for k, v in train_metric_recorder.items():  
                    avg_v = v / (i+1)  
                    writer = self.get_writer('train', self.dataset_name, k)   
                    writer.add_scalar(f'train_metric/{k}', avg_v, global_step)  

                loss_str = " | ".join([f"{k}: {train_loss_recorder[k]/(i+1):.4f}" for k in train_loss_recorder])  
                metric_str = " | ".join([f"{k}: {train_metric_recorder[k]/(i+1):.4f}" for k in train_metric_recorder])  
                self.logger.info(f"Step {global_step} - Losses: {loss_str}")  
                self.logger.info(f"Step {global_step} - Metrics: {metric_str}")  

            global_step += 1  

        # 学习率更新  
        if self.scheduler is not None:  
            self.scheduler.step()  

        # 可根据需要在这里加入验证逻辑
        val_metric = None 
        val_metrics = None 
        if val_loader is not None:  
            val_metrics = self.evaluate(val_loader['test'])  
            val_metric = val_metrics.get(self.metric_scoring, None)  

        return val_metric,val_metrics    

    @torch.no_grad()  
    def inference(self, data_dict):  
        self.set_eval()  
        predictions = self.model(data_dict, inference=True)  
        return predictions  

    @torch.no_grad()  
    def evaluate(self, val_loader):  
        self.logger.info("Start evaluation on validation set")  
        self.set_eval()  

        all_logits = []  
        all_labels = []  

        for batch in val_loader:  
            for k, v in batch.items():  
                if v is not None and isinstance(v, torch.Tensor):  
                    batch[k] = v.to(self.device)  

            pred_dict = self.inference(batch)  # 返回dict，含'logits'  
            all_logits.append(pred_dict['logits'])  
            all_labels.append(batch['label'])  

        all_logits = torch.cat(all_logits, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)  

        input_data = {'label': all_labels}  
        pred_dict = {'logits': all_logits}  

        if isinstance(self.model, DDP):  
            metrics = self.model.module.compute_metrics(input_data, pred_dict)  
        else:  
            metrics = self.model.compute_metrics(input_data, pred_dict)  

        self.logger.info(f"Validation metrics: {metrics}")  
        return metrics  
    @torch.no_grad()  
    def test(self, test_loader):  
        self.logger.info("Start testing on test set")  
        self.set_eval()  

        all_logits = []  
        all_labels = []  
        all_semantic_labels = []

        for batch in test_loader:  
            for k, v in batch.items():  
                if v is not None and isinstance(v, torch.Tensor):  
                    batch[k] = v.to(self.device)  

            pred_dict = self.inference(batch)  # 返回dict，含'logits'  
            all_logits.append(pred_dict['logits'])  
            all_labels.append(batch['label']) 
            all_semantic_labels.append(batch['semantic_label']) 

        all_logits = torch.cat(all_logits, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)  
        all_semantic_labels = torch.cat(all_semantic_labels, dim=0)  

        input_data = {'label': all_labels, "semantic_label": all_semantic_labels}  
        pred_dict = {'logits': all_logits}  

        if isinstance(self.model, DDP):  
            metrics = self.model.module.compute_metrics(input_data, pred_dict,test=True)  
        else:  
            metrics = self.model.compute_metrics(input_data, pred_dict,test=True)  

        self.logger.info(f"Test metrics: {metrics}")  
        return metrics  
    
    def save_checkpoint(self, filename="ckpt_best.pth", best_metrics=None, epoch=None):  
        """  
        保存训练检查点，包含模型、优化器、scheduler状态，以及可选的最佳指标和epoch信息。  
        """  
        save_dir = os.path.join(self.log_dir)  
        os.makedirs(save_dir, exist_ok=True)  
        path = os.path.join(save_dir, filename)  

        # 保存模型的state_dict（考虑DDP）  
        if self.config.get('ddp', False) and hasattr(self.model, 'module'):  
            model_state = self.model.module.state_dict()  
        else:  
            model_state = self.model.state_dict()  

        checkpoint = {  
            'model_state_dict': model_state,  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,  
            'best_metrics': best_metrics,  
            'epoch': epoch  
        }  

        torch.save(checkpoint, path)  
        self.logger.info(f"Checkpoint saved to {path}")  

    def load_checkpoint(self, path):  
        """  
        加载检查点，恢复模型、优化器、scheduler状态，并返回辅助信息。  
        """  
        if not os.path.isfile(path):  
            raise FileNotFoundError(f"No checkpoint found at '{path}'")  

        checkpoint = torch.load(path, map_location='cpu')  

        if self.config.get('ddp', False) and hasattr(self.model, 'module'):  
            self.model.module.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            self.model.load_state_dict(checkpoint['model_state_dict'])  

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

        if self.scheduler is not None and checkpoint.get('scheduler_state_dict', None) is not None:  
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  

        self.logger.info(f"Loaded checkpoint from {path}")  

        best_metrics = checkpoint.get('best_metrics', None)  
        epoch = checkpoint.get('epoch', None)  

        return best_metrics, epoch  
