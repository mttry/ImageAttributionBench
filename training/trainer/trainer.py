import os  
import time  
import datetime  
import torch  
import numpy as np  
from collections import defaultdict  
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter  
from torch.nn.parallel import DistributedDataParallel as DDP  
from attributors import * 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Trainer(object):  
    def __init__(self,  
                 config,  
                 model,  
                 optimizer,  
                 scheduler,  
                 logger,  
                 metric_scoring='acc',  
                 time_now=None,
                 log_dir=None,
                 test_mode=False):    
        if test_mode and (config is None or model is None or logger is None):
            raise ValueError("testing, config, model, logger must be provided")  
        elif (not test_mode) and (config is None or model is None or optimizer is None or logger is None):  
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


        self.time_now = time_now if time_now else datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  
        self.log_dir = log_dir  
        self._setup_model()  
        self.logger.info(f"Trainer initialized, log_dir: {self.log_dir}")  

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
            local_rank = self.config.get(local_rank, 0)
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
        # print(data_dict['label'])
        num_stages = self.config.get("num_stages", 1)
        if num_stages == 1: # most of the attributors
            preds = self.model(data_dict)  
            if isinstance(self.model, DDP):  
                losses = self.model.module.compute_losses(data_dict, preds)  
            else:  
                losses = self.model.compute_losses(data_dict, preds)  
            self.optimizer.zero_grad()  
            losses['overall'].backward()  
            self.optimizer.step()  
            return losses, preds  
        else:
            losses_dict = {}
            for idx in range(num_stages):
                stage_no = idx+1
                preds = self.model(data_dict, stage_no=stage_no)  
                if isinstance(self.model, DDP):  
                    losses = self.model.module.compute_losses(data_dict, preds, stage_no=stage_no)  
                    losses_dict.update(losses)
                else:  
                    losses = self.model.compute_losses(data_dict, preds, stage_no=stage_no)
                    losses_dict.update(losses)  
                self.optimizer.zero_grad()  
                losses['overall'].backward()  
                self.optimizer.step()  
        return losses_dict, preds 

    def train_epoch(self, epoch, train_loader, val_loader=None):  
        self.logger.info(f"===> Epoch[{epoch}] start!")  
        self.set_train()  

        train_loss_recorder = defaultdict(float)  
        train_metric_recorder = defaultdict(float)  
        val_metric = None  
        val_metrics = None  

        if isinstance(self.model, ClipLrAttributor):  
            self.logger.info("Collecting full train set for linear probe training...")  
            self.model.train_linear_probe(train_loader)  
            self.logger.info("Linear probe training done.")  

        elif isinstance(self.model, POSEAttributor):
            self.model.train_epoch_POSE(epoch, train_loader, optimizer=self.optimizer,scheduler=self.scheduler, logger=self.logger)

        else:  
            num_batches = len(train_loader)  
            pbar = tqdm(enumerate(train_loader), total=num_batches)  
            global_step = epoch * num_batches  

            for i, data_dict in pbar:  
                for k, v in data_dict.items():  
                    if v is not None and k != 'name' and isinstance(v, torch.Tensor):  
                        data_dict[k] = v.to(self.device)  

                losses, preds = self.train_step(data_dict)  

                if isinstance(self.model, DDP):  
                    batch_metrics = self.model.module.compute_metrics(data_dict, preds)  
                else:  
                    batch_metrics = self.model.compute_metrics(data_dict, preds)  

                for k, v in losses.items():  
                    train_loss_recorder[k] += v.item() if isinstance(v, torch.Tensor) else float(v)  
                for k, v in batch_metrics.items():  
                    train_metric_recorder[k] += v  

                if i % 300 == 0:  
                    for k, v in train_loss_recorder.items():  
                        avg_v = v / (i + 1)  
                        writer = self.get_writer('train', self.dataset_name, k)  
                        writer.add_scalar(f'train_loss/{k}', avg_v, global_step)  
                    for k, v in train_metric_recorder.items():  
                        avg_v = v / (i + 1)  
                        writer = self.get_writer('train', self.dataset_name, k)  
                        writer.add_scalar(f'train_metric/{k}', avg_v, global_step)  

                    loss_str = " | ".join([f"{k}: {train_loss_recorder[k]/(i+1):.4f}" for k in train_loss_recorder])  
                    metric_str = " | ".join([f"{k}: {train_metric_recorder[k]/(i+1):.4f}" for k in train_metric_recorder])  
                    self.logger.info(f"Step {global_step} - Losses: {loss_str}")  
                    self.logger.info(f"Step {global_step} - Metrics: {metric_str}")  

                global_step += 1  

            if self.scheduler is not None:  
                self.scheduler.step()  
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Current learning rate: {param_group['lr']}")

        if val_loader is not None:  
            val_metrics = self.evaluate(val_loader['test'])  
            val_metric = val_metrics.get(self.metric_scoring, None)  
            self.logger.info(f"Validation metrics: {val_metrics}")  

        return val_metric, val_metrics  

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

            pred_dict = self.inference(batch)
            all_logits.append(pred_dict['logits'])  
            # print(pred_dict['logits'].shape)
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

        for batch in tqdm(test_loader, total=len(test_loader)):  
            for k, v in batch.items():  
                if v is not None and isinstance(v, torch.Tensor):  
                    batch[k] = v.to(self.device)  

            pred_dict = self.inference(batch)
            
            logits_cpu = pred_dict['logits'].detach().cpu()
            label_cpu = batch['label'].detach().cpu()
            semantic_cpu = batch['semantic_label'].detach().cpu()

            all_logits.append(logits_cpu)
            all_labels.append(label_cpu)
            all_semantic_labels.append(semantic_cpu)

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
        save_dir = os.path.join(self.log_dir)  
        os.makedirs(save_dir, exist_ok=True)  
        path = os.path.join(save_dir, filename)  

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
        if not os.path.isfile(path):  
            raise FileNotFoundError(f"No checkpoint found at '{path}'")  

        checkpoint = torch.load(path, map_location='cpu')  

        if self.config.get('ddp', False) and hasattr(self.model, 'module'):  
            self.model.module.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            self.model.load_state_dict(checkpoint['model_state_dict'])  

        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

        if self.scheduler is not None and checkpoint.get('scheduler_state_dict', None) is not None:  
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  

        self.logger.info(f"Loaded checkpoint from {path}")  

        best_metrics = checkpoint.get('best_metrics', None)  
        epoch = checkpoint.get('epoch', None)  

        return best_metrics, epoch  


    @torch.no_grad()
    def visualize(self, test_loader, semantic_seperate=False, feature_reload_path="", label_name_map=None):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from utils.visualize_utils import draw_scatter_with_int_legend
        self.logger.info("Start visualizing features on test set")

        if feature_reload_path and os.path.isfile(feature_reload_path):
            self.logger.info(f"Reloading features from {feature_reload_path}")
            data = np.load(feature_reload_path)
            features_2d = data['features_2d']
            all_labels = data['labels']
            all_semantic_labels = data['semantic_labels']
        else:
            from sklearn.manifold import TSNE
            self.set_eval()
            all_features = []
            all_labels = []
            all_semantic_labels = []

            for batch in tqdm(test_loader, total=len(test_loader)):
                for k, v in batch.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                if isinstance(self.model, DDP):
                    #FIXME: repmix need an arg "inference"
                    features = self.model.module.extract_features(batch)
                    # features = self.model.module.extract_features(batch, True)
                else:
                    features = self.model.extract_features(batch)
                    # features = self.model.extract_features(batch, True)

                label_cpu = batch['label'].detach().cpu()
                semantic_cpu = batch['semantic_label'].detach().cpu()
                features_cpu = features.detach().cpu()

                all_features.append(features_cpu)
                all_labels.append(label_cpu)
                all_semantic_labels.append(semantic_cpu)

            all_features = torch.cat(all_features, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_semantic_labels = torch.cat(all_semantic_labels, dim=0).numpy()
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(all_features)

        all_labels = all_labels.astype(int)
        all_semantic_labels = all_semantic_labels.astype(int)

        unique_semantics = np.unique(all_semantic_labels)
        figs_semantic = {}
        if semantic_seperate:
            for semantic in np.unique(all_semantic_labels):
                mask = all_semantic_labels == semantic
                X = features_2d[mask]
                y = all_labels[mask]
                title = f"Semantic label: {semantic}"
                figs_semantic[semantic] = draw_scatter_with_int_legend(X, y, title=title, label_name_map=label_name_map)
            self.logger.info("Visualization finished with semantic_seperate=True.")

        fig = draw_scatter_with_int_legend(
            features_2d, all_labels,
            title="Feature Visualization (All Semantics, t-SNE)",
            label_name_map=label_name_map
        )
        self.logger.info("Visualization finished with semantic_seperate=False.")

        return {
            "figs_semantic": figs_semantic,      
            "fig_all": fig,                      
            "features_2d": features_2d,          
            "labels": all_labels,                 
            "semantic_labels": all_semantic_labels
        }
    
