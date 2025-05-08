import os
import torch
import numpy as np

import pytorch_lightning as pl

#from SSC.loss.metric import SSCMetrics
from SSC.loss.metric_torch import SSCMetrics
from SSC.loss.utils import get_inv_map

class pl_model(pl.LightningModule):
    def __init__(
        self,
        config,
        model,
    ):
        super(pl_model, self).__init__()

        self.config = config
        self.model_version = config['model_version']
        self.model = model
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])

        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']

        self.debug = config['debug']

        self.version = config['version']
        self.pred_model = config['pred_model']

    def forward_train(self, data_dict):
        gt_occ_256 = data_dict['gt_occ_256']  # [1, 256, 256, 32]
        gt_occ_128 = data_dict['gt_occ_128']  # [1, 128, 128, 16]
        gt_occ_64 = data_dict['gt_occ_64']  # [1, 64, 64, 8]
        gt_occ_32 = data_dict['gt_occ_32']  # [1, 32, 32, 4]

        input_occ = data_dict['input_occ'] # [1, 256, 256, 32]
        
        losses = dict()

        if self.model_version == 'Conv_Conv' or self.model_version == 'Conv_PNA' or self.model_version == 'Conv_PNAV2':
            x_32, x_64, x_128, x_256 = self.model(input_occ)

            losses_occupancy = self.model.loss(
                output_voxels_list=[x_32, x_64, x_128, x_256],
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256]
            )

            losses.update(losses_occupancy)

        elif self.model_version == 'Conv_Conv_lovasz' or self.model_version == 'Conv_PNA_lovasz':
            x_32, x_64, x_128, x_256 = self.model(input_occ)

            losses_occupancy = self.model.loss_V2(
                output_voxels_list=[x_32, x_64, x_128, x_256],
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256]
            )

            losses.update(losses_occupancy)

        elif self.model_version == 'ConvTEXT_ConvTEXT' or self.model_version == 'Conv_ConvPNATEXT':
            x_32, x_64, x_128, x_256 = self.model(input_occ, data_dict['text_feat'])

            losses_occupancy = self.model.loss(
                output_voxels_list=[x_32, x_64, x_128, x_256],
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256]
            )

            losses.update(losses_occupancy)

        elif self.model_version == 'Conv_Conv_single' or self.model_version == 'Conv_PNA_single':
            x_32, x_64, x_128, x_256 = self.model(input_occ)

            losses_occupancy = self.model.loss_single(
                output_voxels_list=[x_256],
                target_voxels_list=[gt_occ_256]
            )

            losses.update(losses_occupancy)

        elif self.model_version == 'ConvTEXT_ConvTEXT_single':
            x_32, x_64, x_128, x_256 = self.model(input_occ, data_dict['text_feat'])

            losses_occupancy = self.model.loss_single(
                output_voxels_list=[x_256],
                target_voxels_list=[gt_occ_256]
            )

            losses.update(losses_occupancy)

        elif self.model_version == 'ConvTEXT_ConvTEXT_lovasz':
            x_32, x_64, x_128, x_256 = self.model(input_occ, data_dict['text_feat'])

            losses_occupancy = self.model.loss_V2(
                output_voxels_list=[x_32, x_64, x_128, x_256],
                target_voxels_list=[gt_occ_32, gt_occ_64, gt_occ_128, gt_occ_256]
            )

            losses.update(losses_occupancy)

        pred = torch.argmax(x_256, dim=1)
            
        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ_256
        }

        return train_output

    def forward_test(self, data_dict):
        input_occ = data_dict['input_occ'] # [1, 256, 256, 32]
        gt_occ_256 = data_dict['gt_occ_256']

        if self.model_version == 'Conv_Conv' or self.model_version == 'Conv_PNA' or self.model_version == 'Conv_PNAV2' or self.model_version == 'Conv_Conv_lovasz' or self.model_version == 'Conv_PNA_lovasz' or self.model_version == 'Conv_Conv_single' or self.model_version == 'Conv_PNA_single':
            x_32, x_64, x_128, x_256 = self.model(input_occ)

        elif self.model_version == 'ConvTEXT_ConvTEXT' or self.model_version == 'Conv_ConvPNATEXT' or self.model_version == 'ConvTEXT_ConvTEXT_single' or self.model_version == 'ConvTEXT_ConvTEXT_lovasz':
            x_32, x_64, x_128, x_256 = self.model(input_occ, data_dict['text_feat'])

        pred = torch.argmax(x_256, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ_256
        }

        return test_output
        
    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True,
            prog_bar=True)
        
        if not self.pretrain:
            #pred = output_dict['pred'].detach().cpu().numpy()
            #gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
            
            #self.train_metrics.add_batch(pred, gt_occ)

            pred = output_dict['pred'].detach()
            gt_occ = output_dict['gt_occ'].detach()

            self.train_metrics.update(pred, gt_occ)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.debug:
            reserved_bytes  = torch.cuda.memory_reserved()

            reserved_MB = reserved_bytes / (1024 ** 2)

            print(f"Reserved_MB: {reserved_MB:.2f} MB")

            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             print(f"[Warning] Parameter '{name}' requires grad but grad is None (maybe unused)")
            #         else:
            #             print(f"[OK] Parameter '{name}' has grad")
            #     else:
            #         print(f"[Info] Parameter '{name}' does not require grad")

    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            #pred = output_dict['pred'].detach().cpu().numpy()
            #gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            #self.val_metrics.add_batch(pred, gt_occ)

            pred = output_dict['pred'].detach()
            gt_occ = output_dict['gt_occ'].detach()

            self.val_metrics.update(pred, gt_occ)

    def on_validation_epoch_end(self):
        if self.debug:
            reserved_bytes  = torch.cuda.memory_reserved()

            reserved_MB = reserved_bytes / (1024 ** 2)

            print(f"Reserved_MB: {reserved_MB:.2f} MB")
            
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            #stats = metric.get_stats()
            stats = metric.compute()

            if prefix == 'val':
                for name, iou in zip(self.class_names, stats['iou_ssc']):
                    self.log(f"{prefix}/{name}/IoU", iou, sync_dist=True)

            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()
        
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach()
        gt_occ = output_dict['gt_occ'].detach()

        if gt_occ is not None:
            self.test_metrics.update(pred, gt_occ)

        pred = output_dict['pred'].cpu().numpy()

        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['sequence'][0]
            frame_id = batch['frame_id'][0]
            save_folder = "{}/{}/{}/{}".format(self.save_path, self.pred_model, self.version, sequence_id)
            #save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            save_file = os.path.join(save_folder, "{}.npy".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)

            output_voxels = output_voxels.squeeze(axis=0)
            np.save(save_file, output_voxels.astype(np.float32))
            
        
    
    def on_test_epoch_end(self):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            #stats = metric.get_stats()
            stats = metric.compute()

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                print(name + ":", iou)

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()

    def configure_optimizers(self):
        if self.config['optimizer']['type'] == 'AdamW':

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )

        else:
            raise NotImplementedError(f"Optimizer {self.config['optimizer']['type']} is not implemented.")
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['lr_scheduler']['max_lr'],
                total_steps=self.config['lr_scheduler']['total_steps'],
                pct_start=self.config['lr_scheduler']['pct_start'],
                cycle_momentum=self.config['lr_scheduler']['cycle_momentum'],
                anneal_strategy=self.config['lr_scheduler']['anneal_strategy'])

            interval=self.config['lr_scheduler']['interval']
            frequency=self.config['lr_scheduler']['frequency']
        else:
            raise NotImplementedError(f"lr_scheduler {self.config['lr_scheduler']['type']} is not implemented.")
        
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': interval,
            'frequency': frequency
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }