import os
import yaml

import torch

from mmengine.config import Config
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, ModelSummary


from configs.config import CONF
from SSC.datasets.semantic_kitti_dm import SemanticKITTIDataModule
from LightningTools.pl_model import pl_model


import statistics

class InferenceTimeCallback(pl.Callback):
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.batch_times = []

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.start_event.record()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        self.batch_times.append(elapsed_time_ms)

    def on_validation_end(self, trainer, pl_module):
        median_time = statistics.median(self.batch_times)
        print(f"每个批次的中位推理时间: {median_time:.3f} 毫秒")


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--config_path', default=None)

    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--save_path', default=None)

    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--eval', action='store_true', help='Enable eval mode')

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pred_model', default='CGFormer')
    parser.add_argument('--model_version', default=None, required=True)
    parser.add_argument('--vlm_model', default=None)
    parser.add_argument('--text_encoder', default=None)

    parser.add_argument(
        '--version', '-v',
        type=str,
        default=None,
        help='Exp Name'
    )

    args = parser.parse_args()
    
    if args.config_path:
        cfg = Config.fromfile(args.config_path)
    else:
        config_path = {
            'Conv_Conv' : CONF.PATH.Config_SemanticKITTI_Conv,
            'Conv_Conv_lovasz' : CONF.PATH.Config_SemanticKITTI_Conv,
            'Conv_Conv_single' : CONF.PATH.Config_SemanticKITTI_Conv,
            'Conv_PNA' : CONF.PATH.Config_SemanticKITTI_PNA,
            'Conv_PNAV2' : CONF.PATH.Config_SemanticKITTI_PNA,
            'Conv_PNA_lovasz' : CONF.PATH.Config_SemanticKITTI_PNA,
            'Conv_PNA_single' : CONF.PATH.Config_SemanticKITTI_PNA,

            'ConvTEXT_ConvTEXT' : CONF.PATH.Config_SemanticKITTI_TEXT,
            'ConvTEXT_ConvTEXT_single' : CONF.PATH.Config_SemanticKITTI_TEXT,
            'Conv_ConvPNATEXT' : CONF.PATH.Config_SemanticKITTI_PNATEXT
        }
            
        cfg = Config.fromfile(config_path[args.model_version])

    cfg.update(vars(args))

    return args, cfg

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model(config):
    if config['model_version'] == 'Conv_Conv' or config['model_version'] == 'Conv_Conv_lovasz':
        from UNet.RefinementModule import RefinementModule
        
        model = RefinementModule(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],
            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'Conv_Conv_single':
        from UNet.RefinementModule import RefinementModuleSingle

        model = RefinementModuleSingle(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],
            class_frequencies=config['semantic_kitti_class_frequencies']
        )


    elif config['model_version'] == 'Conv_PNA' or config['model_version'] == 'Conv_PNA_lovasz':
        from UNet.RefinementModulePNA import RefinementModule
        
        model = RefinementModule(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            num_heads=config['model_pna']['num_heads'],
            ffn_cfg=config['ffn_cfg'],

            use_residual=config['model_pna']['use_residual'],
            bias=config['model_pna']['bias'],

            kernel_size=config['model_pna']['kernel_size'],
            dilation=config['model_pna']['dilation'],
            rel_pos_bias=config['model_pna']['rel_pos_bias'],
            qkv_bias=config['model_pna']['qkv_bias'],
            attn_drop=config['model_pna']['attn_drop'],
            proj_drop=config['model_pna']['proj_drop'],

            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'Conv_PNA_single':
        from UNet.RefinementModulePNA import RefinementModuleSingle

        model = RefinementModuleSingle(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            num_heads=config['model_pna']['num_heads'],
            ffn_cfg=config['ffn_cfg'],

            use_residual=config['model_pna']['use_residual'],
            bias=config['model_pna']['bias'],

            kernel_size=config['model_pna']['kernel_size'],
            dilation=config['model_pna']['dilation'],
            rel_pos_bias=config['model_pna']['rel_pos_bias'],
            qkv_bias=config['model_pna']['qkv_bias'],
            attn_drop=config['model_pna']['attn_drop'],
            proj_drop=config['model_pna']['proj_drop'],

            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'Conv_PNAV2':
        from UNet.RefinementModulePNA import RefinementModuleV2
        
        model = RefinementModuleV2(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            num_heads=config['model_pna']['num_heads'],
            ffn_cfg=config['ffn_cfg'],

            use_residual=config['model_pna']['use_residual'],
            bias=config['model_pna']['bias'],

            kernel_size=config['model_pna']['kernel_size'],
            dilation=config['model_pna']['dilation'],
            rel_pos_bias=config['model_pna']['rel_pos_bias'],
            qkv_bias=config['model_pna']['qkv_bias'],
            attn_drop=config['model_pna']['attn_drop'],
            proj_drop=config['model_pna']['proj_drop'],

            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'ConvTEXT_ConvTEXT':
        from UNet.RefinementModuleTEXT import RefinementModule
        
        model = RefinementModule(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            text_encoder=config['text_encoder'],
            text_encoder_dim=config['text_encoder_dim'],
            num_heads=config['model_text']['num_heads'],

            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'Conv_ConvPNATEXT':
        from UNet.RefinementModulePNATEXT import RefinementModule
        
        model = RefinementModule(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            text_encoder=config['text_encoder'],
            text_encoder_dim=config['text_encoder_dim'],
            num_heads=config['model']['num_heads'],

            ffn_cfg=config['ffn_cfg'],

            use_residual=config['model']['use_residual'],
            bias=config['model']['bias'],

            kernel_size=config['model']['kernel_size'],
            dilation=config['model']['dilation'],
            rel_pos_bias=config['model']['rel_pos_bias'],
            qkv_bias=config['model']['qkv_bias'],
            attn_drop=config['model']['attn_drop'],
            proj_drop=config['model']['proj_drop'],


            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    model = pl_model(
        config=config,
        model=model
        )

    dm = SemanticKITTIDataModule(
        data_root=CONF.PATH.DATA_ROOT,
        ann_file=CONF.PATH.DATA_LABEL,
        pred_model=config["pred_model"],
        vlm_model=config["vlm_model"],
        text_encoder=config["text_encoder"]
    )

    return model, dm

def main():
    args, config = parse_config()

    pl.seed_everything(config['seed'])

    # Get Model, DataModule
    model, dm = get_model(config)

    if config['eval']:
        inference_time_callback = InferenceTimeCallback()

        # trainer
        trainer = pl.Trainer(
            devices=1, 
            accelerator="gpu",
            callbacks=[
                ModelSummary(max_depth=-1),
                inference_time_callback
                ],
            logger=False,
            profiler="simple",
            sync_batchnorm=True,
        )

        trainer.validate(model=model, datamodule=dm, ckpt_path=config["ckpt_path"])
    else:
        trainer = pl.Trainer(
            devices=1, 
            accelerator="gpu",
            logger=False,
            profiler="simple",
            sync_batchnorm=True,
            max_steps=1,
        )

        trainer.fit(model=model, datamodule=dm, ckpt_path=config["ckpt_path"])

if __name__ == "__main__":
    main()

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --eval --debug --ckpt_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/ckpts/Conv_Conv_lr_lr/val/mIoU=0.1703.ckpt --config_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/csv_logs/Conv_Conv_lr_lr/config.py

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --eval --debug --ckpt_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/ckpts/Conv_PNA_lr_lr/val/mIoU=0.1719.ckpt --config_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/csv_logs/Conv_PNA_lr_lr/config.py

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --eval --debug --ckpt_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/ckpts/ConvTEXT_ConvTEXT_lr_lr_JinaCLIP/val/mIoU=0.1712.ckpt --config_path /u/home/caoh/projects/MA_Jiachen/ESSC-RM/output_log/csv_logs/ConvTEXT_ConvTEXT_lr_lr_JinaCLIP/config.py

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --pred_model CGFormer --debug --version debug --model_version Conv_Conv

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --pred_model CGFormer --debug --version debug --model_version Conv_PNA

# CUDA_VISIBLE_DEVICES=7 python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/analysis.py --pred_model CGFormer --debug --version debug --vlm_model LLaVA --text_encoder JinaCLIP --model_version ConvTEXT_ConvTEXT