import os
import yaml

import torch

from mmengine.config import Config
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor


from configs.config import CONF
from SSC.datasets.semantic_kitti_dm import SemanticKITTIDataModule
from LightningTools.pl_model import pl_model

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default=CONF.PATH.LOG_DIR)
    parser.add_argument('--save_path', default=None)

    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')

    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')

    parser.add_argument('--pred_model', default='CGFormer')
    parser.add_argument('--model_version', default=None, required=True)
    parser.add_argument('--vlm_model', default=None)
    parser.add_argument('--text_encoder', default=None)

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument(
        '--version', '-v',
        type=str,
        default=None,
        help='Exp Name'
    )

    args = parser.parse_args()
    
    config_path = {
        'Conv_Conv' : CONF.PATH.Config_SemanticKITTI_Conv,
        'Conv_Conv_lovasz' : CONF.PATH.Config_SemanticKITTI_Conv,
        'Conv_Conv_single' : CONF.PATH.Config_SemanticKITTI_Conv,
        'Conv_PNA' : CONF.PATH.Config_SemanticKITTI_PNA,
        'Conv_PNAV2' : CONF.PATH.Config_SemanticKITTI_PNA,
        'Conv_PNA_lovasz' : CONF.PATH.Config_SemanticKITTI_PNA,
        'Conv_PNA_single' : CONF.PATH.Config_SemanticKITTI_PNA,

        'ConvTEXT_ConvTEXT' : CONF.PATH.Config_SemanticKITTI_TEXT,
        'ConvTEXT_ConvTEXT_lovasz' : CONF.PATH.Config_SemanticKITTI_TEXT,

        'ConvTEXT_ConvTEXT_single' : CONF.PATH.Config_SemanticKITTI_TEXT,
        'Conv_ConvPNATEXT' : CONF.PATH.Config_SemanticKITTI_PNATEXT
    }
        
    cfg = Config.fromfile(config_path[args.model_version])

    cfg.update(vars(args))

    return args, cfg

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_logger(config):
    if config['debug']:

        logger = False

        callbacks=[]

    else:

        log_folder = config['log_folder']

        check_path(log_folder)
        check_path(os.path.join(log_folder, 'tensorboard'))
        check_path(os.path.join(log_folder, 'csv_logs'))

        version = config['version']
        tb_logger = TensorBoardLogger(
            save_dir=log_folder,
            name='tensorboard',
            version=version
        )
        
        csv_logger = CSVLogger(
            save_dir=log_folder,
            name='csv_logs',
            version=version
        )
        
        logger = [tb_logger, csv_logger]

        check_path(csv_logger.log_dir)
        config.dump(os.path.join(csv_logger.log_dir, 'config.py'))

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_folder, f'ckpts/{version}'),
            monitor='val/mIoU',
            mode='max',
            save_top_k=1,
            save_last=True,
            filename='{val/mIoU:.4f}'
            )
        
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
        ]

    return logger, callbacks

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

    elif config['model_version'] == 'ConvTEXT_ConvTEXT' or config['model_version'] == 'ConvTEXT_ConvTEXT_lovasz':
        from UNet.RefinementModuleTEXT import RefinementModule
        
        model = RefinementModule(
            num_class=config['num_class'],
            geo_feat_channels=config['dim'],

            text_encoder=config['text_encoder'],
            text_encoder_dim=config['text_encoder_dim'],
            num_heads=config['model_text']['num_heads'],

            class_frequencies=config['semantic_kitti_class_frequencies']
        )

    elif config['model_version'] == 'ConvTEXT_ConvTEXT_single':
        from UNet.RefinementModuleTEXT import RefinementModuleSingle
        
        model = RefinementModuleSingle(
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

    # Get Logger, Callbacks
    logger, callbacks = get_logger(config)

    # Get Model, DataModule
    model, dm = get_model(config)

    # trainer
    trainer = pl.Trainer(
        devices=[i for i in range(torch.cuda.device_count())],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
        max_steps=config['training_steps'],
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
        sync_batchnorm=True,
        log_every_n_steps=config['log_every_n_steps'],
        check_val_every_n_epoch=config['check_val_every_n_epoch']
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=config["ckpt_path"])

if __name__ == "__main__":
    main()

# Single superviosion
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model CGFormer --version Conv_Conv_lr_lr_single --model_version Conv_Conv_single
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model CGFormer --version Conv_PNA_lr_lr_single --model_version Conv_PNA_single
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model CGFormer --vlm_model LLaVA --text_encoder JinaCLIP --version ConvTEXT_ConvTEXT_lr_lr_JinaCLIP_single --model_version ConvTEXT_ConvTEXT_single


# Mono
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model MonoScene --version MonoScene_Conv_Conv_lr_lr --model_version Conv_Conv
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model CGFormer --version Conv_Conv_ConvDown_lr_lr --model_version Conv_Conv
# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model MonoScene --version MonoScene_Conv_PNA_lr_lr --model_version Conv_PNA



# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --version Conv_PNA_lovasz_lr_lr --model_version Conv_PNA_lovasz

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model CGFormer --vlm_model LLaVA --text_encoder JinaCLIP --version ConvTEXT_ConvTEXT_lovasz_lr_lr_JinaCLIP --model_version ConvTEXT_ConvTEXT_lovasz



# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --version Conv_PNAV2_lr_lr_ffn96 --model_version Conv_PNAV2

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --vlm_model LLaVA --text_encoder CLIP --version ConvTEXT_ConvTEXT_lr_lr_CLIP --model_version ConvTEXT_ConvTEXT


# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --version Conv_PNA_FEBV2_lr_lr --model_version Conv_PNA


# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --pred_model MonoScene --vlm_model LLaVA --text_encoder JinaCLIP --version MonoScene_ConvTEXT_ConvTEXT_lr_lr_CLIP --model_version ConvTEXT_ConvTEXT


# VLGM

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/train.py --vlm_model InstructBLIP --text_encoder BLIP2 --version 434_InstructBLIP_QFormer --model_version ConvTEXT_ConvTEXT