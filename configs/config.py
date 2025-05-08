import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/u/home/caoh/projects/MA_Jiachen/ESSC-RM'  # TODO: Change path to your ESSC-RM root dir

## data
CONF.PATH.DATA_ROOT = '/u/home/caoh/datasets/SemanticKITTI/dataset'  # TODO: Change path to your dataset root dir
CONF.PATH.DATA_LABEL = os.path.join(CONF.PATH.DATA_ROOT, 'labels')
CONF.PATH.DATA_SAVE_PATH = os.path.join(CONF.PATH.DATA_ROOT, 'eval_output')

## log
CONF.PATH.LOG_DIR = os.path.join(CONF.PATH.BASE, 'output_log')

## config
CONF.PATH.CONFIG_DIR = os.path.join(CONF.PATH.BASE, 'configs')
CONF.PATH.Config_SemanticKITTI_Conv = os.path.join(CONF.PATH.CONFIG_DIR, 'Conv_Conv.py')
CONF.PATH.Config_SemanticKITTI_PNA = os.path.join(CONF.PATH.CONFIG_DIR, 'Conv_PNA.py')
CONF.PATH.Config_SemanticKITTI_TEXT = os.path.join(CONF.PATH.CONFIG_DIR, 'ConvTEXT_ConvTEXT.py')
CONF.PATH.Config_SemanticKITTI_PNATEXT = os.path.join(CONF.PATH.CONFIG_DIR, 'Conv_ConvPNATEXT.py')

# Demo
CONF.PATH.DEMO = os.path.join(CONF.PATH.BASE, 'demo')
