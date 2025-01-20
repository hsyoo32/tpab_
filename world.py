import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import logging
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "."
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
RESULT_PATH = join(ROOT_PATH, 'results')
LOG_PATH = join(ROOT_PATH, 'log')

import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)

logging.basicConfig(filename=args.log_file, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(args.log_file)


config = {}
all_models  = ['mf', 'lgn', 'simgcl']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.batch_size
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['lightGCN_n_layers_p']= args.layer_p
config['dropout'] = args.dropout
config['dropout_p'] = args.dropout_p
config['keep_prob']  = args.keepprob
config['keep_prob_p']  = args.keepprob_p
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['log'] = args.log
config['period'] = args.period
config['predict'] = args.predict


config['algo'] = args.algo
config['log_file'] = args.log_file
# TPAB
config['lambda'] = args.lambda1
config['n_pop_group'] = args.n_pop_group


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
# if dataset not in all_dataset:
#     raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

if 'tpab' in args.algo:
    model_name += '-tpab'



TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)