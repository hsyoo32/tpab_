import world
import utils
import torch
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import dataloader
from parse import parse_args
import register
import torch.utils.data as data
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
from model import PopPredictor
# ==============================

# construct the train and test datasets
args = parse_args()
# dataset = dataloader.Loader(path = args.data_path, group_num = args.group_num)
dataset = dataloader.DROData(path = args.data_path, group_num = args.group_num) 
train_loader = data.DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

predictor = PopPredictor()
predictor = predictor.to(world.device)
bpr = utils.BPRLoss(Recmodel, predictor, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    print("not enable tensorflowboard")

print("Validation:")
v_results = Procedure.Test(dataset, Recmodel, predictor, 0, w, world.config['multicore'], 0)
Procedure.print_results(None, v_results, None)
print("Test:")
t_results = Procedure.Test(dataset, Recmodel, predictor, 0, w, world.config['multicore'], 1)
Procedure.print_results(None, None, t_results)

group_num = 5
for i in range(group_num):
    t_group= Procedure.Test_group(i, dataset, Recmodel, predictor, 0, w, world.config['multicore'], 1)
    Procedure.print_results_group(i, None, None, t_group)


