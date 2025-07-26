import world
import utils
import torch
from torch.autograd import grad
from torch import nn, optim
import numpy as np
import math
from tensorboardX import SummaryWriter
import Procedure
from os.path import join
import dataloader
from parse import parse_args
import register
import torch.utils.data as data
from model import PopPredictor
import logging
import os
from tqdm import tqdm
from time import time

def _check_time(time_, start=False):
    if time_ is None or start:
        time_ = [time()] * 2
        return time_[0], time_
    tmp_time_ = time_[1]
    time_[1] = time()
    return time_[1] - tmp_time_, time_

# ==============================
utils.set_seed(world.seed)
logging.info(">>SEED: {}".format(world.seed))
# ==============================

# construct the train and test datasets
args = parse_args()
dataset = dataloader.DisenData(path = args.data_path)
logging.info("using our dataset")

train_loader = data.DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

predictor = PopPredictor()
predictor = predictor.to(world.device)

weight_file = utils.getFileName()
logging.info(f"load and save to {weight_file}")
# if world.LOAD:
#     try:
#         Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
#         logging.info(f"loaded model weights from {weight_file}")
#     except FileNotFoundError:
#         logging.info(f"{weight_file} not exists, start from beginning")
Neg_k = 1

config = world.config

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    logging.info("not enable tensorflowboard")

try:
    best_recall = 0
    best_epoch = 0
    recall_list = []
    cnt = 0

    # use tqdm to show the training progress
    aver_pos_scores_t_var = []
    aver_loss_t_var = []
    recall_list_v = []
    recall_list_t = []
    loss_list_ = []

    time_ = None
    _, time_ = _check_time(time_, start=True)
    time_list = []
    titer = tqdm(range(world.TRAIN_epochs))
    for epoch in titer:
        _, time_ = _check_time(time_)
        torch.cuda.empty_cache()
        Recmodel.train()
        train_loader.dataset.get_pair_bpr()
        aver_loss = 0.
        aver_pos_scores_t = {}
        for i in range(world.config['period']):
            aver_pos_scores_t[i] = []
        aver_loss_t = {}
        for i in range(world.config['period']):
            aver_loss_t[i] = []
        idx = 0


        tmp_train_loader = []
        for X in train_loader:
            tmp_train_loader.append(X)
            batch_users = X[0].to(world.device)
            batch_pos = X[1].to(world.device)
            batch_neg = X[2].to(world.device)
            batch_stage = X[3].to(world.device)
            batch_pos_inter = torch.stack(X[4])
            batch_neg_inter = torch.stack(X[5])
            batch_pos_inter = batch_pos_inter.to(world.device)
            batch_neg_inter = batch_neg_inter.to(world.device)
            if 'tpab' in world.config['algo']:
                batch_pos_local_inter = torch.stack(X[6])
                batch_neg_local_inter = torch.stack(X[7])
                batch_pos_local_inter = batch_pos_local_inter.to(world.device)
                batch_neg_local_inter = batch_neg_local_inter.to(world.device)

                batch_user_inter = torch.stack(X[8])
                batch_user_inter = batch_user_inter.to(world.device)
                batch_local_user_inter = torch.stack(X[9])
                batch_local_user_inter = batch_local_user_inter.to(world.device)
            
            if 'vanilla' in world.config['algo']:
                #print('batch stage cude', batch_stage.device)
                loss, pos_score = Recmodel.bpr_loss(batch_users, batch_pos, batch_neg, batch_stage, batch_pos_inter, batch_neg_inter, predictor, world)
                loss = torch.mean(loss)
                aver_loss += loss
                idx += 1
                loss.backward()
                Recmodel.opt.step()
                Recmodel.opt.zero_grad()

            elif 'tpab' in world.config['algo']:
                loss = Recmodel.bpr_loss(batch_users, batch_pos, batch_neg, batch_stage, 
                    batch_pos_inter, batch_neg_inter, batch_pos_local_inter, batch_neg_local_inter,
                    batch_user_inter, batch_local_user_inter,
                    predictor, world)
                    
                loss = torch.mean(loss)
                aver_loss += loss
                idx += 1
                loss.backward()
                Recmodel.opt.step()
                Recmodel.opt.zero_grad()
                

        aver_loss = aver_loss / idx
        #logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {aver_loss.item()}')
        titer.set_description(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss: {aver_loss.item()} | file_name: {world.config["log_file"][6:]}')
        training_time, time_ = _check_time(time_)
        time_list.append(training_time)

        #logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss: {aver_loss.item()} | file_name: {world.config["log_file"][6:]}')

        early_stop = 100
        patience = 20
        #if epoch >= 20 and (epoch+1) % 5 == 0:
        if epoch >= 0 and (epoch+1) % 5 == 0:
            v_results = Procedure.Test(dataset, Recmodel, predictor, epoch, w, world.config['multicore'], 0)
            if 'variance' in world.config['algo']:
                recall_list_v.append((epoch+1, v_results[1][0]))
                recall_list_t.append((epoch+1, t_results[1][0]))
            if v_results[1][0] > best_recall:
                best_epoch = epoch
                best_recall = v_results[1][0] # top-10 or top-20
                t_results = Procedure.Test(dataset, Recmodel, predictor, epoch, w, world.config['multicore'], 1)
                best_v, best_t = v_results, t_results
                torch.save(Recmodel.state_dict(), weight_file)
            if epoch == 100:
                recall_list.append((epoch, v_results[1][0]))
            # early stopping
            if epoch > 100:
                recall_list.append((epoch, v_results[1][0]))
                if v_results[1][0] < best_recall:
                    cnt += 1
                else:
                    cnt = 1
                if cnt >= 20:
                    break
        
    logging.info("End train and valid. Best validation epoch is {:03d}.".format(best_epoch+1))
    logging.info("Validation:")
    
    val_str = Procedure.print_results(None, best_v, None)  
    logging.info('Validation:')
    logging.info(val_str)
    result_str = Procedure.print_results(None, None, best_t)
    logging.info("Test:")
    logging.info(result_str)

    result_file = f"{world.config['log_file'][6:]}"
    result_file = os.path.join(world.RESULT_PATH, result_file)
    open(result_file, "w").write(result_str)
    #open(result_file+'_val', "w").write(val_str)

    # time_list = np.array(time_list)
    # time_str = 'total_training_time\t'+str(np.sum(time_list)) + '\n' + 'per_epoch_training_time\t'+str(np.mean(time_list)) + '\n' + 'epoch_num\t'+str(epoch+1)
    # time_str += '\nbest_epoch\t{}\n'.format(best_epoch+1)
    # for i in time_list:
    #     time_str += '{}\t'.format(i)
    # open(result_file+'_time', "w").write(time_str)


finally:
    if world.tensorboard:
        w.close()