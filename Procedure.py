import world
import numpy as np
import torch
import utils
import math
import dataloader
from pprint import pprint
# from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import pdb
from torch import nn, optim
import logging


CORES = multiprocessing.cpu_count() // 2
        
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []

    for index in range(len(topN)):
        # print(f'top {topN[index]}\n')
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1

        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return precision, recall, NDCG, MRR


def Test(dataset, Recmodel, predictor, epoch, w=None, multicore=0, flag=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    if flag == 0:
        testDict = dataset.valid_dict
    else:
        testDict = dataset.test_dict
    item_num = dataset.m_item
    user_num = dataset.n_user
    if flag == 0 :
        stage = torch.full((item_num,1), world.config['period'])
    else:
        stage = torch.full((item_num,1), world.config['period']+1)
    stage = torch.squeeze(stage).to(world.device)
    item_inter = dataset.item_inter
    item_pop = []
    for i in range(item_num):
        item_pop.append(item_inter[i])
    # stage = torch.squeeze(stage)

    item_pop = torch.Tensor(item_pop)
    # stage = stage.to(world.device)
    item_pop = item_pop.to(world.device)

    if 'tpab' in world.config['algo']:
        local_item_inter = dataset.local_item_inter
        local_item_pop = []
        for i in range(item_num):
            local_item_pop.append(local_item_inter[i])
        local_item_pop = torch.Tensor(local_item_pop)
        local_item_pop = local_item_pop.to(world.device)


    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            if 'tpab' in world.config['algo']:
                # make user_pop with only batch users
                user_pop = []
                for i in batch_users:
                    user_pop.append(dataset.user_inter[i])
                user_pop = torch.Tensor(user_pop).to(world.device)
                # make batch_stage_user with only batch users
                if flag == 0:
                    batch_stage_user = torch.full((len(batch_users),1), world.config['period'])
                else:
                    batch_stage_user = torch.full((len(batch_users),1), world.config['period']+1)
                batch_stage_user = torch.squeeze(batch_stage_user).to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu, predictor, world, stage, 
                                                 item_pop, local_item_pop, batch_stage_user, user_pop)
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu, predictor, world, stage, item_pop)

            exclude_index = []
            exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if flag:
                for range_i, items in enumerate(valid_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.extend(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(groundTrue)

        assert total_batch == len(users_list)
        precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list,rating_list,[10,20,50,100])

        if multicore == 1:
            pool.close()
        return precision, recall, NDCG, MRR

def print_results(loss, valid_result, test_result):
    result_str = ''
    """output the evaluation results."""
    if loss is not None:
        logging.info("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        logging.info("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
        # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(valid_result[1][0]) + '\n'
        result_str += 'NDCG@10\t' + str(valid_result[2][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(valid_result[1][1]) + '\n'
        result_str += 'NDCG@20\t' + str(valid_result[2][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(valid_result[1][2]) + '\n'
        result_str += 'NDCG@50\t' + str(valid_result[2][2]) + '\n'
    if test_result is not None: 
        logging.info("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
        # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(test_result[1][0]) + '\n'
        result_str += 'NDCG@10\t' + str(test_result[2][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(test_result[1][1]) + '\n'
        result_str += 'NDCG@20\t' + str(test_result[2][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(test_result[1][2]) + '\n'
        result_str += 'NDCG@50\t' + str(test_result[2][2]) + '\n'

    return result_str
        
# def print_results_group(i, loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if i is not None:
        if valid_result is not None: 
            print("[Valid_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))

    else:
        if valid_result is not None: 
            print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))