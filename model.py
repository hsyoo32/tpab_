import world
import torch
from dataloader import BasicDataset
from torch import nn, optim
import numpy as np
import pdb
import torch.nn.functional as F
import logging


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PopPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = world.config['predict']
    def forward(self, stage, item):
        # item: N * 10
        # stage: N * 1
        item = item.T
        c_0 = item[:, 0].clone()
        c_1 = item[:, 1].clone()
        x_1 = c_0 - (c_1 - c_0) / self.a
        x_0 = x_1 - (c_0 - x_1) / self.a
        new_items = torch.cat([x_0.reshape(1, -1), x_1.reshape(1, -1), item.T]).T
        new_stages = stage + 2
        return (self.a * ((new_items * F.one_hot(new_stages-1, num_classes=world.config['period']+4)).sum(1) 
                - (new_items * F.one_hot(new_stages-2, num_classes=world.config['period']+4)).sum(1))
                + (new_items * F.one_hot(new_stages-1, num_classes=world.config['period']+4)).sum(1))

# Vanilla MF
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.__init_weight()

        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    # Inference
    def getUsersRating(self, users, predictor, world, stage, item_pop):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())

        return scores

    def bpr_loss(self, users, pos, neg, stage, pos_pop, neg_pop, predictor, world):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)

        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))

        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))

        loss = loss + reg_loss * self.weight_decay
        return loss, pos_scores


# TPAB (MF)
class PureMF_TPAB(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF_TPAB, self).__init__()
        self.config = config
        self.dataset = dataset
        # self.pop_idx = dataset.pop_idx
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.__init_weight()

        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        
    def __init_weight(self):

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=int(self.latent_dim))
        self.embedding_user_pop = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
        self.embedding_item_pop = torch.nn.Embedding(
            num_embeddings=self.dataset.num_item_pop, embedding_dim=int(self.latent_dim))

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_user_pop.weight, std=0.1)
        nn.init.normal_(self.embedding_item_pop.weight, std=0.1)
        logging.info("using Normal distribution N(0,1) initialization for PureMF_TPAB")

    # Inference
    def getUsersRating(self, users, predictor, world, stage, item_pop, item_local_pop, stage_user, user_pop):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        
        if 'global' in world.config['algo']:
            pred_pop = (item_pop * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)
            #user_pred_pop = (user_pop * F.one_hot(stage_user, num_classes=world.config['period']+2)).sum(1)
        else:
            pred_pop = predictor(stage, item_pop.T)
            #user_pred_pop = predictor(stage_user, user_pop.T)
        items_pop_emb = self.get_pop_embeddings(pred_pop, 'item')
        users_pop_emb = self.embedding_user_pop(users)
        ratings = torch.matmul(users_emb, items_emb.t())
        ratings_pop = torch.matmul(users_pop_emb, items_pop_emb.t())
        ratings = ratings + ratings_pop

        return ratings
    
    def bpr_loss(self, users, pos, neg, stage, pos_pop, neg_pop, pos_local_pop, neg_local_pop,
                 users_pop, users_local_pop, predictor, world):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())

        pos_pred_pop = (pos_pop.T * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)
        neg_pred_pop = (neg_pop.T * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)

        users_pop_emb = self.embedding_user_pop(users.long())

        pos_pop_emb = self.get_pop_embeddings(pos_pred_pop, 'item')
        neg_pop_emb = self.get_pop_embeddings(neg_pred_pop, 'item')

        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        pos_scores_pop = torch.sum(users_pop_emb*pos_pop_emb, dim=1) 
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)
        neg_scores_pop = torch.sum(users_pop_emb*neg_pop_emb, dim=1) 
        

        pos_scores = pos_scores + pos_scores_pop
        neg_scores = neg_scores + neg_scores_pop

        
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
                    +users_pop_emb.norm(2).pow(2) + pos_pop_emb.norm(2).pow(2) + neg_pop_emb.norm(2).pow(2)
                    )/float(len(users))

        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))
        loss = loss + reg_loss * self.weight_decay

        if world.config['lambda'] > 0:
            loss_pop = self.switch_concat_(users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb, pos_pop, neg_pop, len(users))
            loss += world.config['lambda'] * loss_pop
        
        return loss

    # Bootstapping loss
    def switch_concat_(self, users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb, pos_pop, neg_pop, batch_size):
        users_ori = torch.cat([users_pop_emb, users_emb], dim=1)
        pos_ori = torch.cat([pos_pop_emb, pos_emb], dim=1)
        neg_ori = torch.cat([neg_pop_emb, neg_emb], dim=1)
        random_order = torch.randperm(batch_size)
        # random_order = torch.randperm(pos_ori.size()[0])
        pos_pop_new = pos_pop_emb[random_order]
        neg_pop_new = neg_pop_emb[random_order]
        pos_new = torch.cat([pos_pop_new, pos_emb], dim=1)
        neg_new = torch.cat([neg_pop_new, neg_emb], dim=1)
        users_pop_new = users_pop_emb[random_order]
        users_new = torch.cat([users_pop_new, users_emb], dim=1)
        loss_new = self.bpr_loss_(users_ori, pos_new, neg_new, batch_size)

        return loss_new

    def bpr_loss_(self, users_emb, pos_emb, neg_emb, batch_size):
        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))/float(batch_size)
        return loss + reg_loss * self.weight_decay

    # Coarsening
    def get_pop_embeddings(self, pred_pop, flag):
        if flag == 'user':
            if world.config['n_pop_group'] > 0:
                num_pop = self.dataset.num_user_pop
                pthres = self.dataset.user_pthres
            else:
                pop_idx_ = self.dataset.user_pop_idx
        elif flag == 'item':
            if world.config['n_pop_group'] > 0:
                num_pop = self.dataset.num_item_pop
                pthres = self.dataset.pthres
            else:
                pop_idx_ = self.dataset.pop_idx
        else:
            raise NotImplementedError
         
        # using popularity as categorical input
        if world.config['n_pop_group'] > 0:
            reindexed_list = []
            for pop in pred_pop.tolist():
                for i in range(num_pop):
                    if pop <= pthres[i]:
                        pop_idx = i
                        break
                    # if pop is greater then pmax, then pop_idx is the last index
                    elif i == num_pop-1:
                        pop_idx = i
                reindexed_list.append(pop_idx)
            if flag == 'user':
                pop_emb = self.embedding_user_pop(torch.tensor(reindexed_list).to(world.device).long())
            elif flag == 'item':
                pop_emb = self.embedding_item_pop(torch.tensor(reindexed_list).to(world.device).long())
        
        # using popularity as categorical input; redistribute popularities to the new groups
        elif world.config['n_pop_group'] < 0:
            # if the element of pred_pop is not in self.pop_idx keys, then replace the value with the closest key
            reindexed_list = []
            for pop in pred_pop.tolist():
                pop_ = round(pop)
                if pop_ not in pop_idx_.keys():
                    new_pop = min(pop_idx_.keys(), key=lambda k: abs(k-pop_))
                else:
                    new_pop = pop_
                reindexed_list.append(pop_idx_[new_pop])
            if flag == 'user':
                pop_emb = self.embedding_user_pop(torch.tensor(reindexed_list).to(world.device).long())
            elif flag == 'item':
                pop_emb = self.embedding_item_pop(torch.tensor(reindexed_list).to(world.device).long())
            
        return pop_emb
    

# Vanilla LightGCN
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.__init_weight()
        self.Graph = self.dataset.getSparseGraph()
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('use NORMAL distribution initilizer for LightGCN')

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # Inference
    def getUsersRating(self, users, predictor, world, stage, item_pop):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        scores = torch.matmul(users_emb, items_emb.t())
        
        return scores

    def bpr_loss(self, users, pos, neg, stage, pos_pop, neg_pop, predictor, world):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)

        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        loss = loss + reg_loss * self.weight_decay
        return loss, pos_scores


# TPAB (LightGCN)
class LightGCN_TPAB(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN_TPAB, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.n_layers_p = self.config['lightGCN_n_layers_p']
        self.keep_prob = self.config['keep_prob']
        self.keep_prob_p = self.config['keep_prob_p']
        self.A_split = self.config['A_split']
        self.__init_weight()
        self.Graph = self.dataset.getSparseGraph()
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        print(f"LightGCN_TPAB is already to go(dropout:{self.config['dropout']})")

        
    def __init_weight(self):

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=int(self.latent_dim))
        self.embedding_user_pop = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
        self.embedding_item_pop = torch.nn.Embedding(
            num_embeddings=self.dataset.num_item_pop, embedding_dim=int(self.latent_dim))

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_user_pop.weight, std=0.1)
        nn.init.normal_(self.embedding_item_pop.weight, std=0.1)
        logging.info("using Normal distribution N(0,1) initialization for LightGCN_TPAB")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self, users_emb, items_emb):
        """
        propagate methods for lightGCN
        """       
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def computer_p(self, users_emb, items_emb):
        """
        propagate methods for lightGCN
        """       
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout_p']:
            g_droped = self.__dropout(self.keep_prob_p)
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers_p):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer(self.embedding_user.weight, self.embedding_item.weight)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # Inference
    def getUsersRating(self, users, predictor, world, stage, item_pop, item_local_pop, stage_user, user_pop):
        all_users, all_items = self.computer(self.embedding_user.weight, self.embedding_item.weight)
        users_emb = all_users[users.long()]
        items_emb = all_items

        # temporal -> always use predicted next-time popularity for inference
        if 'global' in world.config['algo']:
            pred_pop = (item_pop * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)
            #user_pred_pop = (user_pop * F.one_hot(stage_user, num_classes=world.config['period']+2)).sum(1)
        else:
            pred_pop = predictor(stage, item_pop.T)
            #user_pred_pop = predictor(stage_user, user_pop.T)
        
        # all embeddings -> graph convolution
        if 'global' in world.config['algo']:
            item_idx = self.get_reindexed_list(self.dataset.global_item_pop_idx, 'item')
            self.all_users_pop, self.all_items_pop = self.computer_p(self.embedding_user_pop.weight,
                                                        self.embedding_item_pop.weight[item_idx])
            users_pop_emb = self.all_users_pop[users.long()]
            items_pop_emb = self.all_items_pop

        else:
            batch_pos_reidx = self.get_reindexed_list(pred_pop, 'item')
            item_idx = batch_pos_reidx
            self.all_users_pop, self.all_items_pop = self.computer_p(self.embedding_user_pop.weight,
                                                        self.embedding_item_pop.weight[item_idx])
            users_pop_emb = self.all_users_pop[users.long()]
            items_pop_emb = self.all_items_pop

        ratings = torch.matmul(users_emb, items_emb.t())
        ratings_pop = torch.matmul(users_pop_emb, items_pop_emb.t())
        ratings = ratings + ratings_pop

        return ratings
    
    def bpr_loss(self, users, pos, neg, stage, pos_pop, neg_pop, pos_local_pop, neg_local_pop,
                 users_pop, users_local_pop, predictor, world):

        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # true popularity: pos_pop, neg_pop; 
        # pos_pop => list of list of user popularity at each stage
        # extract the user popularity at the current stage using "stage"
        # pos_pred_pop = (pos_pop.T * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)
        # neg_pred_pop = (neg_pop.T * F.one_hot(stage, num_classes=world.config['period']+2)).sum(1)

        if 'global' in world.config['algo']:
            item_idx = self.get_reindexed_list(self.dataset.global_item_pop_idx, 'item')
            user_idx = users.long()
            self.all_users_pop, self.all_items_pop = self.computer_p(self.embedding_user_pop.weight,
                                                    self.embedding_item_pop.weight[item_idx])
            userpopEmb0 = self.embedding_user_pop(users.long())
            users_pop_emb = self.all_users_pop[users.long()]
            pos_pop_emb = self.all_items_pop[pos.long()]
            neg_pop_emb = self.all_items_pop[neg.long()]
            pospopEmb0 = self.embedding_item_pop(item_idx[pos.long()])
            negpopEmb0 = self.embedding_item_pop(item_idx[neg.long()])

        else:
            item_idx_list = self.dataset.sep_temporal_item_pop_idx
            # graph convolution for each time stage graph
            all_users_pop = []
            all_items_pop = []
            for time in range(world.config['period']):
                item_idx = self.get_reindexed_list(item_idx_list[time], 'item')
                all_users_pop_t, all_items_pop_t = self.computer_p(self.embedding_user_pop.weight,
                                                                self.embedding_item_pop.weight[item_idx])
                all_users_pop.append(all_users_pop_t)
                all_items_pop.append(all_items_pop_t)

            all_users_pop = torch.stack(all_users_pop, dim=0)
            all_items_pop = torch.stack(all_items_pop, dim=0)

            pos_pop_emb = all_items_pop[stage, pos.long()]
            neg_pop_emb = all_items_pop[stage, neg.long()]
            users_pop_emb = all_users_pop[stage, users.long()]
            userpopEmb0 = self.embedding_user_pop(users.long())
            pospopEmb0 = self.embedding_item_pop(item_idx[pos.long()])
            negpopEmb0 = self.embedding_item_pop(item_idx[neg.long()])

        # Compute losses based on the embeddings
        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        pos_scores_pop = torch.sum(users_pop_emb*pos_pop_emb, dim=1) 
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)
        neg_scores_pop = torch.sum(users_pop_emb*neg_pop_emb, dim=1) 

        pos_scores = pos_scores + pos_scores_pop
        neg_scores = neg_scores + neg_scores_pop
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)
                    +userpopEmb0.norm(2).pow(2) + pospopEmb0.norm(2).pow(2) + negpopEmb0.norm(2).pow(2)
                    )/float(len(users))

        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))
        loss = loss + reg_loss * self.weight_decay

        # Bootstrapping loss
        if world.config['lambda'] > 0:
            loss_pop = self.switch_concat_(users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, 
                        users_pop_emb, pos_pop_emb, neg_pop_emb, userpopEmb0, pospopEmb0, negpopEmb0, pos_pop, neg_pop, len(users))
            loss += world.config['lambda'] * loss_pop
        
        return loss

    def get_reindexed_list(self, pred_pop, flag):
        if flag == 'user':
            if world.config['n_pop_group'] > 0:
                num_pop = self.dataset.num_user_pop
                pthres = self.dataset.user_pthres
            else:
                pop_idx_ = self.dataset.user_pop_idx
        elif flag == 'item':
            if world.config['n_pop_group'] > 0:
                num_pop = self.dataset.num_item_pop
                pthres = self.dataset.pthres
            else:
                pop_idx_ = self.dataset.pop_idx
        else:
            raise NotImplementedError
        
        # using popularity as categorical input
        if world.config['n_pop_group'] > 0:
            reindexed_list = []
            for pop in pred_pop.tolist():
                for i in range(num_pop):
                    if pop <= pthres[i]:
                        pop_idx = i
                        break
                    # if pop is greater then pmax, then pop_idx is the last index
                    elif i == num_pop-1:
                        pop_idx = i
                reindexed_list.append(pop_idx)
        
        # using popularity as categorical input; redistribute popularities to the new groups
        elif world.config['n_pop_group'] < 0:
            # if the element of pred_pop is not in self.pop_idx keys, then replace the value with the closest key
            reindexed_list = []
            for pop in pred_pop.tolist():
                pop_ = round(pop)
                if pop_ not in pop_idx_.keys():
                    new_pop = min(pop_idx_.keys(), key=lambda k: abs(k-pop_))
                else:
                    new_pop = pop_
                reindexed_list.append(pop_idx_[new_pop])
        
        return torch.tensor(reindexed_list).to(world.device)
        return np.array(reindexed_list)
    
    def switch_concat_(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, 
                       users_pop_emb, pos_pop_emb, neg_pop_emb, userPopEmb0, posPopEmb0, negPopEmb0,
                         pos_pop, neg_pop, batch_size):

        users_ori = torch.cat([users_pop_emb, users_emb], dim=1)
        users_ori_0 = torch.cat([userPopEmb0, userEmb0], dim=1)
        random_order = torch.randperm(batch_size)
        pos_pop_new = pos_pop_emb[random_order]
        neg_pop_new = neg_pop_emb[random_order]
        pos_new = torch.cat([pos_pop_new, pos_emb], dim=1)
        neg_new = torch.cat([neg_pop_new, neg_emb], dim=1)
        pos_new_0 = torch.cat([posPopEmb0[random_order], posEmb0], dim=1)
        neg_new_0 = torch.cat([negPopEmb0[random_order], negEmb0], dim=1)

        users_pop_new = users_pop_emb[random_order]
        users_new = torch.cat([users_pop_new, users_emb], dim=1)
        users_new_0 = torch.cat([userPopEmb0[random_order], userEmb0], dim=1)

        loss_new = self.bpr_loss_(users_ori, pos_new, neg_new, users_ori_0, pos_new_0, neg_new_0, batch_size)

        return loss_new

    def bpr_loss_(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, batch_size):
        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10))
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/float(batch_size)
        return loss + reg_loss * self.weight_decay