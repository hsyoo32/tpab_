import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from time import time
import pdb
import math
import logging

class BasicDataset(Dataset):
	def __init__(self):
		print("init dataset")
	
	@property
	def n_users(self):
		raise NotImplementedError
	
	@property
	def m_items(self):
		raise NotImplementedError
	
	@property
	def trainDataSize(self):
		raise NotImplementedError
	
	@property
	def validDict(self):
		raise NotImplementedError

	@property
	def testDict(self):
		raise NotImplementedError
	
	@property
	def allPos(self):
		raise NotImplementedError
	
	def getUserItemFeedback(self, users, items):
		raise NotImplementedError
	
	def getUserPosItems(self, users):
		raise NotImplementedError
	
	def getUserNegItems(self, users):
		"""
		not necessary for large dataset
		it's stupid to return all neg items in super large dataset
		"""
		raise NotImplementedError
	
	def getSparseGraph(self):
		"""
		build a graph in torch.sparse.IntTensor.
		Details in NGCF's matrix form
		A = 
			|I,   R|
			|R^T, I|
		"""
		raise NotImplementedError

class DisenData(BasicDataset):

	def __init__(self,config = world.config, path = None):
		# train or test
		print(f'loading [{path}]')
		# self.num_ng = config['num_ng']
		self.split = config['A_split']
		self.folds = config['A_n_fold']
		self.mode_dict = {'train': 0, "test": 1}
		self.mode = self.mode_dict['train']
		self.n_user = 0
		self.m_item = 0
		train_file = path + '/training_dict.npy'
		valid_file = path + '/validation_dict.npy'
		test_file = path + '/testing_dict.npy'
		time_file = path + '/interaction_time_dict.npy'

		self.path = path
		trainUniqueUsers, trainItem, trainUser = [], [], []
		validUniqueUsers, validItem, validUser = [], [], []
		testUniqueUsers, testItem, testUser = [], [], []
		self.traindataSize = 0
		self.validDataSize = 0
		self.testDataSize = 0

		self.train_dict = np.load(train_file, allow_pickle=True).item()
		self.valid_dict = np.load(valid_file, allow_pickle=True).item()
		self.test_dict = np.load(test_file, allow_pickle=True).item()
		self.time_dict = np.load(time_file, allow_pickle=True).item()

		for uid in self.train_dict.keys():
			trainUniqueUsers.append(uid)
			trainUser.extend([uid] * len(self.train_dict[uid]))
			trainItem.extend(self.train_dict[uid])
			self.m_item = max(self.m_item, max(self.train_dict[uid]))
			self.n_user = max(self.n_user, uid)
			self.traindataSize += len(self.train_dict[uid])

		# self.group = self.get_group(self.train_dict, group_num, self.traindataSize)
		# self.group_user = self.get_user_group(self.train_dict, group_num)
		self.trainUniqueUsers = np.array(trainUniqueUsers)
		self.trainUser = np.array(trainUser)
		self.trainItem = np.array(trainItem)

		for uid in self.valid_dict.keys():
			if len(self.valid_dict[uid]) != 0:
				validUniqueUsers.append(uid)
				validUser.extend([uid] * len(self.valid_dict[uid]))
				validItem.extend(self.valid_dict[uid])
				self.m_item = max(self.m_item, max(self.valid_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.validDataSize += len(self.valid_dict[uid])
		self.validUniqueUsers = np.array(validUniqueUsers)
		self.validUser = np.array(validUser)
		self.validItem = np.array(validItem)

		for uid in self.test_dict.keys():
			if len(self.test_dict[uid]) != 0:
				testUniqueUsers.append(uid)
				testUser.extend([uid] * len(self.test_dict[uid]))
				testItem.extend(self.test_dict[uid])
				self.m_item = max(self.m_item, max(self.test_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.testDataSize += len(self.test_dict[uid])
		self.testUniqueUsers = np.array(testUniqueUsers)
		self.testUser = np.array(testUser)
		self.testItem = np.array(testItem)
		self.m_item += 1
		self.n_user += 1
		
		self.Graph = None
		print(f"{self.trainDataSize} interactions for training")
		print(f"{self.validDataSize} interactions for validation")
		print(f"{self.testDataSize} interactions for testing")
		print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}")

		# (users,items), bipartite graph
		self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
									  shape=(self.n_user, self.m_item))
		self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
		self.users_D[self.users_D == 0.] = 1
		self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
		self.items_D[self.items_D == 0.] = 1.
		# pre-calculate
		self._allPos = self.getUserPosItems(list(range(self.n_user)))


		self.pair_stage, self.local_item_inter, self.abs_item_inter = self.get_item_inter(self.train_dict, self.valid_dict, self.test_dict, self.time_dict, config)

		if 'local' in config['algo']:
			self.item_inter = self.local_item_inter.copy()
		else:
			self.item_inter = self.abs_item_inter.copy()

			self.temporal_item_pop_idx = np.zeros(self.m_item, dtype=int)
			for item_id, item_inter in self.item_inter.items():
				self.temporal_item_pop_idx[item_id] = int(sum(item_inter)/len(item_inter))

			#if 'tempgcn' in config['algo']:
			self.sep_temporal_item_pop_idx = []
			for time in range(config['period']):
				self.sep_temporal_item_pop_idx.append(np.zeros(self.m_item, dtype=int))
			for item_id, item_inter in self.item_inter.items():
				for time in range(config['period']):
					self.sep_temporal_item_pop_idx[time][item_id] = item_inter[time]


		if 'global' in config['algo'] or 'baseline' in config['algo']:
			# get global popularity of each item => sum of cnt(popularity)
			self.global_item_inter = {}
			self.global_item_pop_idx = np.zeros(self.m_item)
			for item_id, item_inter in self.item_inter.items():
				self.global_item_inter[item_id] = [sum(item_inter)] * (config['period'] + 1 * 2)
				self.global_item_pop_idx[item_id] = sum(item_inter)
			self.item_inter = self.global_item_inter


		# To show popularity statistics
		pop_stat = {}
		for item_id, item_inter in self.item_inter.items():
			for pop in item_inter:
				if pop not in pop_stat:
					pop_stat[pop] = 0
				pop_stat[pop] += 1
		# sort pop_stat dict based on keys
		pop_stat = dict(sorted(pop_stat.items(), key=lambda item: item[0]))
		#logging.info(pop_stat)


		pop_item = []
		for item_id, item_inter in self.item_inter.items():
			pop_item.extend(item_inter)
		sorted_pop_item = list(set(pop_item))
		sorted_pop_item = sorted(sorted_pop_item)
		#logging.info(sorted_pop_item)
		#logging.info(len(sorted_pop_item))

		if config['n_pop_group'] < 0:
			self.num_item_pop = len(sorted_pop_item)
			self.pop_idx = {}
			for i, pop in enumerate(sorted_pop_item):
				self.pop_idx[pop] = i
			#logging.info('item pop_idx {}'.format(self.pop_idx))
		else:
			self.num_item_pop = config['n_pop_group']
			pmin = min(sorted_pop_item)
			pmax = max(sorted_pop_item)
			self.pthres = []
			for i in range(self.num_item_pop):
				if 'uniform' in config['algo']:
					self.pthres.append(pmax/self.num_item_pop * (i+1))
				else:
					self.pthres.append(pmax** ((1/self.num_item_pop) * (i+1)))
			#logging.info('item pmax {}'.format(pmax))
			#logging.info('item pthres {}'.format(self.pthres))



		# do the same for user interaction
		_, self.local_user_inter, self.abs_user_inter = self.get_user_inter(self.train_dict, self.valid_dict, self.test_dict, self.time_dict, config)

		if 'local' in config['algo']:
			self.user_inter = self.local_user_inter.copy()
		else:
			self.user_inter = self.abs_user_inter.copy()

			self.temporal_user_pop_idx = np.zeros(self.n_user, dtype=int)
			for user_id, user_inter in self.user_inter.items():
				self.temporal_user_pop_idx[user_id] = int(sum(user_inter)/len(user_inter))
		
		if 'global' in config['algo'] or 'baseline' in config['algo']:
			# get global popularity of each item => sum of cnt(popularity)
			self.global_user_inter = {}
			self.global_user_pop_idx = np.zeros(self.n_user, dtype=int)
			for user_id, user_inter in self.user_inter.items():
				self.global_user_inter[user_id] = [sum(user_inter)] * (config['period'] + 1 * 2)
				self.global_user_pop_idx[user_id] = sum(user_inter)
			self.user_inter = self.global_user_inter

		
		# popularity indexing
		pop_user = []
		for user_id, user_inter in self.user_inter.items():
			pop_user.extend(user_inter)
		sorted_pop_user = (list(set(pop_user)))
		sorted_pop_user.sort()
		#logging.info(sorted_pop_user)
		#logging.info(len(sorted_pop_user))

		if config['n_pop_group'] < 0:
			self.num_user_pop = len(sorted_pop_user)
			self.user_pop_idx = {}
			for i, pop in enumerate(sorted_pop_user):
				self.user_pop_idx[pop] = i
			#logging.info('user pop_idx {}'.format(self.user_pop_idx))
		else:
			self.num_user_pop = config['n_pop_group']
			# do not need self.pop_idx because we can directly get the index of popularity while training
			pmin = sorted_pop_user[0]
			pmax = max(sorted_pop_user)
			self.user_pthres = []
			for i in range(self.num_user_pop):
				if 'uniform' in config['algo']:
					self.user_pthres.append(pmax/self.num_user_pop * (i+1))
				else:
					self.user_pthres.append(pmax**((1/self.num_user_pop) * (i+1)))
			#logging.info('user pmax {}'.format(pmax))
			#logging.info('user pthres {}'.format(self.user_pthres))

		#logging.info(f"{world.dataset} is ready to go")

	def get_item_inter(self, train_dict, valid_dict, test_dict, time_dict, config):
		pair_stage = {}
		item_inter = {}
		stage_num = config['period']
		time_stage = self.get_stage(train_dict, valid_dict, test_dict, time_dict, config)
		stage_all_inter = [0] * (stage_num + 1 * 2)

		# get the inter_cnt per stage of data
		# item_inter: {item_0:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}, item_1:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}}
		for user in train_dict:
			for item in train_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (stage_num + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		for user in valid_dict:
			for item in valid_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (stage_num + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		for user in test_dict:
			for item in test_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				# print(time_stage[time])
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (stage_num + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		abs_item_inter = item_inter.copy()

		for item in item_inter:
			item_inter[item] = [a / b for a, b in zip(item_inter[item], stage_all_inter)]
		print(f'the interaction of different stages are as followed:')
		print(stage_all_inter)

		return pair_stage, item_inter, abs_item_inter
	
	def get_user_inter(self, train_dict, valid_dict, test_dict, time_dict, config):
		pair_stage = {}
		# item_inter = {item_id: list of # of interactions (popularity)}
		user_inter = {}
		stage_num = config['period']
		# time_stage = {time:stage}
		time_stage = self.get_stage(train_dict, valid_dict, test_dict, time_dict, config)
		# train/valid/test data
		stage_all_inter = [0] * (stage_num + 1 * 2)

		# get the inter_cnt per stage of data
		# item_inter: {item_0:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}, item_1:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}}
		for user in train_dict:
			if user not in user_inter:
				user_inter[user] = [0] * (stage_num + 1 * 2)
			for item in train_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				stage_all_inter[int(time_stage[time])] += 1
				user_inter[user][time_stage[time]] += 1

		for user in valid_dict:
			if user not in user_inter:
				user_inter[user] = [0] * (stage_num + 1 * 2)
			for item in valid_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				stage_all_inter[int(time_stage[time])] += 1
				user_inter[user][time_stage[time]] += 1

		for user in test_dict:
			if user not in user_inter:
				user_inter[user] = [0] * (stage_num + 1 * 2)
			for item in test_dict[user]:
				time = time_dict[user][item]
				pair_stage[(user,item)] = time_stage[time]
				stage_all_inter[int(time_stage[time])] += 1
				user_inter[user][time_stage[time]] += 1

		# copy item_inter to abs_item_inter
		abs_user_inter = user_inter.copy()

		for item in user_inter:
			user_inter[item] = [a / b for a, b in zip(user_inter[item], stage_all_inter)]
		logging.info(f'the user interaction of different stages are as followed:')
		logging.info(stage_all_inter)

		return pair_stage, user_inter, abs_user_inter
	
	# def get_user_group(self, train_dict, group_num):
	# 	item_dict = {}
	# 	for u in train_dict:
	# 		for i in train_dict[u]:
	# 			if i not in item_dict:
	# 				item_dict[i] = 1
	# 			else:
	# 			# print('ops')
	# 				item_dict[i] += 1   
	# 	item_list = item_dict.items()
	# 	pop_rate = 0.1
	# 	sorted_item_list = sorted(item_list, key=lambda x: x[1], reverse=True)
	# 	pop_tuples = sorted_item_list[: int(pop_rate * len(sorted_item_list))]
	# 	pop_items = [i[0] for i in pop_tuples] 

	# 	group = {}
	# 	user_pop = {}
	# 	for u in train_dict:
	# 		cnt = 0
	# 		for i in train_dict[u]:
	# 			if i in pop_items:
	# 				cnt += 1
	# 		user_pop[u] = cnt / len(train_dict[u])

	# 	user_pop = user_pop.items()
	# 	sort_user_pop = sorted(user_pop, key=lambda x: x[1], reverse=True)
	# 	sort_user_pops = [i[0] for i in sort_user_pop]
		
	# 	size = math.ceil(len(sort_user_pops) / group_num)
	# 	group_infos = [sort_user_pops[i:i + size] for i in range(0, len(sort_user_pops), size)]
		
	# 	for u in train_dict:
	# 		for cnt, group_info in enumerate(group_infos):
	# 			if u in group_info:
	# 				group[u] = cnt

	# 	group_inter_cnt = {}
	# 	for u in train_dict:
	# 		for i in train_dict[u]:
	# 			if group[u] not in group_inter_cnt:
	# 				group_inter_cnt[group[u]] = 1
	# 			else:
	# 				group_inter_cnt[group[u]] += 1

	# 	# for i in range(len(group_inter_cnt)):
	# 	# 	print(f'{group_inter_cnt[i]} interactions for Group {i}')
		
	# 	return group

	def get_stage(self, train_dict, valid_dict, test_dict, time_dict, config):
		time_list = []
		stage_num = config['period']
		time_stage = {}

		# get train data
		for user in train_dict:
			for item in train_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of train data
		time_duration = time_list[-1] - time_list[0]
		size = math.ceil(time_duration / stage_num) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size

		# get valid data
		time_list = []
		for user in valid_dict:
			for item in valid_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of valid data
		time_duration = time_list[-1] - time_list[0]
		size = math.ceil(time_duration / 1) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size + stage_num

		# get test data
		time_list = []
		for user in test_dict:
			for item in test_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of valid data
		time_duration = time_list[-1] - time_list[0]
		size = math.ceil(time_duration / 1) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size + stage_num + 1

		return time_stage


	@property
	def n_users(self):
		return self.n_user
	
	@property
	def m_items(self):
		return self.m_item
	
	@property
	def trainDataSize(self):
		return self.traindataSize
	

	@property
	def trainDict(self):
		return self.train_dict

	@property
	def validDict(self):
		return self.valid_dict
	
	@property
	def pairStage(self):
		return self.pair_stage
	
	@property
	def itemInter(self):
		return self.item_inter

	@property
	def testDict(self):
		return self.test_dict

	@property
	def allPos(self):
		return self._allPos
		
	def get_group(self, train_dict, group_num, traindataSize):
		item_dict = {}
		group = {}

		for u in train_dict:
			for i in train_dict[u]:
				if i not in item_dict:
					item_dict[i] = 1
				else:
				# print('ops')
					item_dict[i] += 1   
		item_list = item_dict.items()

		# get the sorted item by interaction
		sorted_item_list = sorted(item_list, key=lambda x: x[1], reverse=True)
		# print(sorted_item_list)
		
		group_index = 0
		cnt = 0
		size = math.ceil(traindataSize / group_num)
		for item in sorted_item_list:
			group[item[0]] = group_index
			cnt += item[1]
			if cnt >= size:
				cnt = 0
				group_index += 1

		group_inter_cnt = {}
		for u in train_dict:
			for i in train_dict[u]:
				if group[i] not in group_inter_cnt:
					group_inter_cnt[group[i]] = 1
				else:
					group_inter_cnt[group[i]] += 1

		for i in range(len(group_inter_cnt)):
			print(f'{group_inter_cnt[i]} interactions for Group {i}')
				   
		return group


	def _split_A_hat(self,A):
		A_fold = []
		fold_len = (self.n_users + self.m_items) // self.folds
		for i_fold in range(self.folds):
			start = i_fold*fold_len
			if i_fold == self.folds - 1:
				end = self.n_users + self.m_items
			else:
				end = (i_fold + 1) * fold_len
			A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
		return A_fold

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		row = torch.Tensor(coo.row).long()
		col = torch.Tensor(coo.col).long()
		index = torch.stack([row, col])
		data = torch.FloatTensor(coo.data)
		return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
		
	def getSparseGraph(self):
		print("loading adjacency matrix")
		if self.Graph is None:
			try:
				pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
				print("successfully loaded...")
				norm_adj = pre_adj_mat
			except :
				print("generating adjacency matrix")
				s = time()
				adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
				adj_mat = adj_mat.tolil()
				R = self.UserItemNet.tolil()
				adj_mat[:self.n_users, self.n_users:] = R
				adj_mat[self.n_users:, :self.n_users] = R.T
				adj_mat = adj_mat.todok()
				# adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
				
				rowsum = np.array(adj_mat.sum(axis=1))
				d_inv = np.power(rowsum, -0.5).flatten()
				d_inv[np.isinf(d_inv)] = 0.
				d_mat = sp.diags(d_inv)
				
				norm_adj = d_mat.dot(adj_mat)
				norm_adj = norm_adj.dot(d_mat)
				norm_adj = norm_adj.tocsr()
				end = time()
				print(f"costing {end-s}s, saved norm_mat...")
				sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

			if self.split == True:
				self.Graph = self._split_A_hat(norm_adj)
				print("done split matrix")
			else:
				self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
				self.Graph = self.Graph.coalesce().to(world.device)
				print("don't split the matrix")
		return self.Graph

	def getUserItemFeedback(self, users, items):
		"""
		users:
			shape [-1]
		items:
			shape [-1]
		return:
			feedback [-1]
		"""
		# print(self.UserItemNet[users, items])
		return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

	def getUserPosItems(self, users):
		posItems = []
		for user in users:
			posItems.append(self.UserItemNet[user].nonzero()[1])
		return posItems

	def getUserValidItems(self, users):
		validItems = []
		for user in users:
			if user in self.valid_dict:
				validItems.append(self.valid_dict[user])
		return validItems
	
	def get_pair_bpr(self):
		"""
		the original impliment of BPR Sampling in LightGCN
		:return:
			np.array
		"""
		user_num = self.traindataSize
		# print(user_num)
		users = np.random.randint(0, self.n_users, user_num)

		self.user = []
		self.posItem = []
		self.negItem = []
		a = 0
		cnt = 0
		# for i, user in enumerate(users):
		while True:
			for user in users:
				posForUser = self._allPos[user]
				# the users have no positive items -> unseen items
				if len(posForUser) == 0:
					a += 1
					continue
				cnt += 1
				posindex = np.random.randint(0, len(posForUser))
				positem = posForUser[posindex]
				while True:
					negitem = np.random.randint(0, self.m_items)
					if negitem in posForUser:
						continue
					else:
						break
				self.user.append(user)
				self.posItem.append(positem)
				self.negItem.append(negitem)
				if cnt == user_num:
					break
			if cnt == user_num:
				break

	def __getitem__(self, idx):
		
		#return self.user[idx], self.posItem[idx], self.negItem[idx], self.pair_stage[(self.user[idx], self.posItem[idx])], self.item_inter[self.posItem[idx]], self.item_inter[self.negItem[idx]]
		return self.user[idx], self.posItem[idx], self.negItem[idx], self.pair_stage[(self.user[idx], self.posItem[idx])], self.item_inter[self.posItem[idx]], self.item_inter[self.negItem[idx]], \
			self.local_item_inter[self.posItem[idx]], self.local_item_inter[self.negItem[idx]], self.user_inter[self.user[idx]], self.local_user_inter[self.user[idx]] 
	
	def __len__(self):
		return self.traindataSize