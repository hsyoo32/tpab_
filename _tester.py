import os

log_ = '250101'


decays = ['1e-3']
# Embedding dimension
redcims = [64]
# Random seeds
#seeds = [2018,2019,2020,2021,2022]
seeds = [2018]

# Datasets
#datasets = ['micro_video','kuairand','yelp_10years']
datasets = ['micro_video']

backbones = ['mf','lgn']
backbones = ['mf']
# backbones = ['lgn']

gpu = 0


# algos = ['vanilla'] # vanilla
# ablation study of tpab. three variants: 
# tpab-c: 'tpab' with k=-1, lambda=x
# tpab-i: 'tpab' with k=x, lambda=0
# tpab-t: 'tpab-global' with k=x, lambda=x

algos = ['tpab-global']
algos = ['tpab']


# Bootstrap parameter lambda
# lambda1 = [0.5,1.0,1.5]
lambda1 = [1.0]

# Coarsening parameter K; when -1, no coarsening
n_pop_groups = [-1,10,20,30,40]
n_pop_groups = [20]


for dataset in datasets:
    for seed in seeds:
        for backbone in backbones:
            for algo in algos:
                n_pop_groups_ = n_pop_groups
                lambda1_ = lambda1
                for n_pop_group in n_pop_groups_:
                    for recdim in redcims:
                        for decay in decays:
                            for lamb1 in lambda1_:

                                if 'tpab' in algo:
                                    log = './log/{}_{}_{}_{}_{}_{}npop_{}decay_{}lr_{}lamb1_{}dim.txt'.format(
                                    log_, seed, dataset, backbone, algo, n_pop_group, decay, 0.001, lamb1, recdim)
                                    a = 'python -u main.py --data_path={} --dataset={} --model={} --epochs=600 \
                                    --decay={} --lr={} --gpu={} --log={} --algo={} --log_file={} --recdim={} --n_pop_group={} \
                                    --lambda1={} --seed={}'.format(
                                            dataset, dataset, backbone, decay, 0.001, gpu, log_, algo, log, recdim, n_pop_group, 
                                            lamb1, seed)
                                elif 'vanilla' in algo:
                                    log = './log/{}_{}_{}_{}_{}_{}decay_{}lr_{}dim.txt'.format(
                                        log_, seed, dataset, backbone, algo, decay, 0.001, recdim)
                                    a = 'python -u main.py --data_path={} --dataset={} --model={} --epochs=600 \
                                    --decay={} --lr={} --gpu={} --log={} --algo={} --log_file={} --recdim={} --seed={}'.format(
                                            dataset, dataset, backbone, decay, 0.001, gpu, log_, algo, log, recdim, seed)

                                print(a)
                                os.system(a)
