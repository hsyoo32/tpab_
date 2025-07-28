import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--batch_size', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=float,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="")
    
    parser.add_argument('--layer_p', type=int,default=0,
                        help="the layer num of lightGCN for popularity")
    parser.add_argument('--dropout_p', type=float,default=0,
                        help="using the dropout or not for popularity")
    parser.add_argument('--keepprob_p', type=float,default=0,
                        help="")

    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=2048,
                        help="the batch size of users for testing")
    parser.add_argument('--data_path', type=str, default='micro_video/',
                        help='the path to dataset')
    parser.add_argument('--dataset', type=str,default='micro_video',
                        help="available datasets: ['micro_video', 'kuai', 'amazon_book']")
    parser.add_argument('--data_type', type=str, default='time',
                        help='time or random')
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20, 50, 100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--period', type=int, default=8, help='number of stage')
    parser.add_argument('--log', type=str, help='the name of your log')
    parser.add_argument('--predict', type=float, default=0.2)


    parser.add_argument('--algo', type=str, default='vanilla', help='rec_algo, support []')
    parser.add_argument('--log_file', type=str,default="log.txt", help="path of the log file.")

    # TPAB
    parser.add_argument('--n_pop_group', type=int, default=10, help='number of popularity groups')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='weight for bootstrapping loss')

    return parser.parse_args()