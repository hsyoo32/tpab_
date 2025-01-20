import world
import dataloader
import model
import utils
from pprint import pprint
from parse import parse_args
import logging


logging.info('===========config================')
logging.info(world.config)
logging.info('cores for test: {}'.format(world.CORES))
logging.info('comment: {}'.format(world.comment))
logging.info('tensorboard: {}'.format(world.tensorboard))
logging.info('LOAD: {}'.format(world.LOAD))
logging.info('Weight path: {}'.format(world.PATH))
logging.info('Test Topks: {}'.format(world.topks))

logging.info('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'mf-tpab': model.PureMF_TPAB,
    'lgn-tpab': model.LightGCN_TPAB,
}