# -*- coding: utf-8 -*-
# !/usr/bin/env python


import argparse
import multiprocessing
import os

import tensorflow as tf
from tensorpack.dataflow.remote import RemoteDataZMQ
from tensorpack.callbacks.base import Callback
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig, SimpleTrainer
from tensorpack.train.interface import launch_train_with_config
from tensorpack.dataflow.common import BatchData
from tensorpack.utils import logger
from tensorpack import TestDataSpeed
from tensorpack.dataflow.prefetch import PrefetchData

from data_load import DataLoader, AudioMeta
from eval import get_eval_input_names, get_eval_output_names
from hparam import hparam as hp
from model import ClassificationModel
from tensorpack_extension import FlexibleQueueInput
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated


class EvalCallback(Callback):

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            get_eval_input_names(),
            get_eval_output_names())

    def _trigger_epoch(self):
        _, mel_spec, speaker_id = next(self.data_loader.get_data())
        acc, = self.pred(mel_spec, speaker_id)
        self.trainer.monitors.put_scalar('eval/accuracy', acc)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_remote_data_loader(port, batch_size):
    url = 'tcp://0.0.0.0:%d' % port
    data_loader = RemoteDataZMQ(url, hwm=10000)
    data_loader = BatchData(data_loader, batch_size=batch_size)
    data_loader = PrefetchData(data_loader, 2048, 16)
    return data_loader


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--r', action='store_true', help='start training from the beginning.')
    parser.add_argument('--use_feeder', type=str2bool, default='false')
    parser.add_argument('--train_port', type=int, default=0)
    parser.add_argument('--test_port', type=int, default=0)
    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    # dataflow
    audio_meta = AudioMeta(hp.train.data_path)
    if args.use_feeder:
        data_loader = get_remote_data_loader(args.train_port, hp.train.batch_size)
        test_loader = get_remote_data_loader(args.test_port, hp.embed.batch_size)
    else:
        data_loader = DataLoader(audio_meta, hp.train.batch_size).dataflow()
        test_loader = DataLoader(audio_meta, hp.eval.batch_size).dataflow()

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)
    if True:
        # set train config
        train_conf = TrainConfig(
            model=ClassificationModel(num_classes=audio_meta.num_speaker, **hp.model),
            dataflow=data_loader,
            callbacks=[
                ModelSaver(checkpoint_dir=hp.logdir),
                EvalCallback(test_loader)
            ],
            steps_per_epoch=100
        )

        ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)
        if ckpt and not args.r:
            train_conf.session_init = SaverRestore(ckpt)

        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            train_conf.nr_tower = len(args.gpu.split(','))

        trainer = SyncMultiGPUTrainerReplicated(1)

        launch_train_with_config(train_conf, trainer=trainer)
    else:
        test_loader = TestDataSpeed(data_loader, 100000)
        for _ in test_loader.get_data():
            pass
