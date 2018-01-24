# -*- coding: utf-8 -*-
# !/usr/bin/env python


import argparse
import os

import tensorflow as tf
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig, SimpleTrainer
from tensorpack.train.interface import launch_train_with_config
from tensorpack.utils import logger

from data_load_tensorpack import DataLoader, AudioMeta
from hparam import hparam as hp
from model_tensorpack import Model
from tensorpack_extension import FlexibleQueueInput

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--r', action='store_true', help='start training from the beginning.')
    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    # dataflow
    audio_meta = AudioMeta(hp.train.data_path)
    data_loader = DataLoader(audio_meta, hp.train.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)

    # set train config
    train_config = TrainConfig(
        model=Model(**hp.model),
        data=FlexibleQueueInput(data_loader.dataflow(nr_prefetch=5000), capacity=1000),
        callbacks=[
            ModelSaver(checkpoint_dir=hp.logdir),
            # TODO inference in training
            # InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
    )

    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt and not args.r:
        train_config.session_init = SaverRestore(ckpt)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_config.nr_tower = len(args.gpu.split(','))

    launch_train_with_config(train_config, SimpleTrainer())