# -*- coding: utf-8 -*-
# !/usr/bin/env python


import argparse
import multiprocessing
import os

import tensorflow as tf
from tensorpack.callbacks.base import Callback
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated, SimpleTrainer
from tensorpack.utils import logger

from data_load import DataLoader, AudioMeta
from eval import get_eval_input_names, get_eval_output_names
from hparam import hparam as hp
from model import ClassificationModel
from tensorpack_extension import FlexibleQueueInput


class EvalCallback(Callback):
    def _setup_graph(self):

        self.pred = self.trainer.get_predictor(
            get_eval_input_names(),
            get_eval_output_names())

    def _before_train(self):
        self.audio_meta = AudioMeta(hp.embed.data_path)
        self.data_loader = DataLoader(audio_meta, hp.embed.batch_size)

    def _trigger_epoch(self):
        _, mel_spec, speaker_id = self.data_loader.dataflow().get_data().next()
        acc, = self.pred(mel_spec, speaker_id)
        self.trainer.monitors.put_scalar('eval/accuracy', acc)

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--r', action='store_true', help='start training from the beginning.')
    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    # dataflow
    audio_meta = AudioMeta(hp.train.data_path)
    data_loader = DataLoader(audio_meta, hp.train.batch_size)
    df = data_loader.dataflow(nr_prefetch=5000, nr_thread=int(multiprocessing.cpu_count() // 1.5))

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)

    # set train config
    train_conf = TrainConfig(
        model=ClassificationModel(num_classes=audio_meta.num_speaker, **hp.model),
        data=FlexibleQueueInput(df, capacity=3000),
        callbacks=[
            ModelSaver(checkpoint_dir=hp.logdir),
            EvalCallback()
        ],
        steps_per_epoch=100
    )

    ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)
    if ckpt and not args.r:
        train_conf.session_init = SaverRestore(ckpt)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    trainer = SyncMultiGPUTrainerReplicated(hp.train.num_gpu)
    # trainer = SimpleTrainer()

    launch_train_with_config(train_conf, trainer=trainer)

    # test_loader = TestDataSpeed(data_loader, 100000)
    # for _ in test_loader.get_data():
    #     pass
