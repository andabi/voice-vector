# -*- coding: utf-8 -*-
# !/usr/bin/env python


import argparse
import multiprocessing
import os

import tensorflow as tf
from tensorpack.callbacks.base import Callback
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.common import TestDataSpeed
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.dataflow.remote import RemoteDataZMQ
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
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
        self.data_loader = DataLoader(audio_meta, hp.eval.batch_size).dataflow()

    def _trigger_epoch(self):
        _, mel_spec, speaker_id = next(self.data_loader.get_data())
        acc, = self.pred(mel_spec, speaker_id)
        self.trainer.monitors.put_scalar('eval/accuracy', acc)


def get_remote_dataflow(port, nr_prefetch=1000, nr_thread=1):
    ipc = 'ipc:///tmp/ipc-socket'
    tcp = 'tcp://0.0.0.0:%d' % port
    data_loader = RemoteDataZMQ(ipc, tcp, hwm=50000)
    data_loader = BatchData(data_loader, batch_size=hp.train.batch_size)
    data_loader = PrefetchData(data_loader, nr_prefetch, nr_thread)
    return data_loader


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--r', action='store_true', help='start training from the beginning.')
    parser.add_argument('--remote', action='store_true', help='use remote dataflow.')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    # dataflow
    audio_meta = AudioMeta(hp.train.data_path)
    if args.remote:
        df = get_remote_dataflow(args.port, hp.train.batch_size)
    else:
        df = DataLoader(audio_meta, hp.train.batch_size).dataflow(nr_prefetch=5000, nr_thread=int(multiprocessing.cpu_count() // 1.5))

    # set logger for event and model saver
    logger.set_logger_dir(hp.logdir)
    if True:
        # set train config
        train_conf = TrainConfig(
            model=ClassificationModel(num_classes=audio_meta.num_speaker, **hp.model),
            data=FlexibleQueueInput(df, capacity=3000),
            callbacks=[
                ModelSaver(checkpoint_dir=hp.logdir),
                EvalCallback()
            ],
            steps_per_epoch=hp.train.steps_per_epoch
        )

        ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)
        if ckpt and not args.r:
            train_conf.session_init = SaverRestore(ckpt)

        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            train_conf.nr_tower = len(args.gpu.split(','))

        trainer = SyncMultiGPUTrainerReplicated(hp.train.num_gpu)

        launch_train_with_config(train_conf, trainer=trainer)
    else:
        df = TestDataSpeed(df, 100000)
        for _ in df.get_data():
            pass