# -*- coding: utf-8 -*-
# !/usr/bin/env python


import argparse

import tensorflow as tf
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

from data_load import DataLoader, AudioMeta
from hparam import hparam as hp
from model import ClassificationModel


def compute_accuracy(model, mel_spec, speaker_id, ckpt=None):
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=SaverRestore(ckpt) if ckpt else None)
    accuracy_pred = OfflinePredictor(pred_conf)

    acc, = accuracy_pred(mel_spec, speaker_id)

    return acc


def get_eval_input_names():
    return ['x', 'speaker_id']


def get_eval_output_names():
    return ['accuracy']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    args = parser.parse_args()

    hp.set_hparam_yaml(args.case)

    audio_meta = AudioMeta(hp.eval.data_path)
    data_loader = DataLoader(audio_meta, hp.eval.batch_size)

    # samples
    _, mel_spec, speaker_id = data_loader.dataflow().get_data().next()

    model = ClassificationModel(num_classes=audio_meta.num_speaker, **hp.model)

    ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)

    acc = compute_accuracy(model, mel_spec, speaker_id, ckpt)

    writer = tf.summary.FileWriter(hp.logdir)
    with tf.Session() as sess:
        summ = sess.run(tf.summary.scalar('eval/accuracy', acc))
        writer.add_summary(summ)
    writer.close()

