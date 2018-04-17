# -*- coding: utf-8 -*-
# !/usr/bin/env python


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore

from data_load import *
from hparam import hparam as hp
from model import ClassificationModel
from audio import read_wav, fix_length


def plot_embedding(embedding, annotation=None, filename='outputs/embedding.png'):
    reduced = TSNE(n_components=2).fit_transform(embedding)
    plt.figure(figsize=(20, 20))
    max_x = np.amax(reduced, axis=0)[0]
    max_y = np.amax(reduced, axis=0)[1]
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))

    plt.scatter(reduced[:, 0], reduced[:, 1], s=20, c=["r"] + ["b"] * (len(reduced) - 1))

    # Annotation
    if annotation:
        for i in range(embedding.shape[0]):
            target = annotation[i]
            x = reduced[i, 0]
            y = reduced[i, 1]
            plt.annotate(target, (x, y))

    plt.savefig(filename)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    args = parser.parse_args()

    hp.set_hparam_yaml(args.case)

    # model
    audio_meta_train = VoxCelebMeta(hp.train.data_path, hp.train.meta_path)
    model = ClassificationModel(num_classes=audio_meta_train.num_speaker, **hp.model)

    # data loader
    audio_meta_class = globals()[hp.embed.audio_meta_class]
    params = {'data_path': hp.embed.data_path}
    if hp.embed.meta_path:
            params['meta_path'] = hp.embed.meta_path
    audio_meta = audio_meta_class(**params)
    data_loader = DataLoader(audio_meta, hp.embed.batch_size)

    # samples
    wav, mel_spec, speaker_id = data_loader.dataflow().get_data().next()

    ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)

    pred_conf = PredictConfig(
        model=model,
        input_names=['x'],
        output_names=['embedding/embedding', 'prediction'],
        session_init=SaverRestore(ckpt) if ckpt else None,
    )

    embedding_pred = OfflinePredictor(pred_conf)

    embedding, pred_speaker_id = embedding_pred(mel_spec)

    # get a random audio of the predicted speaker.
    wavfile_pred_speaker = np.array(map(lambda s: audio_meta_train.get_random_audio(s), pred_speaker_id))
    length = int(hp.signal.duration * hp.signal.sr)
    wav_pred_speaker = np.array(
        map(lambda w: fix_length(read_wav(w, hp.signal.sr, duration=hp.signal.duration), length),
            wavfile_pred_speaker))

    # write audio
    tf.summary.audio('wav', wav, hp.signal.sr, max_outputs=10)
    tf.summary.audio('wav_pred', wav_pred_speaker, hp.signal.sr, max_outputs=10)

    # write prediction
    speaker_name = [audio_meta.speaker_dict[sid] for sid in speaker_id]
    pred_speaker_name = [audio_meta_train.speaker_dict[sid] for sid in pred_speaker_id]

    meta = [tuple(audio_meta.meta_dict[sid][k] for k in audio_meta.target_meta_field()) for sid in speaker_id]
    pred_meta = [tuple(audio_meta_train.meta_dict[sid][k] for k in audio_meta_train.target_meta_field()) for sid in pred_speaker_id]
    prediction = ['{} ({}) -> {} ({})'.format(s, s_meta, p, p_meta)
                  for s, p, s_meta, p_meta in zip(speaker_name, pred_speaker_name, meta, pred_meta)]
    tf.summary.text('prediction', tf.convert_to_tensor(prediction))

    writer = tf.summary.FileWriter(hp.logdir)

    # visualization of embedding (t-SNE)
    if hp.embed.meta_field_viz:
        annotation = [audio_meta.meta_dict[sid][hp.embed.meta_field_viz] for sid in speaker_id]
    else:
        # annotation = meta if meta else speaker_name
        annotation = None
    plot_embedding(embedding, annotation, filename='outputs/embedding-{}.png'.format(hp.case))

    # TODO Write embeddings to tensorboard
    # config = projector.ProjectorConfig()
    # embedding_conf = config.embeddings.add()
    # embedding_conf.tensor_name = 'embedding/embedding'
    # projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        writer.add_summary(sess.run(tf.summary.merge_all()))

    writer.close()

    print "done"