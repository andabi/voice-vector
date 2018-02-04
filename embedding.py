# -*- coding: utf-8 -*-
# !/usr/bin/env python


from data_load import DataLoader, AudioMeta
from model import ClassificationModel
import tensorflow as tf
from hparam import hparam as hp
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from prepro import read_wav, fix_length
from tensorflow.contrib.tensorboard.plugins import projector


def plot_embedding(embedding, annotation, filename='outputs/embedding.png'):
    reduced = TSNE(n_components=2).fit_transform(embedding)
    plt.figure(figsize=(20, 20))
    max_x = np.amax(reduced, axis=0)[0]
    max_y = np.amax(reduced, axis=0)[1]
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))

    plt.scatter(reduced[:, 0], reduced[:, 1], s=20, c=["r"] + ["b"] * (len(reduced) - 1))

    # Annotation
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
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    args = parser.parse_args()

    hp.set_hparam_yaml(args.case)

    # model
    audio_meta_train = AudioMeta(hp.train.data_path)
    model = ClassificationModel(num_classes=audio_meta_train.num_speaker, **hp.model)

    # data loader
    audio_meta = AudioMeta(hp.embed.data_path)
    data_loader = DataLoader(audio_meta, hp.embed.batch_size)

    # samples
    wav, mel_spec, speaker_id = data_loader.dataflow().get_data().next()

    ckpt = args.ckpt if args.ckpt else tf.train.latest_checkpoint(hp.logdir)

    pred_conf = PredictConfig(
        model=model,
        input_names=['x'],
        output_names=['embedding/embedding', 'prediction'],
        session_init=SaverRestore(ckpt) if ckpt else None)
    embedding_pred = OfflinePredictor(pred_conf)

    embedding, similar_speaker_id = embedding_pred(mel_spec)

    # get a random audio of the predicted speaker.
    wavfile_similar_speaker = np.array(map(lambda s: audio_meta_train.get_random_audio(s), similar_speaker_id))
    length = int(hp.signal.duration * hp.signal.sr)
    wav_similar_speaker = np.array(
        map(lambda w: fix_length(read_wav(w, hp.signal.sr, duration=hp.signal.duration), length),
            wavfile_similar_speaker))

    # write audio
    tf.summary.audio('wav', wav, hp.signal.sr, max_outputs=10)
    tf.summary.audio('wav_most_similar', wav_similar_speaker, hp.signal.sr, max_outputs=10)

    # write prediction
    speaker_name = [audio_meta.get_speaker_dict()[sid] for sid in speaker_id]
    similar_speaker_name = [audio_meta_train.get_speaker_dict()[sid] for sid in similar_speaker_id]
    prediction = ['{} -> {}'.format(s, p) for s, p in zip(speaker_name, similar_speaker_name)]
    tf.summary.text('prediction', tf.convert_to_tensor(prediction))

    writer = tf.summary.FileWriter(hp.logdir)

    # t-SNE
    # speaker_name = map(lambda i: audio_meta.speaker_dict[i], speaker_id)
    plot_embedding(embedding, speaker_id, filename='outputs/embedding-{}.png'.format(hp.case))

    ## TODO Write embeddings to tensorboard
    # config = projector.ProjectorConfig()
    # embedding_conf = config.embeddings.add()
    # embedding_conf.tensor_name = 'embedding/embedding'
    # projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        writer.add_summary(sess.run(tf.summary.merge_all()))

    writer.close()

    print "done"