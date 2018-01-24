# -*- coding: utf-8 -*-
# !/usr/bin/env python


from data_load_tensorpack import DataLoader
from model_classification import Model
import tensorflow as tf
from hparam import Hparam
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def compute_embed():
    hp = Hparam.get_global_hparam()

    # Data loader
    arg_data_loader = {'data_path': hp.embed.data_path, 'batch_size': hp.embed.batch_size}
    arg_data_loader.update(hp.signal)
    data_loader = DataLoader(**arg_data_loader)

    # Model
    model = Model(data_loader, is_training=False, **hp.model)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, logdir=hp.logdir)

        writer = tf.summary.FileWriter(hp.logdir)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        data_loader.queue.set_coordinator(tf.train.Coordinator())
        data_loader.queue.start()

        embedding, speaker_id = sess.run(model())

        # Write embeddings
        # config = projector.ProjectorConfig()
        # embedding_conf = config.embeddings.add()
        # embedding_conf.tensor_name = model.y.name
        # projector.visualize_embeddings(writer, config)

        writer.close()
        # coord.request_stop()
        # coord.join(threads)

    # speaker_name = data_loader.speaker_dict[speaker_id]

    return embedding, speaker_id


def plot_embedding(embedding, annotation):
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

    plt.savefig("outputs/embedding.png")
    # plt.show()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    # Set hyper-parameters globally
    hp = Hparam(args.case).set_as_global_hparam()

    embedding, speaker_id = compute_embed()
    plot_embedding(embedding, speaker_id)