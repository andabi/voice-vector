# -*- coding: utf-8 -*-
# !/usr/bin/env python


from data_load import DataLoader
from model import Model
import tensorflow as tf
from hparam import Hparam
import argparse
from tqdm import tqdm
from tensorflow.contrib.tensorboard.plugins import projector
import os


def train():
    hp = Hparam.get_global_hparam()

    # Data loader
    arg_data_loader = {'data_path': hp.train.data_path, 'batch_size': hp.train.batch_size}
    arg_data_loader.update(hp.signal)
    data_loader = DataLoader(**arg_data_loader)

    # Model
    model = Model(data_loader, is_training=True, **hp.model)
    loss_op = model.loss()

    # Training
    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=hp.train.lr)
    train_op = optimizer.minimize(loss_op, global_step=global_step)

    # Summary
    tf.summary.scalar('train/loss', loss_op)
    summ_op = tf.summary.merge_all()

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
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in tqdm(range(global_step.eval() + 1, hp.train.num_steps + 1), leave=False, unit='step'):
            sess.run(train_op)

            # Write checkpoint files at every step
            summ, gs = sess.run([summ_op, global_step])

            if step % hp.train.save_per_step == 0:
                saver.save(sess, os.path.join(hp.logdir, "model.ckpt"), global_step=gs)

                # Write embeddings
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = model.y.name
                projector.visualize_embeddings(writer, config)

                # Write eval accuracy at every n step
                # with tf.Graph().as_default():
                # eval(logdir=logdir, queue=False, writer=writer)

            writer.add_summary(summ, global_step=gs)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    Hparam(args.case).set_as_global_hparam()

    train()

    print("Done")
