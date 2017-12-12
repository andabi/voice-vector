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
from utils import remove_all_files


def train():
    hp = Hparam.get_global_hparam()

    # Data loader
    arg_data_loader = {'data_path': hp.train.data_path, 'batch_size': hp.train.batch_size}
    arg_data_loader.update(hp.signal)
    data_loader = DataLoader(**arg_data_loader)

    # Model
    model = Model(data_loader, is_training=True, **hp.model)
    loss_op, sim_pos, sim_neg = model.loss(hp.train.margin)

    # Training
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=hp.train.lr)

    # Gradient clipping to prevent loss explosion
    gvs = optimizer.compute_gradients(loss_op)
    gvs = [(tf.clip_by_value(grad, hp.train.clip_value_min, hp.train.clip_value_max), var) for grad, var in gvs]
    gvs = [(tf.clip_by_norm(grad, hp.train.clip_norm), var) for grad, var in gvs]

    train_op = optimizer.apply_gradients(gvs, global_step=global_step)

    # Summary
    tf.summary.scalar('train/loss', loss_op)
    # tf.summary.histogram('x', model.x)
    # tf.summary.histogram('x_pos', model.x_pos)
    # tf.summary.histogram('x_neg', model.x_neg)
    # tf.summary.histogram('y', model.y)
    # tf.summary.histogram('y_pos', model.y_pos)
    # tf.summary.histogram('y_neg', model.y_neg)
    # tf.summary.histogram('sim/pos', sim_pos)
    # tf.summary.histogram('sim/neg', sim_neg)
    # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net'):
    #     tf.summary.histogram(v.name, v)

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
                saver.save(sess, os.path.join(hp.logdir, hp.train.ckpt_prefix), global_step=gs)

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
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('-r', action='store_true', help='start training from the beginning.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    # Set hyper-parameters globally
    hp = Hparam(args.case).set_as_global_hparam()

    if args.r:
        ckpt = '{}/checkpoint'.format(os.path.join(hp.logdir))
        if os.path.exists(ckpt):
            os.remove(ckpt)
            remove_all_files(os.path.join(hp.logdir, 'events.out'))
            remove_all_files(os.path.join(hp.logdir, hp.train.ckpt_prefix))

    train()

    print("Done")
