# -*- coding: utf-8 -*-
# !/usr/bin/env python
import argparse
import multiprocessing

from tensorpack.dataflow.remote import send_dataflow_zmq
from data_load import DataLoader, AudioMeta
from hparam import hparam as hp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='start training from the beginning.')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('--ckpt', help='checkpoint to load model.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--r', action='store_true', help='start training from the beginning.')
    parser.add_argument('--dest_url', type=str)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--nb_proc', type=int, default=1)

    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    # dataflow
    if args.phase == 'train':
        audio_meta = AudioMeta(hp.train.data_path)
        data_loader = DataLoader(audio_meta, 1)
    else:
        audio_meta = AudioMeta(hp.embed.data_path)
        data_loader = DataLoader(audio_meta, 1)
    data_loader = data_loader.dataflow(nr_prefetch=5000, nr_thread=args.nb_proc)

    send_dataflow_zmq(data_loader, args.dest_url)
