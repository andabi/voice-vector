# -*- coding: utf-8 -*-
# !/usr/bin/env python
import argparse
from tensorpack.dataflow.remote import send_dataflow_zmq
from data_load import DataLoader, AudioMeta
from hparam import hparam as hp
import multiprocessing


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name.')
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-dest_url', type=str)
    parser.add_argument('-num_thread', type=int, default=1)
    args = parser.parse_args()

    # set hyper-parameters from yaml file
    hp.set_hparam_yaml(case=args.case)

    if args.data_path:
        hp.train.data_path = args.data_path

    # dataflow
    audio_meta = AudioMeta(hp.train.data_path)
    data_loader = DataLoader(audio_meta, 1)
    num_thread = args.num_thread if args.num_thread else multiprocessing.cpu_count() // 1.5
    data_loader = data_loader.dataflow(nr_prefetch=5000, nr_thread=args.num_thread)

    send_dataflow_zmq(data_loader, args.dest_url)