# -*- coding: utf-8 -*-
# !/usr/bin/env python

import argparse
import glob
import os
import sys
import threading
sys.path.append('.')
import librosa
from audio import trim_wav, write_wav


nthreads = 20


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target, args):
        super(StoppableThread, self).__init__(target=target, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ThreadsafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


class Audio:
    def __init__(self, src_dir, tar_dir):
        self.tar_dir = tar_dir
        self.src_dir = src_dir
        self.lists = ThreadsafeIter(glob.iglob('{}/*.wav'.format(src_dir)))

    def next(self):
        src_path = next(self.lists)
        if src_path is None:
            raise StopIteration()
        relpath = os.path.relpath(src_path, self.src_dir)
        base, _ = os.path.split(relpath)
        tar_base = os.path.join(self.tar_dir, base)
        if not os.path.exists(tar_base):
            os.mkdir(tar_base)
        tar_path = os.path.join(self.tar_dir, relpath)
        return src_path, tar_path


def do_task(nthreads, audio):
    print 'Thread-{} start.\n'.format(nthreads)
    try:
        while True:
            src_path, tar_path = audio.next()
            wav, sr = librosa.load(src_path)
            wav = trim_wav(wav)
            write_wav(wav, sr, tar_path)
    except StopIteration:
        print 'Thread-{} done.\n'.format(nthreads)


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, help='source directory path.')
    parser.add_argument('tar_dir', type=str, help='target directory path.')
    args = parser.parse_args()

    print str(args)

    if not os.path.exists(args.tar_dir):
        os.mkdir(args.tar_dir)

    audio = Audio(args.src_dir, args.tar_dir)

    threads = [StoppableThread(target=do_task, args=(i, audio))
               for i in range(nthreads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print('done.')