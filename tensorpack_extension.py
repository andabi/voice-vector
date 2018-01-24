# -*- coding: utf-8 -*-
#!/usr/bin/env python


import tensorflow as tf
try:
    from tensorflow.python.ops.data_flow_ops import StagingArea
except ImportError:
    pass

from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput, EnqueueThread


class FlexibleQueueInput(QueueInput):
    """ Extend QueueInput to set queue capacity.
    """

    def __init__(self, ds, capacity=1000):
        """
        Args:
            ds(DataFlow): the input DataFlow.
            queue (tf.QueueBase): A :class:`tf.QueueBase` whose type
                should match the corresponding InputDesc of the model.
                Defaults to a FIFO queue of size 50.
        """
        super(FlexibleQueueInput, self).__init__(ds)
        self.capacity = capacity

    def _setup(self, inputs):
        self._input_placehdrs = [v.build_placeholder_reuse() for v in inputs]
        assert len(self._input_placehdrs) > 0, \
            "QueueInput has to be used with some inputs!"
        with self.cached_name_scope():
            if self.queue is None:
                self.queue = tf.FIFOQueue(
                    self.capacity, [x.dtype for x in self._input_placehdrs],
                    name='input_queue')
            logger.info("Setting up the queue '{}' for CPU prefetching ...".format(self.queue.name))
            self.thread = EnqueueThread(self.queue, self._inf_ds, self._input_placehdrs)

            self._dequeue_op = self.queue.dequeue(name='dequeue_for_reset')