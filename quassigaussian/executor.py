import logging
import psutil
from typing import Union, Generator, Callable
import os

import numpy as np
import pandas as pd


class Executor:
    """
    Executor for multiprocessing
    """
    _wait_function: Callable
    MIN_MEMORY_PER_WORKER = 4  # in GB - standard is 3
    MIN_SYSTEM_MEMORY = 4  # in GB
    RESERVED_CPUS = 0  # for Postgres DB, etc.

    def __init__(self, use_dask: bool = True, logfile='main.log', nr_processes=None):

        self.executor = None
        self.lock = None
        self.logfile = logfile

        self.dashboard_address = None
        self.use_dask = use_dask
        self._infer_number_worker()

        if nr_processes:
           self.nr_processes = nr_processes

        self._start_concurrent_futures_exectutor()

    def _infer_number_worker(self):
        self.cpus = psutil.cpu_count(logical=False)
        self.total_memory = round(psutil.virtual_memory().total / 1024**3)
        worker_memory_limit = (self.total_memory - self.MIN_SYSTEM_MEMORY) // self.MIN_MEMORY_PER_WORKER
        self.nr_processes = max(min(self.cpus - self.RESERVED_CPUS, worker_memory_limit), 1)

    def _start_concurrent_futures_exectutor(self):
        import concurrent.futures
        self._wait_function = concurrent.futures.wait
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.nr_processes)



    def submit(self, *args, **kwargs):
        __doc__ = self.executor.submit.__doc__
        return self.executor.submit(*args, **kwargs)

    def map(self, *args, **kwargs):
        __doc__ = self.executor.map.__doc__
        return self.executor.map(*args, **kwargs)

    def await_futures(self, futures):
        """
        waits until all async db write tasks are finished

        :return:
        """
        if len(futures) == 0:
            return

        self._wait_function(futures)
        self._check_futures_exception(futures)

    def shutdown(self):
        """
        Shuts down multiprocessing executor
        Make sure that all futures are finished before (using await_futures)!
        """
        try:
            self.executor.close()
        except AttributeError:
            pass

    def scatter(self, item):
        """
        Submits large data items to calculator and returns a future
        Only required for Dask, for concurrent.futures it returns the original item

        :param item:
        :return:
        """
        if not self.use_dask:
            return item
        else:
            return self.executor.scatter(item)

    def acquire_lock(self):
        if self.lock is not None and self.executor is not None:
            assert self.lock.acquire()

    def release_lock(self):
        if self.lock is not None:
            self.lock.release()

    def __del__(self):
        self.shutdown()

    @staticmethod
    def _check_futures_exception(futures):
        """
        check for exceptions

        :param futures:
        """
        is_error = False
        process_errors = []
        for future in futures:
            process_error = future.exception()
            if process_error is not None:
                process_errors.append(process_error)
                is_error = True
