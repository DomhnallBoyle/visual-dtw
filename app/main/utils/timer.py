"""Timer utility.

Contains code for timing code blocks
"""
import os
import time
from contextlib import contextmanager

FILE_PATH = 'timer.txt'


class Timer:

    def __init__(self, debug=False, to_file=False):
        self.archive = dict()
        self.debug = debug
        self.to_file = to_file

        if self.to_file and os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)

    @contextmanager
    def time(self, message='Timer'):
        """Context Manager as a class, used to time blocks of code.

        Usage:
            with timer() as f:
                <!-- code to time here -->

        Args:
            message (str): to print out with the time

        Returns:
            None
        """
        start = time.time()
        yield
        end = time.time()

        elapsed_time = (end - start) * 1000

        archived_times = self.archive.get(message, [])
        archived_times.append(elapsed_time)
        self.archive[message] = archived_times

        if self.debug:
            print(f'{message} - elapsed time: {elapsed_time:.3f} ms',
                  flush=True)

    def analyse(self):
        print_str = ''
        for message, timings in self.archive.items():
            average_time_ms = sum(timings) / len(timings)
            average_time_s = average_time_ms / 1000
            print_str += f'{message} - {average_time_ms:.3f} ms - ' \
                         f'{average_time_s:.3f} s\n'
        print(print_str, flush=True)

        if self.to_file:
            with open(FILE_PATH, 'a') as f:
                f.write(f'{print_str}\n')


def timeit(f):
    """Simplified decorator for timing a function.

    Args:
        f (function): to be timed

    Returns:
        obj: result of running the function decorated with timeit
    """
    def _timeit(*args, **kwargs):
        with Timer(debug=True).time():
            return f(*args, **kwargs)

    return _timeit
