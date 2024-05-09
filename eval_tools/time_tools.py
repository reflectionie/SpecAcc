import time
class Timer:
    def __init__(self):
        self.time_elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.time_elapsed = time.time() - self.start