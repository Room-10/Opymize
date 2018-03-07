
import signal

class GracefulInterruptHandler(object):
    """ Context manager for handling SIGINT (e.g. if the user presses Ctrl+C)

    Taken from https://gist.github.com/nonZero/2907502

    >>> with GracefulInterruptHandler() as h:
    >>>     for i in xrange(1000):
    >>>         print "..."
    >>>         time.sleep(1)
    >>>         if h.interrupted:
    >>>             print "interrupted!"
    >>>             time.sleep(2)
    >>>             break
    """
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)
        signal.signal(self.sig, self.handle)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def handle(self, signum, frame):
        self.release()
        self.interrupted = True

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True