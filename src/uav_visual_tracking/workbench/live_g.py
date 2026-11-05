"""Run the sealed U3-G broker on Linux."""
import os
import signal
import sys
from threading import Event, Thread

from . import broker
from .joint import Planner


def main():
    cancelled = Event()

    def cancel(*_):
        cancelled.set()

    def watch_parent():
        while os.read(0, 4096):
            pass
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, cancel)
    Thread(target=watch_parent, daemon=True).start()
    broker.run(sys.argv[1], Planner, cancelled)


if __name__ == "__main__":
    main()
