#! /usr/bin/env python3

from pytocl.main import main
from neat_driver import NeatDriver

if __name__ == '__main__':
    main(NeatDriver(logdata=False))
