#!/usr/bin/env python3

import time
from hailo_platform import Device

def _run_periodic(delay=1):
    target = Device()
    try:
        while True:
            temp = target.control.get_chip_temperature().ts0_temperature
            print(f'{temp:.2f} C', end='\r')
            time.sleep(delay)
    except KeyboardInterrupt:
        print('-I- Received keyboard intterupt, exiting')

if __name__ == "__main__":
    _run_periodic()
