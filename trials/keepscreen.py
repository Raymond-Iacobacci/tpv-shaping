import time
from pynput.mouse import Controller

mouse = Controller()
while True:
    mouse.move(1, 0)
    mouse.move(-1, 0)
    time.sleep(20)
