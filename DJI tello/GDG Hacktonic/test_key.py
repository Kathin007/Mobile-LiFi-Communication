import KeyPressModule as kpm
import time
import sys
sys.stdout.reconfigure(encoding="utf-8")
kpm.init()
print("Press ARROW keys, SPACE, W, A, S, D; press ESC to quit.")

while True:
    if kpm.get_key("LEFT"):
        print("← LEFT arrow detected")
    if kpm.get_key("RIGHT"):
        print("→ RIGHT arrow detected")
    if kpm.get_key("UP"):
        print("↑ UP arrow detected")
    if kpm.get_key("DOWN"):
        print("↓ DOWN arrow detected")

    if kpm.get_key("SPACE"):
        print("SPACE detected")
    if kpm.get_key("w"):
        print("W detected")
    if kpm.get_key("a"):
        print("A detected")
    if kpm.get_key("s"):
        print("S detected")
    if kpm.get_key("d"):
        print("D detected")

    if kpm.get_key("ESCAPE"):
        print("ESC pressed – exiting test")
        break

    time.sleep(0.05)
