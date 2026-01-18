import pygame

def init():
    pygame.init()
    window = pygame.display.set_mode((400, 400))

def get_key(key_name):
    key_press = False
    for event in pygame.event.get():
        pass
    key_input = pygame.key.get_pressed()
    my_key = getattr(pygame, f"K_{key_name}")

    if key_input[my_key]:
        key_press = True
    pygame.display.update()

    return key_press

def main():
    if get_key("LEFT"):
        print("Left key Pressed")
    elif get_key("RIGHT"):
        print("Right key Pressed")
    elif get_key("UP"):
        print("Up key Pressed")
    elif get_key("DOWN"):
        print("Down key Pressed")
    elif get_key("q"):
        return True

if __name__ == '__main__':
    init()
    while True:
        if main():
            break