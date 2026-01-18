# tello_pygame_text_cm.py
# Controls DJI Tello via djitellopy with a Pygame text input for cm-based moves.

import pygame
import sys
import time
from djitellopy import Tello

# ---------- Pygame UI helpers ----------
WIDTH, HEIGHT = 560, 240
BG = (18, 18, 18)
FG = (230, 230, 230)
ACCENT = (64, 160, 255)
ERROR = (255, 80, 80)
OK = (80, 220, 120)

def draw_text(surface, text, x, y, color=FG, size=22, bold=False):
    font = pygame.font.SysFont("consolas", size, bold=bold)
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tello CM Control (Pygame input)")
    clock = pygame.time.Clock()
    input_active = True
    input_text = ""
    status_lines = [
        "Connected? Take off once, then enter commands below.",
        "Format: direction distance_cm (e.g., forward 100)",
        "Commands: forward/back/left/right/up/down/cw/ccw, takeoff, land, battery",
    ]
    last_feedback = ""
    last_feedback_color = FG

    # ---------- Tello setup ----------
    tello = Tello()
    connected = False
    try:
        tello.connect()
        connected = True
        b = tello.get_battery()
        last_feedback = f"Battery: {b}%"
        last_feedback_color = OK
    except Exception as e:
        last_feedback = f"Connect error: {e}"
        last_feedback_color = ERROR

    # Do a single takeoff at start if connected (optional; can be removed)
    airborne = False
    if connected:
        try:
            tello.takeoff()
            airborne = True
            last_feedback = "Takeoff OK"
            last_feedback_color = OK
            time.sleep(0.8)
        except Exception as e:
            last_feedback = f"Takeoff error: {e}"
            last_feedback_color = ERROR

    # ---------- Command execution ----------
    def execute_command(cmd_line: str):
        nonlocal airborne, last_feedback, last_feedback_color
        s = cmd_line.strip()
        if not s:
            last_feedback, last_feedback_color = "No input", ERROR
            return

        parts = s.lower().split()

        try:
            # meta commands
            if parts[0] in ("battery", "bat"):
                b = tello.get_battery()
                last_feedback, last_feedback_color = f"Battery: {b}%", OK
                return

            if parts[0] in ("takeoff", "to"):
                tello.takeoff()
                airborne = True
                last_feedback, last_feedback_color = "Takeoff OK", OK
                return

            if parts[0] in ("land", "ld"):
                tello.land()
                airborne = False
                last_feedback, last_feedback_color = "Landing OK", OK
                return

            if parts[0] == "stop":
                # stop all rc velocities (useful only if using rc control)
                tello.send_rc_control(0, 0, 0, 0)
                last_feedback, last_feedback_color = "RC stop", OK
                return

            # movement in cm
            if parts[0] in ("forward", "back", "left", "right", "up", "down"):
                if len(parts) != 2 or not parts[1].isdigit():
                    last_feedback, last_feedback_color = "Usage: direction distance_cm (e.g. forward 100)", ERROR
                    return
                dist = int(parts[1])
                if dist < 20 or dist > 500:
                    last_feedback, last_feedback_color = "Distance must be 20-500 cm", ERROR
                    return
                if not airborne:
                    last_feedback, last_feedback_color = "Drone not airborne (use takeoff)", ERROR
                    return

                if parts[0] == "forward":
                    tello.move_forward(dist)
                elif parts[0] == "back":
                    tello.move_back(dist)
                elif parts[0] == "left":
                    tello.move_left(dist)
                elif parts[0] == "right":
                    tello.move_right(dist)
                elif parts[0] == "up":
                    tello.move_up(dist)
                elif parts[0] == "down":
                    tello.move_down(dist)

                last_feedback, last_feedback_color = f"{parts[0]} {dist} cm OK", OK
                time.sleep(0.4)
                return

            # rotation in degrees
            if parts[0] in ("cw", "ccw"):
                if len(parts) != 2 or not parts[1].isdigit():
                    last_feedback, last_feedback_color = "Usage: cw|ccw degrees (e.g. cw 90)", ERROR
                    return
                deg = int(parts[1])
                # djitellopy allows up to 3600 in some builds; here accept 1-3600
                if deg < 1 or deg > 3600:
                    last_feedback, last_feedback_color = "Degrees 1-3600", ERROR
                    return
                if not airborne:
                    last_feedback, last_feedback_color = "Drone not airborne (use takeoff)", ERROR
                    return

                if parts[0] == "cw":
                    tello.rotate_clockwise(deg)
                else:
                    tello.rotate_counter_clockwise(deg)

                last_feedback, last_feedback_color = f"{parts[0]} {deg}° OK", OK
                time.sleep(0.4)
                return

            last_feedback, last_feedback_color = "Unknown command", ERROR

        except Exception as e:
            last_feedback, last_feedback_color = f"Error: {e}", ERROR

    # ---------- Main loop ----------
    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and input_active:
                if event.key == pygame.K_RETURN:
                    if input_text.strip():
                        execute_command(input_text)
                        input_text = ""
                    else:
                        last_feedback, last_feedback_color = "Enter a command", ERROR
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    # Append typed character; event.unicode holds the character
                    if len(input_text) < 40:
                        input_text += event.unicode

        # Draw UI
        screen.fill(BG)
        y = 16
        for ln in status_lines:
            draw_text(screen, ln, 16, y, FG, 20)
            y += 24

        # Input label and box
        draw_text(screen, "Input:", 16, y + 10, ACCENT, 22, True)
        pygame.draw.rect(screen, (40, 40, 40), (100, y, WIDTH - 120, 40), border_radius=6)
        pygame.draw.rect(screen, ACCENT, (100, y, WIDTH - 120, 40), width=2, border_radius=6)
        draw_text(screen, input_text, 112, y + 8, FG, 22)
        y += 60

        # Feedback line
        draw_text(screen, f"Status: {last_feedback}", 16, y, last_feedback_color, 20)

        pygame.display.flip()

    # Cleanup
    try:
        if connected and airborne:
            tello.land()
    except:
        pass
    try:
        tello.end()
    except:
        pass
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
