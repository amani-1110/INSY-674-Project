import pygame
from track import Track
from simulator import reset_simulation, simulate_step, render, close_renderer, set_car

def scale_car(image, max_width, max_height):
    width, height = image.get_size()

    scale = min(max_width / width, max_height / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    return pygame.transform.smoothscale(
        image, (new_width, new_height)
    )

def choose_car(screen):
    """Show the four car options; return the selected zero-based index."""
    from pathlib import Path
    images = [pygame.image.load(str(Path(__file__).parent / 'assets' / f'car_{i}.png')).convert_alpha() for i in range(1, 5)]
    font = pygame.font.SysFont('arial', 30, bold=True)
    small = pygame.font.SysFont('arial', 20)
    clock = pygame.time.Clock()
    boxes = []
    for i in range(4):
        boxes.append(pygame.Rect(100 + i * 250, 230, 200, 350))
    while True:
        screen.fill((24, 35, 49))
        title = font.render('CHOOSE YOUR F1 CAR', True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(550, 90)))
        hint = small.render('Click a car or press 1, 2, 3, or 4', True, (210, 220, 235))
        screen.blit(hint, hint.get_rect(center=(550, 145)))
        for i, rect in enumerate(boxes):
            pygame.draw.rect(screen, (48, 63, 82), rect, border_radius=16)
            sprite = scale_car(images[i], 160, 310)
            screen.blit(sprite, sprite.get_rect(center=rect.center))
            label = small.render(str(i + 1), True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=(rect.centerx, rect.bottom + 30)))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return None
            if event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                return event.key - pygame.K_1
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(boxes):
                    if rect.collidepoint(event.pos):
                        return i
        clock.tick(30)


def main():
    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption('F1 AI Simulator - Car Selection')
    selection = choose_car(screen)
    if selection is None:
        pygame.quit()
        return
    set_car(selection)
    track = Track()
    state, info = reset_simulation(track)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(60) / 1000, .05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state, info = reset_simulation(track)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                selection = choose_car(screen)
                if selection is None:
                    running = False
                else:
                    set_car(selection)
                    state, info = reset_simulation(track)
        if not running:
            break
        keys = pygame.key.get_pressed()
        action = dict(
            steering=.28 * (int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])),
            throttle=float(keys[pygame.K_UP] or keys[pygame.K_w]),
            brake=float(keys[pygame.K_DOWN] or keys[pygame.K_s]),
        )
        state, info = simulate_step(state, action, dt, track)
        render(track, state)
    close_renderer()


if __name__ == '__main__':
    main()
