import pygame

from ..utils.assets import load_car_images

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
    images = load_car_images()
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

