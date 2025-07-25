import pygame
import time  # or use datetime if you prefer

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Clock Example")

font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill background
    screen.fill((30, 30, 30))

    # Get current time string (you can replace this with your own logic)
    time_str = time.strftime("%H:%M:%S")

    # Render text
    text_surf = font.render(time_str, True, (255, 255, 255))
    text_rect = text_surf.get_rect(topright=(790, 10))  # Adjust for padding

    # Draw to screen
    screen.blit(text_surf, text_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
