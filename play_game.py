"""
Play the game as a user
"""

from reinforcement_app.simulation.environment import Environment


config = {
    "EXPERIMENT_DESCRIPTION": "default",
    "N_FOODS": 5,
    "N_DANGERS": 0,
    "RANDOM_SEED": 40,
    "ARENA_SIZE": 200,
    "PLAYER_SPEED": 1,
    "FPS": 120,
    "RENDER_MULT": 1,
    "PLAYER_RADIUS": 5,
    "DUMMY_VAR": 0,
}

game_thread = Environment(config)
game_thread.play()
