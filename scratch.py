from medium.environment import Environment


replay_file_path = r"C:\Users\b5050d\Workspace\reinforcement\replay\8.json"

import json

with open(replay_file_path, "r") as f:
    replay_dict = json.load(f)

game_thread = Environment(replay_dict)
game_thread.play(True)