import os

currdir = os.path.dirname(__file__)

DATABASE_PATH = os.path.join(currdir, "src", "reinforcement_app", "storage", "main.db")

REPLAY_PATH = os.path.join(currdir, "src", "reinforcement_app", "storage", "replay")
MODEL_PATH = os.path.join(currdir, "src", "reinforcement_app", "storage", "model")
CONFIG_PATH = os.path.join(currdir, "src", "reinforcement_app", "storage", "config")

if __name__ == "__main__":
    assert os.path.exists(DATABASE_PATH)
    assert os.path.exists(REPLAY_PATH)
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(CONFIG_PATH)
