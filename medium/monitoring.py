"""
Contains some methods for monitoring purposes
"""

import os
import pickle
import torch


def save_ai_run(foods, actions, run_iter, run_tag):
    """
    Save an AI run to the archive
    """
    assert type(foods) is dict
    assert type(actions) is list
    assert type(actions[0]) is int

    run = {}
    run["foods"] = foods
    run["actions"] = actions

    run_iter = f"{run_iter:04d}"

    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{run_tag}_{run_iter}.pkl")

    with open(filepath, "wb") as f:
        pickle.dump(run, f)
    assert os.path.exists(filepath)


def save_ai_model(model, model_tag, model_num):
    """
    Save the AI model for future re-use
    """
    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{model_tag}_{model_num:04d}.pth")
    torch.save(model.state_dict(), filepath)


def load_ai_model(model_class, filepath):
    """
    Load an existing AI model into evaluation mode to see how it does
    """
    assert os.path.exists(filepath)
    model = model_class
    model.load_state_dict(torch.load(filepath))
    model.eval


def load_ai_run(run_iter, run_tag):
    """
    Load an AI run to see how it has done
    """
    run_iter = f"{run_iter:04d}"
    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{run_tag}_{run_iter}.pkl")
    with open(filepath, "rb") as f:
        ans = pickle.load(f)
    return ans


