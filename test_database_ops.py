"""
Testing script for the database operations
"""


import os
from database_ops import *
import pytest
import torch.nn as nn


# Sample Model for testing the storage
class DQN(nn.Module):
    """
    Simple NN to train on our little reinforcement
    learning simulation
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.model(x)


def test_table_exists(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")

    assert not table_exists(db_path, "Experiment")

    add_experiment(db_path, "Test")

    assert table_exists(db_path, "Experiment")
    assert not table_exists(db_path, "Model")
    

def test_add_experiment(tmp_path):
    # Add an experiment to a non existing database
    db_path = os.path.join(tmp_path, "test1.db")

    assert not os.path.exists(db_path)
    ans = add_experiment(db_path, "Testing")
    assert ans == 1
    assert os.path.exists(db_path)

    # Check that the experiment is present in the table now
    ans = get_all_experiments(db_path)
    assert len(ans) == 1
    assert "Testing" == ans[0][1]

    # Add an experiment to an existing table
    ans = add_experiment(db_path, "Checking")
    assert ans == 2
    
    # Check that both experiments are now present
    ans = get_all_experiments(db_path)
    assert len(ans) == 2
    assert "Testing" == ans[0][1]
    assert "Checking" == ans[1][1]


def test_add_model(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    model_folder = os.path.join(tmp_path, "model")
    
    sample_model = DQN()
    
    # Add a model to a non existing database
    with pytest.raises(FileNotFoundError):
        add_model(db_path, 1, sample_model, model_folder)

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a model when the experiment does not exist
    with pytest.raises(Exception):
        add_model(db_path, 2, sample_model, model_folder)

    ans = get_all_models(db_path)
    assert ans == []

    # Add a model when there is no model table
    add_model(db_path, 1, sample_model, model_folder)

    ans = get_all_models(db_path)
    assert len(ans) == 1
    assert ans[0][1] == 1
    assert ans[0][0] == 1

    # Add a model to an existing table
    add_model(db_path, 1, sample_model, model_folder)

    ans = get_all_models(db_path)
    assert len(ans) == 2
    assert ans[1][1] == 1
    assert ans[1][0] == 2


def test_add_replay(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    replay_folder = os.path.join(tmp_path, "replay")

    sample_replay = {}
    sample_replay["test"] = 1
    
    # Add a replay to a non existing database
    with pytest.raises(FileNotFoundError):
        add_replay(db_path, 1, sample_replay, replay_folder)

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a replay when the experiment does not exist
    with pytest.raises(Exception):
        add_replay(db_path, 2, sample_replay, replay_folder)

    ans = get_all_replays(db_path)
    assert ans == []

    # Add a model when there is no model table
    add_replay(db_path, 1, sample_replay, replay_folder)

    ans = get_all_replays(db_path)
    assert len(ans) == 1
    assert ans[0][1] == 1
    assert ans[0][0] == 1

    # Add a model to an existing table
    add_replay(db_path, 1, sample_replay, replay_folder)

    ans = get_all_replays(db_path)
    assert len(ans) == 2
    assert ans[1][1] == 1
    assert ans[1][0] == 2


def test_add_training_run(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    replay_folder = os.path.join(tmp_path, "replay")
    model_folder = os.path.join(tmp_path, "model")

    # Add a run to a non existing database
    with pytest.raises(FileNotFoundError):
        add_training_run(db_path, 1, 0, .9, 17, None, None)

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a training when the experiment does not exist
    with pytest.raises(Exception):
        add_training_run(db_path, 2, 0, .9, 17, None, None)

    ans = get_all_training_runs(db_path)
    assert ans == []

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a training run when the replayid does not exist
    with pytest.raises(Exception):
        add_training_run(db_path, 2, 0, .9, 17, 8, None)

    # Add a training run when the id does not exist
    with pytest.raises(Exception):
        add_training_run(db_path, 2, 0, .9, 17, None, 9)

    # Add a training run when there is no Training table
    add_training_run(db_path, 1, 0, .9, 17, None, None)

    ans = get_all_training_runs(db_path)
    assert len(ans) == 1

    add_replay(db_path, 1, {"test":1}, replay_folder)
    add_training_run(db_path, 2, 0, .9, 17, 1, None)
    ans = get_all_training_runs(db_path)
    assert len(ans) == 2

    sample_model = DQN()
    add_model(db_path, 1, sample_model, model_folder)
    add_training_run(db_path, 2, 0, .9, 17, None, 1)
    ans = get_all_training_runs(db_path)
    assert len(ans) == 3

    add_replay(db_path, 1, {"test":1}, replay_folder)
    add_model(db_path, 1, sample_model, model_folder)
    add_training_run(db_path, 2, 0, .9, 17, 2, 2)
    ans = get_all_training_runs(db_path)
    assert len(ans) == 4


def test_add_evaluation_run(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    replay_folder = os.path.join(tmp_path, "replay")
    model_folder = os.path.join(tmp_path, "model")

    # Add a run to a non existing database
    with pytest.raises(FileNotFoundError):
        add_evaluation_run(db_path, 1, 0, 17, None, None)

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a training when the experiment does not exist
    with pytest.raises(Exception):
        add_evaluation_run(db_path, 2, 0, 17, None, None)

    ans = get_all_training_runs(db_path)
    assert ans == []

    # Create the database so it exists
    add_experiment(db_path, "Testing")

    # Add a training run when the replayid does not exist
    with pytest.raises(Exception):
        add_evaluation_run(db_path, 2, 0, 17, 8, None)

    # Add a training run when the id does not exist
    with pytest.raises(Exception):
        add_evaluation_run(db_path, 2, 0, 17, None, 9)

    # Add a training run when there is no Training table
    add_evaluation_run(db_path, 1, 0, 17, None, None)

    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 1

    add_replay(db_path, 1, {"test":1}, replay_folder)
    add_evaluation_run(db_path, 2, 0, 17, 1, None)
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 2

    sample_model = DQN()
    add_model(db_path, 1, sample_model, model_folder)
    add_evaluation_run(db_path, 2, 0, 17, None, 1)
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 3

    add_replay(db_path, 1, {"test":1}, replay_folder)
    add_model(db_path, 1, sample_model, model_folder)
    add_evaluation_run(db_path, 2, 0, 17, 2, 2)
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 4


def test_delete_experiment(tmp_path):
    pass


def test_get_all_experiments(tmp_path):
    """
    """
    db_path = os.path.join(tmp_path, "test1.db")

    ans = get_all_experiments(db_path)
    assert ans == []

    add_experiment(db_path, "Testing")
    ans = get_all_experiments(db_path)
    assert len(ans) == 1

    add_experiment(db_path, "Testing2")
    ans = get_all_experiments(db_path)
    assert len(ans) == 2


def test_get_all_models(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    model_folder = os.path.join(tmp_path, "model")

    ans = get_all_models(db_path)
    assert ans == []

    sample_model = DQN()

    add_experiment(db_path, "testing")
    ans = add_model(db_path, 1, sample_model, "model")
    assert ans == 1
    
    ans = get_all_models(db_path)
    assert len(ans) == 1

    ans = add_model(db_path, 1, sample_model, "model")
    assert ans == 2

    ans = get_all_models(db_path)
    assert len(ans) == 2


def test_get_all_replays(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    replay_folder = os.path.join(tmp_path, "replay")

    ans = get_all_replays(db_path)
    assert ans == []

    add_experiment(db_path, "testing")
    ans = add_replay(db_path, 1, {'test': 1}, replay_folder)
    assert ans == 1

    ans = get_all_replays(db_path)
    assert len(ans) == 1

    ans = add_replay(db_path, 1, {'test': 1}, replay_folder)
    assert ans == 2

    ans = get_all_replays(db_path)
    assert len(ans) == 2


def test_get_all_training_runs(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")

    ans = get_all_training_runs(db_path)
    assert ans == []

    add_experiment(db_path, "testing")
    add_training_run(db_path, 1, 1, .9, .4, None, None)
    
    ans = get_all_training_runs(db_path)
    assert len(ans) == 1

    add_training_run(db_path, 1, 1, .9, .4, None, None)
    
    ans = get_all_training_runs(db_path)
    assert len(ans) == 2


def test_get_all_evaluation_runs(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")

    ans = get_all_evaluation_runs(db_path)
    assert ans == []

    add_experiment(db_path, "testing")
    add_evaluation_run(db_path, 1, 1, .4, None, None)
    
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 1

    add_evaluation_run(db_path, 1, 1, .4, None, None)
    
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 2


def test_delete_experiment(tmp_path):
    db_path = os.path.join(tmp_path, "test1.db")
    replay_folder = os.path.join(tmp_path, "replay")
    model_folder = os.path.join(tmp_path, "model")

    # Delete when the db doesnt exist
    with pytest.raises(FileNotFoundError):
        delete_experiment(db_path, replay_folder, model_folder, 1)
    
    experiment_id = add_experiment(db_path, "testing")
    experiment_id = add_experiment(db_path, "testing2")
    experiment_id = add_experiment(db_path, "testing3")

    ans = get_all_experiments(db_path)
    assert len(ans) == 3

    delete_experiment(db_path, replay_folder, model_folder, experiment_id)
    
    ans = get_all_experiments(db_path)
    assert len(ans) == 2


    experiment_id = add_experiment(db_path, "testing")
    sample_model = DQN()
    model_id = add_model(db_path, experiment_id, sample_model, model_folder)
    replay_id = add_replay(db_path, experiment_id, {'test':1}, replay_folder)
    add_training_run(db_path, experiment_id, 1, .9, 14, model_id, replay_id)
    add_evaluation_run(db_path, experiment_id, 1, 16, model_id, replay_id)

    ans = get_all_experiments(db_path)
    assert len(ans) == 3
    ans = get_all_models(db_path)
    assert len(ans) == 1
    model_path = get_model_path(model_folder, model_id)
    assert os.path.exists(model_path)
    ans = get_all_replays(db_path)
    assert len(ans) == 1
    replay_path = get_replay_path(replay_folder, replay_id)
    assert os.path.exists(replay_path)
    ans = get_all_training_runs(db_path)
    assert len(ans) == 1
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 1

    delete_experiment(db_path, replay_folder, model_folder, experiment_id)
    
    ans = get_all_experiments(db_path)
    assert len(ans) == 2
    
    ans = get_all_models(db_path)
    assert len(ans) == 0
    assert not os.path.exists(model_path)
    
    ans = get_all_replays(db_path)
    assert len(ans) == 0
    assert not os.path.exists(replay_path)
    
    ans = get_all_training_runs(db_path)
    assert len(ans) == 0
    
    ans = get_all_evaluation_runs(db_path)
    assert len(ans) == 0
    
    