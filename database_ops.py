"""
Database operation scripts
"""


def generate_uuid():
    """
    Generate a uuid to use as the id for a given file
    """
    pass


def add_experiment(database_path, description, config):
    """
    Add an experiment to the Experiement table in the database

    returns the experiment ID
    """
    pass


def add_model(model):
    """
    Saves the model to a file and then adds it to the database

    returns the model id
    """
    pass


def add_replay(database_path, replay_path, replay_dict):
    """
    Saves the replay to a json file and then adds a reference to the database

    returns the replay id    
    """
    assert type(replay_dict) is dict


def add_training_run(database_path, experiment, episode, epsilon, reward, replay_id=None, model_id=None):
    """
    Adds the training run stats to the Training database table
    """
    # Check that the experiment is in the database already

    # Check that the episode is not already populated

    # Check that epsilon is valid

    # Check that replay_id and model_id are present in the database


def add_evaluation_run(database_path, experiment, episode, reward, replay_id=None, model_id=None):
    """
    Adds an evaluation run to the Evaluation database table
    """
    # Check that the experiment is in the database already

    # Check that the episode is not already populated

    # Check that replay_id and model_id are present in the database


def delete_experiment(database_path, experiement_id, model_path, replay_path):
    """
    Delete the experiment and all ties to it
    """
    # Find all Replays associated with the experiment