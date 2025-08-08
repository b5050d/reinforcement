"""
Database operation scripts
"""

import sqlite3
import os
import torch
import json


def table_exists(db_path: str, table_name: str) -> bool:
    """
    Check if a table exists in the given SQLite database.
    """
    if not os.path.exists(db_path):
        return False

    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name=?;
        """,
            (table_name,),
        )
        return cursor.fetchone() is not None


# TODO - Add config to the experiment table...
create_experiment_table_cmd = """
CREATE TABLE IF NOT EXISTS Experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL
)
"""


create_model_table_cmd = """
CREATE TABLE IF NOT EXISTS Model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
)
"""


create_replay_table_cmd = """
CREATE TABLE IF NOT EXISTS Replay (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
)
"""


create_training_table_cmd = """
CREATE TABLE IF NOT EXISTS Training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    episode INTEGER,
    epsilon REAL,
    reward REAL,
    replay_id INTEGER,
    model_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
)
"""

create_evaluation_table_cmd = """
CREATE TABLE IF NOT EXISTS Evaluation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    episode INTEGER,
    reward REAL,
    replay_id INTEGER,
    model_id INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
)
"""

insert_experiment_cmd = """
INSERT INTO Experiment (description) VALUES (?)
"""


insert_model_cmd = """
INSERT INTO Model (experiment_id) VALUES (?)
"""


insert_replay_cmd = """
INSERT INTO Replay (experiment_id) VALUES (?)
"""


insert_training_cmd = """
INSERT INTO Training (experiment_id, episode, epsilon, reward, replay_id, model_id) VALUES (?,?,?,?,?,?)
"""

insert_evaluation_cmd = """
INSERT INTO Evaluation (experiment_id, episode, reward, replay_id, model_id) VALUES (?,?,?,?,?)
"""


def get_connection(database_path):
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_experiment_table(database_path):
    """
    Creates a table if DNE
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute(create_experiment_table_cmd)
        connection.commit()


def create_model_table(database_path):
    """
    Creates a table if DNE
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute(create_model_table_cmd)
        connection.commit()


def create_replay_table(database_path):
    """
    Creates a table if DNE
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute(create_replay_table_cmd)
        connection.commit()


def create_training_table(database_path):
    """
    Creates a table if DNE
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute(create_training_table_cmd)
        connection.commit()


def create_evaluation_table(database_path):
    """
    Creates a table if it DNE
    """
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute(create_evaluation_table_cmd)
        connection.commit()


def add_experiment(database_path, description):
    """
    Add an experiment to the Experiement table in the database

    returns the experiment ID
    """
    create_experiment_table(database_path)

    # Add the experiment to the experiement table
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        # Create the Users table if it doesn't exist yet
        cursor.execute(insert_experiment_cmd, (description,))
        experiment_id = cursor.lastrowid
        connection.commit()

    # TODO - Return the Experiment ID
    return experiment_id


def add_model(database_path, experiment, model, model_folder):
    """
    Saves the model to a file and then adds it to the database

    returns the model id
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database DNE, cant add model")

    # Check that the experiment exists
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment:
            success = True

    if not success:
        raise Exception("No Experiement Found")

    create_model_table(database_path)

    # Add the model to the model table
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        # Create the Users table if it doesn't exist yet
        cursor.execute(insert_model_cmd, (experiment,))
        model_id = cursor.lastrowid
        connection.commit()

    # Saving of the model
    os.makedirs(model_folder, exist_ok=True)
    model_file_path = get_model_path(model_folder, model_id)
    torch.save(model.state_dict(), model_file_path)
    assert os.path.exists(model_file_path)

    # Return the Model ID
    return model_id


def add_replay(database_path, experiment, replay_dict, replay_folder):
    """
    Saves the replay to a json file and then adds a reference to the database

    returns the replay id
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database DNE, cant add model")

    # Check that the experiment exists
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    create_replay_table(database_path)

    # Add the replay to the replay table
    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        # Create the Users table if it doesn't exist yet
        cursor.execute(insert_replay_cmd, (experiment,))
        replay_id = cursor.lastrowid
        connection.commit()

    # TODO - Add in the saving of the files
    os.makedirs(replay_folder, exist_ok=True)
    replay_file_path = get_replay_path(replay_folder, replay_id)
    with open(replay_file_path, "w") as f:
        json.dump(replay_dict, f)
    assert os.path.exists(replay_file_path)

    # Return the Replay ID
    return replay_id


def add_training_run(
    database_path, experiment, episode, epsilon, reward, replay_id=None, model_id=None
):
    """
    Adds the training run stats to the Training database table
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database DNE, cant add model")

    # Check that the experiment is in the database already
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    create_training_table(database_path)

    # Check that epsilon is valid

    # Check that replay_id and model_id are present in the database
    if replay_id is None:
        replay_id = -1
    else:
        success = False
        ans = get_all_replays(database_path)
        for a in ans:
            if replay_id == a[0]:
                success = True

        if not success:
            raise Exception("No matching Replay ID present")

    if model_id is None:
        model_id = -1
    else:
        success = False
        ans = get_all_models(database_path)
        for a in ans:
            if model_id == a[0]:
                success = True

        if not success:
            raise Exception("No matching Model ID present")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        # Create the Training table if it doesn't exist yet
        cursor.execute(
            insert_training_cmd,
            (experiment, episode, epsilon, reward, replay_id, model_id),
        )
        connection.commit()


def add_evaluation_run(
    database_path, experiment, episode, reward, replay_id=None, model_id=None
):
    """
    Adds an evaluation run to the Evaluation database table
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database DNE, cant add model")

    # Check that the experiment is in the database already
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    create_evaluation_table(database_path)

    # Check that replay_id and model_id are present in the database
    if replay_id is None:
        replay_id = -1
    else:
        success = False
        ans = get_all_replays(database_path)
        for a in ans:
            if replay_id == a[0]:
                success = True

        if not success:
            raise Exception("No matching Replay ID present")

    if model_id is None:
        model_id = -1
    else:
        success = False
        ans = get_all_models(database_path)
        for a in ans:
            if model_id == a[0]:
                success = True

        if not success:
            raise Exception("No matching Model ID present")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()

        # Create the Training table if it doesn't exist yet
        cursor.execute(
            insert_evaluation_cmd, (experiment, episode, reward, replay_id, model_id)
        )
        connection.commit()


def get_all_experiments(database_path):
    """
    Get all the experiments in the table
    """
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Experiment"):
        return []

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Experiment")
        rows = cursor.fetchall()
    return rows


def get_all_models(database_path):
    """
    Get all models in the table
    """
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Model"):
        return []

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Model")
        rows = cursor.fetchall()
    return rows


def get_all_replays(database_path):
    """
    Get all the replays in the table
    """
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Replay"):
        return []

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Replay")
        rows = cursor.fetchall()
    return rows


def get_all_training_runs(database_path):
    """
    Get all the tables in the database
    """
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Training"):
        return []

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Training")
        rows = cursor.fetchall()
    return rows


def get_all_evaluation_runs(database_path):
    """
    Get all the evaluation datas in the database
    """
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Evaluation"):
        return []

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Evaluation")
        rows = cursor.fetchall()
    return rows


def query_models_per_experiment(database_path, experiment_id):
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Model"):
        return []

    # Is the Experiment present
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment_id:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Model WHERE experiment_id = ?", (experiment_id,))
        rows = cursor.fetchall()
    return rows


def query_replays_per_experiment(database_path, experiment_id):
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Replay"):
        return []

    # Is the Experiment present
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment_id:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Replay WHERE experiment_id = ?", (experiment_id,))
        rows = cursor.fetchall()
    return rows


def query_training_loops_by_experiment(database_path, experiment_id):
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Training"):
        return []

    # Is the Experiment present
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment_id:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM Training WHERE experiment_id = ?", (experiment_id,)
        )
        rows = cursor.fetchall()
    return rows


def query_evaluation_loops_by_experiment(database_path, experiment_id):
    if not os.path.exists(database_path):
        return []

    if not table_exists(database_path, "Evaluation"):
        return []

    # Is the Experiment present
    ans = get_all_experiments(database_path)
    success = False
    for item in ans:
        if item[0] == experiment_id:
            success = True

    if not success:
        raise Exception("No Experiment Found")

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM Evaluation WHERE experiment_id = ?", (experiment_id,)
        )
        rows = cursor.fetchall()
    return rows


def get_model_path(model_folder_path, model_id):
    """
    Get a model path from the id
    """
    return os.path.join(model_folder_path, f"{model_id}.pth")


def get_replay_path(replay_folder_path, replay_id):
    """
    Get a replay path from the id
    """
    return os.path.join(replay_folder_path, f"{replay_id}.json")


def delete_experiment(
    database_path, replay_folder_path, model_folder_path, experiment_id
):
    """
    Delete the experiment and all ties to it
    """
    if not os.path.exists(database_path):
        raise FileNotFoundError("Database DNE, cant add model")

    # Find all Replays associated with the experiment
    replays = query_replays_per_experiment(database_path, experiment_id)

    # Find all Models associated with the experiment
    models = query_models_per_experiment(database_path, experiment_id)

    # Go and delete the replays
    for replay in replays:
        replay_path = get_replay_path(replay_folder_path, replay[0])
        if not os.path.exists(replay_path):
            print("Warning! The Replay didnt exist")
        else:
            os.remove(replay_path)

    # Go and delete the models
    for model in models:
        model_path = get_model_path(model_folder_path, model[0])
        if not os.path.exists(model_path):
            print("Warning! The Model didnt exist")
        else:
            os.remove(model_path)

    with get_connection(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM Experiment WHERE id = ?", (experiment_id,))


if __name__ == "__main__":
    from config import DATABASE_PATH

    ans = get_all_training_runs(DATABASE_PATH)

    print(ans)
