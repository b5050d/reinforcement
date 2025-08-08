from flask import (
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
)
import os
import sys
from threading import Thread
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reinforcement_app.management.database_ops import (
    get_all_experiments,
    query_training_loops_by_experiment,
    query_evaluation_loops_by_experiment,
    add_experiment,
    get_replay_path,
    add_config,
    load_config,
)
from config import DATABASE_PATH, REPLAY_PATH, MODEL_PATH, CONFIG_PATH
from reinforcement_app.rl.model_training import training_loop
from reinforcement_app.simulation.environment import Environment


def set_up_routes(app):
    """
    Set up routes for our app
    """

    @app.route("/")
    def home():
        # Find the Experiments

        # Alright I need to find the experiments
        experiments = get_all_experiments(DATABASE_PATH)
        print(experiments)

        experiments_table = []
        for exp in experiments:
            new_row = []
            for e in exp:
                new_row.append(e)
            new_row.append(f"/experiment/{exp[0]}")
            experiments_table.append(new_row)
        print(experiments_table)

        return render_template("home.html", experiments_table=experiments_table)

    @app.route("/experiment/<experiment_id>")
    def experiment(experiment_id):
        # Check if the experiment is present in the Experiment db
        # Is the Experiment present
        experiment_id = int(experiment_id)
        ans = get_all_experiments(DATABASE_PATH)
        print(ans)
        success = False
        for item in ans:
            if item[0] == experiment_id:
                success = True
        if not success:
            return "No Experiment Found"

        # TODO - Add in a view of the Training Loops table
        # TODO - Add in a plot of the Training Rewards
        # TODO - Add in a plot of the Training Epsilons
        # TODO - Add in a plot of the Training Runtimes

        training_runs = query_training_loops_by_experiment(DATABASE_PATH, experiment_id)

        print("Training Runs")
        print(training_runs)

        # TODO - Add in a view of the Evaluation Loops
        # TODO - Add in a view of the Evaluation Rewards
        # TODO - Add in a view of the Evaluation Run Times

        evaluation_runs = query_evaluation_loops_by_experiment(
            DATABASE_PATH, experiment_id
        )

        return render_template(
            "experiment.html",
            experiment_id=experiment_id,
            training_runs=training_runs,
            evaluation_runs=evaluation_runs,
        )

    @app.route("/run_training_loop", methods=["POST"])
    def run_training_loop():
        """
        Run the training loop
        """
        # print("Request data (raw):", request.data)
        # print("Request content type:", request.content_type)
        config = request.get_json(force=True)

        config_id = add_config(CONFIG_PATH, config)

        # Create a new experiment
        experiment_id = add_experiment(
            DATABASE_PATH, config["EXPERIMENT_DESCRIPTION"], config_id
        )

        # Add the experiment ID to the config
        config["EXPERIMENT_ID"] = experiment_id
        config["DATABASE_PATH"] = DATABASE_PATH
        config["MODEL_PATH"] = MODEL_PATH
        config["REPLAY_PATH"] = REPLAY_PATH
        config["CONFIG_PATH"] = CONFIG_PATH

        # This should be started in a new thread or something
        new_thread = Thread(
            target=training_loop,
            args=(
                config,
                experiment_id,
            ),
        )
        new_thread.start()

        # time.sleep(1)

        # return jsonify({"status": "ok"})
        return redirect(url_for("home"))

    @app.route("/play_game", methods=["POST"])
    def play_game():
        """
        Play the game as a user
        """

        config = request.get_json(force=True)

        game_thread = Environment(config)

        new_thread = Thread(target=game_thread.play, daemon=True)
        new_thread.start()

        return jsonify({"status": "ok"})

    @app.route("/replay_<replay_id>", methods=["POST"])
    def replay(replay_id):
        # TODO - Should add the food locations to the config
        # If using random, then the food locations will be different
        # from the replay.
        print(f"Got the replay: {replay_id}")

        replay_file_path = get_replay_path(REPLAY_PATH, replay_id)
        if os.path.exists(replay_file_path):
            # Load the replay into a dict
            with open(replay_file_path, "r") as f:
                replay_dict = json.load(f)

            game_thread = Environment(replay_dict)

            new_thread = Thread(target=game_thread.play, args=(True,), daemon=True)

            new_thread.start()
        else:
            # TODO - Flash that the replay was not found
            pass
        return jsonify({"status": "ok"})

    @app.route("/replay_<replay_id>", methods=["POST"])
    def config(config_id):
        load_config()


# Kind of need some way to do the config
# Kind of need some way to start the training threads in the background and
# make sure they don't di
