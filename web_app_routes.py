from flask import render_template, request, flash, session, redirect, url_for, send_file, jsonify
import os
import sys
import io
import base64

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database_ops import *
from config import DATABASE_PATH, REPLAY_PATH, MODEL_PATH
from medium.model_training import training_loop

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


        # TODO - Add in a view of the Evaluation Loops
        # TODO - Add in a view of the Evaluation Rewards
        # TODO - Add in a view of the Evaluation Run Times

        evaluation_runs = query_evaluation_loops_by_experiment(DATABASE_PATH, experiment_id)

        return render_template(
            "experiment.html",
            experiment_id=experiment_id,
            training_runs=training_runs,
            evaluation_runs=evaluation_runs,
            )


    @app.route("/run_training_loop", methods = ["POST"])
    def run_training_loop():
        """
        Run the training loop
        """
        # print("Request data (raw):", request.data)
        # print("Request content type:", request.content_type)
        config = request.get_json(force=True)

        # Create a new experiment
        experiment_id = add_experiment(DATABASE_PATH, config["EXPERIMENT_DESCRIPTION"])

        # Add the experiment ID to the config
        config["EXPERIMENT_ID"] = experiment_id
        config["DATABASE_PATH"] = DATABASE_PATH
        config["MODEL_PATH"] = MODEL_PATH
        config["REPLAY_PATH"] = REPLAY_PATH


        print("Got the config")
        print(config)
        print(type(config))

        # This should be started in a new thread or something 
        # training_loop()
        return jsonify({"status": "ok"})
        return redirect(url_for("home"))


# Kind of need some way to do the config
# Kind of need some way to start the training threads in the background and
# make sure they don't di