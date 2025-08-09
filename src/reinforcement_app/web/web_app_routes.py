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
import io
import base64

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

os.environ["MPLBACKEND"] = "Agg"
from matplotlib import pyplot as plt


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
                config_id = item[2]
        if not success:
            return "No Experiment Found"

        training_runs = query_training_loops_by_experiment(DATABASE_PATH, experiment_id)

        # TODO - Add in a plot of the Training Rewards
        training_rewards = []
        training_runtimes = []
        training_epsilons = []
        x_axis = []
        for iter, tr in enumerate(training_runs):
            x_axis.append(iter)
            training_epsilons.append(float(tr[3]))
            training_rewards.append(float(tr[4]))
            training_runtimes.append(float(tr[5]))

        def generate_plot(title, x_data, y_data, x_axis, y_axis):
            """
            Generate a plot for display in the webpage
            """
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_data, y_data)
            ax.set_title(title)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.grid()

            # Save to BytesIO
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)
            plt.cla()

            return image_base64

        train_eps_plot = generate_plot(
            "Training Epsilons",
            x_axis,
            training_epsilons,
            "Episodes",
            "Epsilon Value [0-1]",
        )
        train_rwd_plot = generate_plot(
            "Training Rewards",
            x_axis,
            training_rewards,
            "Episodes",
            "Rewards per Episode",
        )
        train_tim_plot = generate_plot(
            "Training Runtimes", x_axis, training_runtimes, "Episodes", "Runtime [s]"
        )

        train_images = [train_eps_plot, train_rwd_plot, train_tim_plot]

        evaluation_runs = query_evaluation_loops_by_experiment(
            DATABASE_PATH, experiment_id
        )

        evaluation_rewards = []
        evaluation_runtimes = []
        x_axis = []
        for iter, tr in enumerate(evaluation_runs):
            x_axis.append(iter)
            evaluation_rewards.append(float(tr[3]))
            evaluation_runtimes.append(float(tr[4]))

        eval_rwd_plot = generate_plot(
            "Evaluation Rewards",
            x_axis,
            evaluation_rewards,
            "Episodes",
            "Rewards per Eval Run",
        )
        eval_tim_plot = generate_plot(
            "Evaluation Runtimes",
            x_axis,
            evaluation_runtimes,
            "Episodes",
            "Runtime [s]",
        )

        eval_images = [eval_rwd_plot, eval_tim_plot]

        # Load the json as a dict
        config = load_config(CONFIG_PATH, config_id)
        # Dump the json to a string which looks nice
        config_data = json.dumps(config, indent=2)

        return render_template(
            "experiment.html",
            experiment_id=experiment_id,
            training_runs=training_runs,
            evaluation_runs=evaluation_runs,
            config_data=config_data,
            train_images=train_images,
            eval_images=eval_images,
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
