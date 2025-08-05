"""
Application Frontend
"""


from flask import Flask
from web_app_routes import set_up_routes


class AppFactory:
    """
    Class to handle web app creation
    """

    def __init__(self):
        self.define_config()

    def define_config(self):
        pass

    def create_app(self):
        self.app = Flask(__name__)

        self.set_up_routes()

        return self.app

    def set_up_routes(self):
        set_up_routes(self.app)


# TODO - Implement a button that allows the user to play the game
# with the selected config

# TODO - Implement a plot that will plot the selected experiement's loss / rewards


if __name__ == "__main__":
    factory = AppFactory()
    app = factory.create_app()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )