"""
Main App Functionality.
"""

from flask import Flask
import os
from utils.frontend_link import FrontendLink
from utils.database_methods import UsersTable
from utils.routes import init_routes
from mock import MagicMock


class AppFactory:
    """
    Class to handle web app creation,

    makes it a lot easier to mock out things that need to get mocked out
    """

    def __init__(self):
        """
        Set up the app
        """
        self.define_config()

    def define_config(self):
        """
        Define the config to use in the app
        """
        self.config = {
            "TELEM_ATTEMPT_GEN": "telem:1",
            "TELEM_SUCCESS_GEN": "telem:2",
            "TELEM_FAILURE_GEN": "telem:3",
            "TELEM_SIGNUPS": "telem:4",
            "TELEM_LOGINS": "telem:5",
            "TELEM_LOGOUTS": "telem:6",
            "TELEM_EMAIL_VERS": "telem:7",
            "SECRET_KEY": os.getenv("SECRET_KEY", "random_secret_key"),
            # "MAIL_SERVER": "smtp.sendgrid.net",
            # "MAIL_PORT": 587,
            # "MAIL_USE_TLS": True,
            # "MAIL_USERNAME": "apikey",
            # "MAIL_PASSWORD": os.getenv("MAIL_PASSWORD", "no_password_provided"),
            # "MAIL_DEFAULT_SENDER": "no-reply@yourdomain.com",
            "DATABASE_PATH": os.getenv("DATABASE_PATH", "/app/app_data/sample.db"),
        }

    def create_app(self, test=False):
        """
        Create the actual Flask app and return the app as an obj
        """
        self.app = Flask(__name__, static_folder="static")
        self.app.config.update(self.config)

        if test:
            self._mock_connections(test)
        else:
            self._set_up_connections()

        self._set_up_routes()

        return self.app

    def _set_up_connections(self):
        """
        Set up the connections to necessary interfaces

        Can be abstracted out for test purposes
        """
        self.app.frontend_link = FrontendLink()
        # self.app.mail = Mail(self.app)
        # self.app.s = URLSafeTimedSerializer(self.app.config["SECRET_KEY"])

        database_path = self.app.config["DATABASE_PATH"]
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        self.app.users_table = UsersTable(database_path)

    def _mock_connections(self, test_path):
        """
        Set up mocked connections

        Hint provide test_path as the path to the test database
        """
        self.app.frontend_link = MagicMock()
        self.app.mail = MagicMock()

        database_path = test_path
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        self.app.users_table = UsersTable(database_path)

    def _set_up_routes(self):
        """
        Set up the routes
        """
        init_routes(self.app)


if __name__ == "__main__":
    factory = AppFactory()
    app = factory.create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)