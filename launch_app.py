"""
Simple Script to allow the user to launch the application
"""

from src.reinforcement_app.web.web_app import AppFactory


if __name__ == "__main__":
    factory = AppFactory()
    app = factory.create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
