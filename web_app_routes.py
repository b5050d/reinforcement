from flask import render_template, request, flash, session, redirect, url_for, send_file
import os
import sys
import io
import base64

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def set_up_routes(app):
    """
    Set up routes for our app
    """

    @app.route("/")
    def index():
        return render_template("home.html")
