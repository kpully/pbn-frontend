import os
from flask import Flask
from flask_session import Session

sess = Session()


def create_app():
	app = Flask(__name__)
	# app.config.from_object("config.py")
	# app.config.from_pyfile('config.py')
	app.config['SECRET_KEY']="test"

	app.config['SESSION_TYPE'] = 'filesystem'

	# initialize plugins
	sess.init_app(app)


	with app.app_context():
		from . import routes

	return app




