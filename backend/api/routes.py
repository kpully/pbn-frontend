from flask import Flask, send_file, request, current_app, session
import io
from .classes.image_processor import ImageProcessor
import logging
import json

# app = Flask(__name__)
# from api import app
logging.basicConfig(level=logging.INFO)


# @app.route('/hello_world')
# def hello_world():
# 	app.logger.info("hello world")
# 	return {"message":"Hello, World!"}

@current_app.route('/get_palette', methods=['GET', 'POST'])
def get_limited_palette_image():
	# app.logger.info("get palette")
	file = request.files["file"]
	processor = ImageProcessor(file)
	processor.resize_image(3)
	processor.set_palette()
	# app.logger.info(processor.palette)
	#
	file_object = io.BytesIO()
	processor.palette_limited_image.save(file_object, 'PNG')
	file_object.seek(0)

	session['pixel_data'] = json.dumps(processor.pixel_data_limited_palette.tolist())

	return send_file(file_object, mimetype="image/PNG")

@current_app.route('/get_outline')
def get_outline():
	processor = session.get("pixel_data")
	if processor:
		return {"message":"success"}
	else:
		return {"message":"fail"}


# if __name__ == "__main__":
#     app.run(debug=True)

