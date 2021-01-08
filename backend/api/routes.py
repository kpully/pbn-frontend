from flask import Flask, send_file, request, current_app, session
import io
from .classes.image_processor import ImageProcessor
import logging
import json
import numpy as np

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

	file_object = io.BytesIO()
	processor.palette_limited_image.save(file_object, 'PNG')
	file_object.seek(0)

	# session['pixel_data'] = json.dumps(processor.pixel_data_limited_palette.tolist())
	session['pixel_data'] = processor.pixel_data_limited_palette
	session['img_width'] = processor.image.width
	session['img_height'] = processor.image.height

	return send_file(file_object, mimetype="image/PNG")

@current_app.route('/get_outline')
def get_outline():
	pixel_data = np.array(session.get("pixel_data"))

	outline = ImageProcessor.make_outline(pixel_data)
	file_object = io.BytesIO()
	outline.save(file_object, "PNG")
	file_object.seek(0)
	return send_file(file_object, mimetype="image/PNG")


# if __name__ == "__main__":
#     app.run(debug=True)

