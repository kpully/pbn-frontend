from flask import Flask, send_file, request
import io
from .classes.image_processor import ImageProcessor
import logging
import numpy as np

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


@app.route('/hello_world')
def hello_world():
	app.logger.info("hello world")
	return {"message":"Hello, World!"}

@app.route('/get_palette', methods=['GET', 'POST'])
def get_limited_palette_image():
	app.logger.info("get palette")
	app.logger.info(request.files)
	app.logger.info("setting palette...")
	file = request.files["file"]
	processor = ImageProcessor(file)
	processor.resize_image(3)
	processor.set_palette()
	app.logger.info(processor.palette)
	# return {"message": processor.palette}


	file_object = io.BytesIO()
	processor.palette_limited_image.save(file_object, 'PNG')
	file_object.seek(0)
	return send_file(file_object, mimetype="image/PNG")

# if __name__ == "__main__":
#     app.run(debug=True)

