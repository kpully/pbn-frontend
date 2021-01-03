from flask import Flask
from .classes.image_processor import ImageProcessor

app = Flask(__name__)

@app.route('/hello_world')
def hello_world():
    return {"message":"Hello, World!"}

@app.route('/get_limited_palette_image')
def get_limited_palette_image():
	img_path = "../public/images/pupper_small.jpg"
	processor = ImageProcessor(img_path)
	processor.set_palette()
	return processor.palette_limited_image

if __name__ == "__main__":
    app.run()

