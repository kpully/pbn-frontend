from classes.image_processor import ImageProcessor

if __name__ == "__main__":
	img_path = "../public/images/pupper_small.jpg"
	processor = ImageProcessor(img_path)
	processor.set_palette()