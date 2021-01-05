from classes.image_processor import ImageProcessor

if __name__ == "__main__":
	img_path = "../public/images/pupper_small.jpg"
	processor = ImageProcessor(img_path)
	processor.resize_image(6)
	# processor.set_palette()
	processor.make_blobs()
	processor.make_outline(processor.pixel_data_blobbed)