from classes.image_processor import ImageProcessor

if __name__ == "__main__":
	img_path = "../public/images/pupper_small.jpg"
	processor = ImageProcessor(img_path)
	processor.img_name
	processor.process_image()
	processor.remove_small_blobs()
	processor.make_image()
	processed_img = processor.processed_img

	with open('Failed.png', 'w') as file:
		file.write(processed_img)
