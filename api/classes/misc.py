    # @classmethod
    # def make_image(cls, original_image, pixel_to_blob, blobs):
    #     new_pixels = []
    #     for j in range(original_image.height):
    #         row = []
    #         for i in range(original_image.width):
    #             b = pixel_to_blob[(i, j)]
    #             bl = blobs[b]
    #             row.append(bl.rgb)
    #         new_pixels.append(row)
    #     array = np.array(new_pixels, dtype=np.uint8)
    #     new_image = Image.fromarray(array)
    #     return new_image


        # @classmethod
    # def _distance(cls, c1, c2):
    #     """
    #     Utility method to calculate RGB euclidean distance between anchor pixel and given pixel
    #     """
    #     d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
    #     return d
