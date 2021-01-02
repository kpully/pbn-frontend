from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import cv2
from sklearn.cluster import KMeans

from classes.blob import Blob


class ImageProcessor:
    def __init__(self, image_name, threshold=50, complex_palette=False):
        # image versions
        self.image_name = image_name
        self.image = Image.open(image_name).convert("RGB")

        # image versions
        self.original_image = Image.open(image_name).convert("RGB")
        self.processed_image = None
        self.palette_limited_image = None

        # pixel, blob data structures
        self.pixel_data = np.reshape(np.array(self.image.getdata()), (self.image.height, self.image.width, 3))
        self.pixel_data_limited_palette = np.empty((self.image.height, self.image.width, 3), dtype=np.uint8)
        self.blobs = {}  # map of integer keys mapping to Blob object
        self.pixel_to_blob = {}  # map of pixel coordinate keys to key in self.blobs

        # palette data
        self.palette = None
        self.palette_viz = None

        # constants
        self.THRESHOLD = threshold


    def make_outline(self):
        pass


    # @classmethod
    # def _distance(cls, c1, c2):
    #     """
    #     Utility method to calculate RGB euclidean distance between anchor pixel and given pixel
    #     """
    #     d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
    #     return d


    def process_image(self):
        """
        Create blobs with similarly-colored pixels by calling floodfill
        """
        for x in range(self.image.width):
            for y in range(self.image.height):
                if (x, y) not in self.pixel_to_blob:
                    blob_count = len(self.blobs)
                    self._flood_fill(x, y, blob_count)
        self.processed_image = ImageProcessor.make_image(self.original_image, self.pixel_to_blob, self.blobs)


    def remove_small_blobs(self):
        """
        Loop through blobs map from 0 to len(blobs)
        If blob size is under a designated threshold, merge the blob with its largest neighbor
        After merge, update blobs map accordingly by removing small blob
        """

        # find small blobs
        blob_count = len(self.blobs)
        for i in range(0, blob_count):
            if self.blobs[i]:  # if blob still exists in map
                if self.blobs[i].get_size() < 25:
                    small_blob = self.blobs[i]
                    neighbor_blobs = small_blob._neighbor_blobs(self.image, self.pixel_to_blob)
                    m = max(neighbor_blobs, key=lambda x: self.blobs[x].get_size())
                    self.blobs[m].merge(small_blob)
                    # reassign pixels in smaller blob to largest neighbor blob
                    for p_x, p_y in small_blob.pixels:
                        self.pixel_to_blob[(p_x, p_y)] = m
                    del self.blobs[i]


    def resize_image(self, scale):
        """
        Reduce size of image by given scale to improve performance on large images
        """
        self.image = self.image.resize((self.image.width // scale, self.image.height // scale))
        self.original_image = self.original_image.resize(
            (self.original_image.width // scale, self.original_image.height // scale))
        # reset pixel data to new bounds
        self.pixel_data = np.reshape(np.array(self.image.getdata()), (self.image.height, self.image.width, 3))
        self.pixel_data_limited_palette = np.empty((self.image.height, self.image.width, 3))


    def _flood_fill(self, x1, y1, blob_count):
        """
        Internal method
        Stack-based DFS implementation of floodfill to group like-colored neighboring pixels
        """
        # constants
        p1 = self.image.getpixel((x1, y1))
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        # initialize data structures
        seen = set()
        curr_blob = set()

        def _get_distance(x, y):
            """
            Utility method internal to _flood_fill to calculate RGB euclidean distance
            between a blob's anchor pixel and given pixel
            """
            p2 = self.image.getpixel((x, y))
            d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            return d

        # begin DFS
        seen.add((x1, y1))
        stack = [(x1, y1)]
        while stack:
            x, y = stack.pop()
            # initialize Pixel object from coordinates popped off stack
            curr_blob.add((x, y))
            # map pixel coordinates to current blob
            self.pixel_to_blob[(x, y)] = blob_count
            # iterate through neighbors to check for similarly-colored pixels
            for dx, dy in directions:
                if (x + dx, y + dy) in seen:
                    continue
                else:
                    seen.add((x + dx, y + dy))
                    if (0 <= x + dx < self.image.width) and (0 <= y + dy < self.image.height):
                        if _get_distance(x + dx, y + dy) <= self.THRESHOLD:
                            stack.append((x + dx, y + dy))
        # once stack is empty, add curr_blob as a Blob object blobs map
        b = Blob(pixels=curr_blob, palette=self.palette)
        b.update_rgb(self.pixel_data)
        self.blobs[blob_count] = b


    @classmethod
    def make_image(cls, pixel_data):
        array = np.array(pixel_data, dtype=np.uint8)
        new_image = Image.fromarray(array)
        return new_image


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

    def set_palette(self, num_clusters=5):
        # ing = cv2.
        # img = cv2.cvtColor(np.array(self.image), )
        img = self.pixel_data.reshape((self.pixel_data.shape[0] * self.pixel_data.shape[1], 3))
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(img)
        colors = [[int(color) for color in cluster] for cluster in kmeans.cluster_centers_]
        assert len(kmeans.labels_)==self.image.width*self.image.height
        # create pixel data with limited palette
        rows, cols = self.pixel_data_limited_palette.shape[0], self.pixel_data_limited_palette.shape[1]
        for x in range(rows):
            for y in range(cols):
                index = y+cols*x
                self.pixel_data_limited_palette[x][y] = colors[kmeans.labels_[index]]

        self.palette = colors
        palette_viz = Image.new("RGB", (500, 50), (255, 255, 255))
        draw = ImageDraw.Draw(palette_viz)
        i = 0
        radius = 20
        spacing = 25
        for color in colors:
            draw.ellipse(xy=[0 + i * spacing, 0, radius + i * spacing, radius], fill=tuple(color))
            i += 1
        self.palette_viz = palette_viz
        self.palette_limited_image = ImageProcessor.make_image(self.pixel_data_limited_palette)
