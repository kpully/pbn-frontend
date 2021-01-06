from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import cv2
from sklearn.cluster import KMeans

from .blob import Blob



class ImageProcessor:
    # class variables
    WHITE = [0, 0, 0]
    BLACK = [255, 255, 255]
    DIRECTIONS = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def __init__(self, image_name, threshold=50, complex_palette=False):
        # image vars
        self.image_name = image_name
        self.image = Image.open(image_name).convert("RGB")

        # image versions
        self.original_image = Image.open(image_name).convert("RGB")
        self.blobbed_image = None
        self.palette_limited_image = None
        self.outline_image = None

        # pixel, blob data structures
        self.pixel_data = np.reshape(np.array(self.image.getdata()), (self.image.height, self.image.width, 3))
        self.pixel_data_limited_palette = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)
        self.pixel_data_blobbed = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)
        self.pixel_data_outline = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)

        self.blobs = {}  # map of integer keys mapping to Blob object
        self.pixel_to_blob = {}  # map of pixel coordinate keys to key in self.blobs

        # palette data
        self.palette = None
        self.palette_viz = None

        # constants
        self.THRESHOLD = threshold


    def make_blobs(self, reduced_palette=False, override=False):
        """
        Create blobs with similarly-colored pixels by calling floodfill
        """
        pixel_data = self.pixel_data_limited_palette if reduced_palette else self.pixel_data

        # reset blob data
        if override:
            self.pixel_to_blob, self.blobs = {}, {}

        for x in range(self.image.height):
            for y in range(self.image.width):
                if (x, y) not in self.pixel_to_blob:
                    blob_count = len(self.blobs)
                    self._flood_fill(x, y, blob_count, pixel_data)

        # make blobbed pixel data based off results from floodfill
        self.make_blobbed_pixel_data()
        self.blobbed_image = ImageProcessor.make_image(self.pixel_data_blobbed)


    def make_outline(self, pixel_data):
        for i in range(1, self.image.height-1):
            for j in range(1, self.image.width-1):
                if self._neighbors_same(i, j, pixel_data):
                    self.pixel_data_outline[i][j] = ImageProcessor.BLACK
                else:
                    self.pixel_data_outline[i][j] = ImageProcessor.WHITE
        self.outline_image = ImageProcessor.make_image(self.pixel_data_outline)


    def _neighbors_same(self, x, y, pixel_data):
        for dx, dy in ImageProcessor.DIRECTIONS:
            if not all(pixel_data[x][y] == pixel_data[x+dx][y+dy]):
                return False
        return True


    def make_blobbed_pixel_data(self):
        h = self.image.height
        w = self.image.width
        for i in range(self.image.height):
            for j in range(self.image.width):
                b = self.pixel_to_blob[(i, j)]
                blb = self.blobs[b]
                self.pixel_data_blobbed[i][j] = blb.rgb


    def _flood_fill(self, x1, y1, blob_count, pixel_data):
        """
        Internal method
        Stack-based DFS implementation of floodfill to group like-colored neighboring pixels
        """
        # constants
        p1 = pixel_data[x1][y1]

        # initialize data structures
        seen = set()
        curr_blob = set()

        def _get_distance(x, y):
            """
            Utility method internal to _flood_fill to calculate RGB euclidean distance
            between a blob's anchor pixel and given pixel
            """
            p2 = pixel_data[x][y]
            d = math.sqrt((int(p1[0]) - int(p2[0])) ** 2 + (int(p1[1]) - int(p2[1])) ** 2 + (int(p1[2]) - int(p2[2])) ** 2)
            return d

        # begin DFS
        seen.add((x1, y1))
        stack = [(x1, y1)]
        while stack:
            x, y = stack.pop()
            # add coords to current blob and map coords to current blob
            curr_blob.add((x, y))
            self.pixel_to_blob[(x, y)] = blob_count

            # iterate through neighbors to check for similarly-colored pixels
            for dx, dy in ImageProcessor.DIRECTIONS:
                # continue if pixel is already assigned to a blob
                if (x+dx, y+dy) in self.pixel_to_blob:
                    continue
                # continue if pixel has already been seen or is out of bounds
                elif (x + dx, y + dy) in seen or not ((0 <= (x + dx) < self.image.height) and (0 <= (y + dy) < self.image.width)):
                        continue
                else:
                    seen.add((x + dx, y + dy))
                    if _get_distance(x + dx, y + dy) <= self.THRESHOLD:
                        stack.append((x + dx, y + dy))
        # once stack is empty, add curr_blob as a Blob object blobs map
        b = Blob(pixels=curr_blob, palette=self.palette)
        b.update_rgb(pixel_data)
        self.blobs[blob_count] = b


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
        # resize images
        self.image = self.image.resize((self.image.width // scale, self.image.height // scale))
        self.original_image = self.original_image.resize(
            (self.original_image.width // scale, self.original_image.height // scale))
        # reset pixel data to conform to new dimensions
        self.pixel_data = np.reshape(np.array(self.image.getdata()), (self.image.height, self.image.width, 3))
        self.pixel_data_limited_palette = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)
        self.pixel_data_blobbed = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)
        self.pixel_data_outline = np.empty((self.image.height, self.image.width, 3), dtype=np.uint64)


    @classmethod
    def make_image(cls, pixel_data):
        array = np.array(pixel_data, dtype=np.uint8)
        new_image = Image.fromarray(array)
        return new_image


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
        # create palette visual for QA purposes
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
