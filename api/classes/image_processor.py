from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import cv2
from sklearn.cluster import KMeans

# from classes.blob import Blob


class ImageProcessor:
    def __init__(self, img_name, THRESHOLD=50, complex_palette=False):
        self.img_name = img_name
        self.original_img = Image.open(img_name).convert("RGB")
        self.processed_img = None
        self.img = Image.open(img_name).convert("RGB")
        self.THRESHOLD = THRESHOLD
        self.pixel_data = np.reshape(np.array(self.img.getdata()), (self.img.height, self.img.width, 3))

        # initialize empty data structures
        self.blobs = {}  # map of integer keys mapping to Blob object
        self.pixel_to_blob = {}  # map of pixel coordinate keys to key in self.blobs
        self.palette = None

    def process_image(self):
        """
        Create blobs with similarly-colored pixels by calling floodfill
        """
        for i in range(self.img.width):
            for j in range(self.img.height):
                if (i, j) not in self.pixel_to_blob:
                    blob_count = len(self.blobs)
                    self._flood_fill(i, j, blob_count)

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
                    neighbor_blobs = small_blob._neighbor_blobs(self.img, self.pixel_to_blob)
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
        self.img = self.img.resize((self.img.width // scale, self.img.height // scale))
        self.original_img = self.original_img.resize(
            (self.original_img.width // scale, self.original_img.height // scale))

    def _flood_fill(self, x1, y1, blob_count):
        """
        Internal method
        Stack-based DFS implementation of floodfill to group like-colored neighboring pixels
        """
        # constants
        p1 = self.img.getpixel((x1, y1))
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        # initialize data structures
        seen = set()
        curr_blob = set()

        def _get_distance(x, y):
            """
            Utility method internal to _flood_fill to calculate RGB euclidean distance
            between a blob's anchor pixel and given pixel
            """
            p2 = self.img.getpixel((x, y))
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
                    if (0 <= x + dx < self.img.width) and (0 <= y + dy < self.img.height):
                        if _get_distance(x + dx, y + dy) <= self.THRESHOLD:
                            stack.append((x + dx, y + dy))
        # once stack is empty, add curr_blob as a Blob object blobs map
        b = Blob(pixels=curr_blob, palette=self.palette)
        b.update_rgb(self.pixel_data)
        self.blobs[blob_count] = b

    def make_image(self):
        new_pixels = []
        for j in range(self.img.height):
            row = []
            for i in range(self.img.width):
                b = self.pixel_to_blob[(i, j)]
                bl = self.blobs[b]
                row.append(bl.rgb)
            new_pixels.append(row)
        array = np.array(new_pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        self.processed_img = new_image

    def _get_dominant_colors(self, clusters):
        img = cv2.imread(self.img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(img)
        colors = [[int(color) for color in cluster] for cluster in kmeans.cluster_centers_]
        self.palette = colors
        palette = Image.new("RGB", (500, 50), (255, 255, 255))
        draw = ImageDraw.Draw(palette)
        i = 0
        radius = 20
        spacing = 25
        for color in colors:
            draw.ellipse(xy=[0 + i * spacing, 0, radius + i * spacing, radius], fill=tuple(color))
            i += 1
        palette.show()
