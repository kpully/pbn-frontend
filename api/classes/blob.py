class Blob:
    def __init__(self, pixels=set(), palette=None):
        self.pixels = pixels
        self.palette = palette
        self.rgb = [0, 0, 0]

    def get_size(self):
        return len(self.pixels)

    def merge(self, other):
        self.pixels.update(other.pixels)
        # self._update_rgb()

    def update_rgb(self, pixel_data):
        r, g, b = 0, 0, 0
        for pixel_x, pixel_y in list(self.pixels):
            r_, g_, b_ = pixel_data[pixel_y][pixel_x]
            r += r_
            g += g_
            b += b_
        l = len(self.pixels)
        self.rgb = [r / l, g / l, b / l]

        if self.palette:
            self._conform_to_palette()

    def _distance(self, c1, c2):
        """
        Utility method to calculate RGB euclidean distance between anchor pixel and given pixel
        """
        d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
        return d

    def _conform_to_palette(self):
        self.rgb = min(self.palette, key=lambda c: self._distance(self.rgb, c))

    def _neighbor_blobs(self, img, pixel_to_blob):
        """
        Iterate through all pixels of blob and check if neighboring pixel are part of a different blob
        If neighbor pixel does not belong to current blob, add neighbor pixel's blob to neighbor_blobs
        """
        # constants
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        neighbor_blobs = set()  # set of tuples (blob_number, blob_size)
        seen = set()

        for pixel_x, pixel_y in self.pixels:
            for dx, dy in directions:
                if (0 <= pixel_x + dx < img.width) and (0 <= pixel_y + dy < img.height):
                    if (pixel_x + dx, pixel_y + dy) not in seen:
                        seen.add((pixel_x + dx, pixel_y + dy))
                        if (pixel_x + dx, pixel_y + dy) not in self.pixels:
                            neighbor_blob = pixel_to_blob[(pixel_x + dx, pixel_y + dy)]
                            neighbor_blobs.add(neighbor_blob)
        return neighbor_blobs

    def __str__(self):
        return "%s, %d" % (str(self.rgb), self.get_size())
