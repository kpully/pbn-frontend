class Pixel:
    def __init__(self, coordinates=[0,0], colors=[0,0,0]):
        self.x=coordinates[0]
        self.y=coordinates[1]
        
        self.r=colors[0]
        self.g=colors[1]
        self.b=colors[2]

    def __str__(self):
        return f"({self.x},{self.y})"