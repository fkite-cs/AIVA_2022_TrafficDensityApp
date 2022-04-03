class Vehicle():

    def __init__(self, global_hw, y, x, dy, dx) -> None:
        self.global_hw = global_hw
        self.y = y
        self.x = x
        self.dy = dy
        self.dx = dx

    def get_bbox(self):
        bbox = [self.y, self.x, self.dy, self.dx]
        return bbox

    def get_global_coordinates(self):
        return (self.global_hw[0] + self.y, self.global_hw[1] + self.x)


class BBox():

    def __init__(self, x, y, dx, dy) -> None:
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy