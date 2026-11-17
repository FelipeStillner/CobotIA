# Student Task:
def pixel_to_robot(px, py, scale=0.5, offset=(0.3, -0.2)):
    X = offset[0] + px * scale / 1000
    Y = offset[1] + py * scale / 1000
    return X, Y

print(pixel_to_robot(200, 150))