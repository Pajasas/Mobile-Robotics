def subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad

    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])


    return cv2.warpAffine(
        image,
        mapping,
        (int(width), int(height)),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)

def getMinRect(image, points):
    rect = cv2.minAreaRect(points)
    centre, (width_r, height_r), theta = rect
    return subimage(image, centre, theta, width_r, height_r)