import copy
import time
import json
import itertools
from functools import wraps

import cv2
import numpy as np

class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/a/57915246"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta_time = round(float(end - start), 8)
        print("- Function {} run in {}'s".format(func.__name__, delta_time))
        return result

    return wrapper

def resize_and_pad(img, size=1024, pad=False, value=0):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        if value == 0:
            new_img = np.zeros((size, size, c), img.dtype)
        else:
            new_img = np.ones((size, size, c), img.dtype) * value
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def create_poly_from_polys(cells):
    x1, y1, *_, x4, y4 = cells[0]["poly"]
    _, _, x2, y2, x3, y3, _, _ = cells[-1]["poly"]
    poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    return poly


def make_pad32_img(img):
    target_h, target_w, channel = img.shape
    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = img

    return resized


def update_field_ids(cells, new_polys):
    new_cells = []
    assert len(cells) == len(new_polys)
    for cell_index, cell in enumerate(cells):
        new_cell = copy.deepcopy(cell)
        new_cell["poly"] = new_polys[cell_index]
        new_cells.append(new_cell)

    return new_cells


def make_crop_img(img):
    tmp = img != 0
    x_min = np.where(np.sum(tmp, axis=(0, 2)) != 0)[0][0]
    x_max = np.where(np.sum(tmp, axis=(0, 2)) != 0)[0][-1]
    y_min = np.where(np.sum(tmp, axis=(1, 2)) != 0)[0][0]
    y_max = np.where(np.sum(tmp, axis=(1, 2)) != 0)[0][-1]
    crop_bbox = (x_min, y_min, x_max, y_max)
    sub_img = img[y_min:y_max, x_min:x_max]
    return sub_img, crop_bbox


def update_coord_crop_img(cells, crop_bbox):
    x_min, y_min, _, _ = crop_bbox
    crop_cells = copy.deepcopy(cells)
    for cell in crop_cells:
        poly = cell["poly"]
        xs, ys = poly[::2], poly[1::2]
        xs = [x - x_min for x in xs]
        ys = [y - y_min for y in ys]
        new_poly = list(itertools.chain(*zip(xs, ys)))
        cell["poly"] = np.array(new_poly)

    return crop_cells


def get_largest_poly_with_coord(mask_img, reshape=False):
    # mask_img: gray img
    _, mask_img = cv2.threshold(mask_img, 127, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    mask_img = cv2.dilate(mask_img, kernel, iterations=3)
    contours = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    poly = np.int0(cv2.boxPoints(rect))
    point = np.array(poly).reshape(-1, 2).astype(int)

    return point


def get_max_hw(point):
    rect = order_points(np.array(point.squeeze()))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight


def get_transform_matrix(point, maxWidth, maxHeight):
    pts1 = order_points(point.squeeze().copy().astype(np.float32))
    pts2 = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix


@timer
def make_warp_img(img, mask_img):
    warped_point = get_largest_poly_with_coord(mask_img)
    maxWidth, maxHeight = get_max_hw(warped_point)
    matrix = get_transform_matrix(warped_point, maxWidth, maxHeight)
    warped_img = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    return warped_img


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped, M


