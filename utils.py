import json
import pycocotools.coco as cocoapi
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
from matplotlib import pyplot as pyplot
import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



def shortest_path(start, end, edge_response):
    """
    Finds the shortest path between two points in a given 2D image
    with edge response as the edge weight.

    Parameters:
    start (tuple): The start point coordinates (row, col).
    end (tuple): The end point coordinates (row, col).
    edge_response (np.ndarray): The edge response map of the same size as the original image.

    Returns:
    np.ndarray: A binary mask indicating the pixels in the shortest path.
    """
    rows, cols = edge_response.shape
    heap = [(0, start)]
    visited = np.zeros(edge_response.shape, dtype=np.bool)
    parent = {}

    while heap:
        (cost, curr) = heap.pop(0)
        if curr == end:
            break

        if visited[curr]:
            continue

        visited[curr] = True

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = curr[0] + dx, curr[1] + dy
            if x < 0 or x >= rows or y < 0 or y >= cols:
                continue
            new_cost = cost + edge_response[x, y]
            new_point = (x, y)
            if new_point not in parent or new_cost < parent[new_point][0]:
                parent[new_point] = (new_cost, curr)
                heap.append((new_cost, new_point))
                heap = sorted(heap, key=lambda x: x[0])

    path = []
    curr = end
    check_inifinite_loop = 0 
    while curr != start:
        check_inifinite_loop = check_inifinite_loop +1
        path.append(list(curr))
        curr = parent[curr][1]
        
        if check_inifinite_loop > 2500 :
            raise TimeoutException("Timed out by Infinite Loop")
            break
    path = path[::-1]

    return path

def find_surface_mask(img, annot, edge_prob):
    # Apply edge detector to obtain boundary probability
    # Find 4 paths connecting consecutive extreme points
    paths = []
    extreme_points = annot['extreme_points']
    for i in range(4):
        start, end = extreme_points[i], extreme_points[(i+1)%4]
        path = shortest_path(tuple(start), tuple(end), np.transpose(edge_prob))
        paths.extend(path)

    boundary_points = np.array(paths, dtype=np.int32)
    boundary_points = boundary_points.reshape(-1,1,2)
    binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(binary_mask, [boundary_points], 255)
    return binary_mask, paths

def EP_error_check(extreme_points, img_shape): #extreme point가 img_shape 가 x-640 , y -434 인데, ep 좌표가 (a,434) 이런식으로 있는경우 대비
    if extreme_points[3][0] == img_shape[1]:
        extreme_points[3][0] = extreme_points[3][0]-1
    if extreme_points[2][1] == img_shape[0]:
        extreme_points[2][1] = extreme_points[2][1]-1
    return extreme_points

def find_surface_mask_via_crop(img, annot, whole_edge_prob_map):
    """
    Parameters:
    img: original image.
    annot: which contians 4 extreme points([t,t],[l,l],[b,b],[r,r]) / bbox (x,y,width,height)
    whole_edge_response (np.ndarray): The edge response map of the same size as the original image.

    1. crop the edge prob map using bounding box coordinate.
    2. find shortes path whose minimum edge response is hegihest within the cropped region

    Returns:
    np.ndarray: A binary mask indicating the pixels in the shortest path.
    """

    paths = []
    #bx, by, bwidth, bheight = list(map(int,annot['bbox']))    
    tt,ll,bb,rr = EP_error_check(annot['extreme_points'], img.shape)
    bheight = bb[1]-tt[1]
    bwidth = rr[0]-ll[0]
    bx = ll[0]
    by = tt[1]
    max_y, max_x = whole_edge_prob_map.shape

    from_x = max(0,bx-1)
    from_y = max(0,by-1)
    to_x = min(max_x,bx+bwidth+3)
    to_y = min(max_y,by+bheight+3)

    cropped_edge_prob_map = whole_edge_prob_map[from_y:to_y, from_x:to_x] #could make an error when bbox is tighted to the bottom
   
    for i in range(4):
        extreme_points = EP_error_check(annot['extreme_points'], img.shape)
        start, end = extreme_points[i], extreme_points[(i+1)%4]
        relocated_start = np.subtract(start,[from_x,from_y])
        reloacted_end = np.subtract(end,[from_x,from_y])
        path = shortest_path(tuple(relocated_start), tuple(reloacted_end), np.transpose(cropped_edge_prob_map))
        paths.extend(path)

    paths = list(map(lambda x: np.add(x, [from_x,from_y]), paths))

    boundary_points = np.array(paths, dtype=np.int32)
    boundary_points = boundary_points.reshape(-1,1,2)
    binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(binary_mask, [boundary_points], color=(255))
    return binary_mask, paths

# To make skeleton
def get_skeleton(binary_mask):
    # Get the morphological skeleton using cv2.morphologyEx
    skeleton = np.zeros(binary_mask.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(binary_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_mask, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_mask = eroded.copy()

        zeros = np.count_nonzero(binary_mask)
        if zeros == 0:
            done = True

    return skeleton
# To make Background model
def get_bbox_format_from_extreme_point(EP):
    tt,ll,bb,rr = EP
    return ll[0], tt[1], rr[0]-ll[0]+1, bb[1]-tt[1]+1

def get_double_size_bounding_box(bbox):
    x, y, width, height = bbox
    new_width = width * np.sqrt(2)
    new_height = height * np.sqrt(2) 
    new_x = x - (new_width - width) / 2
    new_y = y - (new_height - height) / 2
    return [int(new_x), int(new_y), int(new_width), int(new_height)]

def draw_ring(larger_box, original_box, img_shape):
    x, y, width, height = original_box
    new_x, new_y, new_width, new_height =larger_box
    x_end = min(img_shape[1], new_x + new_width)
    y_end = min(img_shape[0], new_y + new_height)
    x_start = max(0, new_x)
    y_start = max(0, new_y)
    
    binary_mask = np.zeros(img_shape,dtype = np.uint8)
    binary_mask[y_start:y_end, x_start:x_end] = 255
    binary_mask[y:y+height, x:x+width]=0
    
    return binary_mask
