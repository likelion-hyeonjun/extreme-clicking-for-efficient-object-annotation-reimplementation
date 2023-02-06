import json
import pycocotools.coco as cocoapi
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
from matplotlib import pyplot as pyplot
from utils import *
import json
import numpy as np
from pycocotools import mask as cocomask
from skimage import measure
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import parmap
import sys

sys.stdout = open('pseudo_mask_log.txt', 'w')
cpu_count = multiprocessing.cpu_count()
print("number of usable CPU ", cpu_count)

ANN_PATH = '/data02/hyeonjun1882/VOC_dataset/annotations/voc_2012_extreme_train.json'
OUT_PATH = '/data02/hyeonjun1882/VOC_dataset/annotations/voc_2012_pseudo_mask_from_extreme_train.json'
IMG_DIR = '/data02/hyeonjun1882/VOC_dataset/train/'

voc = cocoapi.COCO(ANN_PATH)
EdgeDetector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')



def make_coco_annot_from_binary_mask(psuedo_mask, real_annotation):
    fortran_ground_truth_binary_mask = np.asfortranarray(psuedo_mask)
    encoded_ground_truth = cocomask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = cocomask.area(encoded_ground_truth)
    ground_truth_bounding_box = cocomask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(psuedo_mask, 0.5)
    
    annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": real_annotation['image_id'],
        "bbox": real_annotation['bbox'],
        "category_id": real_annotation['category_id'],
        "id": real_annotation['id']
    }
    
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    
    return annotation

def get_pseudo_mask_from_bbox(img, annot):

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = list(map(int,annot['bbox']))
    #Grabcut can handle when bounding box cover whole image.
    if img.shape[0] == rect[3] and img.shape[1] == rect[2]:
        rect[0] = rect[0] +1

    rect = tuple(rect)
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    pseudo_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    return pseudo_mask

def get_pseudo_mask_from_extreme_point(img, annot, edge_detector):
    #FIND EDGE
    #edge_detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = edge_detector.detectEdges(np.float32(rgb_im)/255.0)

    #GET INITIAL SURFACE
    initial_surface, path_pixels = find_surface_mask_via_crop(img, annot, 1-im) #mask has initial binary mask 
    #GET SKELETON
    skeleton = get_skeleton(initial_surface)
    #GET BACKGROUND
    bbox = get_bbox_format_from_extreme_point(annot['extreme_points'])
    larger_box = get_double_size_bounding_box(bbox)
    background = draw_ring(larger_box, bbox, img.shape[:2])

    mask = np.zeros(img.shape[:2],np.uint8)
    mask[initial_surface== 255] = cv2.GC_PR_FGD
    mask[skeleton == 255] = cv2.GC_FGD
    mask[background ==255] = cv2.GC_BGD

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    pseudo_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    return pseudo_mask


def pseudo_mask_coco_json_from_real_annot(annot):
    img_id_from_annot = annot['image_id']
    Img = voc.loadImgs(img_id_from_annot)[0]
    voc_img_path = IMG_DIR +Img['file_name']
    img = cv2.imread(voc_img_path)
    
    try:
        pseudo_mask = get_pseudo_mask_from_extreme_point(img, annot, EdgeDetector)
        print("ExtremePoint!")
    except TimeoutException as e:
        pseudo_mask = get_pseudo_mask_from_bbox(img, annot)
        print("BOX!")

    pseudo_mask_annot = make_coco_annot_from_binary_mask(pseudo_mask, annot)
    return pseudo_mask_annot

if __name__=="__main__":
    pseudo_mask_annotations_list = []

    with open(ANN_PATH, "r") as f:
        extreme_data_json = json.load(f)

    original_annotations = extreme_data_json['annotations']
    pool = multiprocessing.Pool(processes=cpu_count)   
    
    pseudo_mask_annotations_list = parmap.map(pseudo_mask_coco_json_from_real_annot, original_annotations, pm_pbar=True, pm_processes=cpu_count)
    pseudo_mask_from_extreme_point_json = {'images':extreme_data_json['images'],'annotations':pseudo_mask_annotations_list,'categories':extreme_data_json['categories']}


    with open(OUT_PATH, 'w') as outfile:
        json.dump(pseudo_mask_from_extreme_point_json, outfile)
    
    sys.stdout.close()