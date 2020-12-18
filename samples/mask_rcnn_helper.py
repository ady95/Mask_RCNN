
import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2
import matplotlib
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath("../")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNHelper:

    CLASSID_CAR = 3

    def __init__(self, model_path):
        config = InferenceConfig()
        # config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(model_path, by_name=True)

    def detect(self, image):
        results = self.model.detect([image], verbose=0)
        return results[0]

    # 리턴되는 결과중 가장 대각선 길이가 큰 차량 1대를 리턴함
    def detect_car_one(self, image):
        result = self.detect(image)
        
        class_ids = result['class_ids']
        scores = result['scores']
        rois = result['rois']
        masks = result['masks']
        # mask = masks[:, :, 0]

        result_list = []
        for i, class_id in enumerate(class_ids):
            if class_id != self.CLASSID_CAR:
                continue

            y1, x1, y2, x2 = rois[i]
            box_size = math.sqrt((x2-x1) **2 + (y2-y1) **2) # ROI 대각선 길이

            car_result = {
                "class_id": class_id,
                "score": scores[i],
                "mask": masks[:, :, i],
                "roi": rois[i],
                "roi_size": box_size,
            }
            result_list.append(car_result)

        if len(result_list) == 0:
            return None

        # 4. 박스의 넓이가 큰 순으로 정렬 (제일 큰게 맨 마지막으로 있어야 라벨이 겹치지 않음)
        result_list = sorted(result_list, key=lambda x: x["roi_size"], reverse=True)

        return result_list[0]

    def get_masked_image(self, image, mask, crop_box = None):
        
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c],
                                      image[:, :, c]*0)

        if type(crop_box) == np.ndarray:
            y1, x1, y2, x2 = crop_box
            image = image[y1:y2, x1:x2]

        return image


def read_hangul_path_file( file_path ) :

    stream = open( file_path.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    np_array = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(np_array , cv2.IMREAD_UNCHANGED)

if __name__ == "__main__":
    import time

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images2")
    # IMAGE_DIR = r"D:\DATA\@car\car_photo\carphoto_20200518"
    # IMAGE_DIR = r"D:\DATA\@car\car_photo\carphoto_20190609"
    # IMAGE_DIR = r"D:\DATA\@car\car_photo\carphoto_20190612"
    # IMAGE_DIR = r"E:\DATA\@car2\brand\encar_20201107\현대_팰리세이드_팰리세이드"

    OUTPUT_FOLDER = r"D:\GIT_AI\Mask_RCNN_ady95\output"

    helper = MaskRCNNHelper(COCO_MODEL_PATH)

    file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    for file_name in file_names[:]:
        if not file_name.endswith(".jpg"):
            continue

        # image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
        # image = cv2.imread(os.path.join(IMAGE_DIR, file_name))
        image = read_hangul_path_file(os.path.join(IMAGE_DIR, file_name))

        s1 = time.time()

        # Run detection
        result = helper.detect_car_one(image)
        if result == None:
            continue

        mask = result["mask"]
        roi_box = result["roi"]
        masked_image = helper.get_masked_image(image, mask, roi_box)

        interval = time.time() - s1

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, file_name), masked_image)

        print(file_name, interval, result)
#         cv2.imshow('', masked_image)
#         cv2.waitKey(0)

# cv2.destroyAllWindows()