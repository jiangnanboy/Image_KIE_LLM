# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys


sys.path.insert(0, os.path.dirname(os.getcwd()))

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# import sys
#
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
#
# os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import logging

import paddleocr.tools.infer.utility as utility
import paddleocr.tools.infer.predict_rec as predict_rec
import paddleocr.tools.infer.predict_det as predict_det
import paddleocr.tools.infer.predict_cls as predict_cls
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

logger = get_logger()

class OcrModel(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), "elapse"))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res

def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def parse_ocr_opt(det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path):
    args = utility.parse_args()
    args.use_gpu = False
    args.use_onnx = False
    args.enable_mkldnn = True
    args.use_angle_cls = True
    args.use_tensorrt = False
    args.rec_char_dict_path = rec_char_dict_path
    args.det_model_dir = det_model_dir
    args.rec_model_dir = rec_model_dir
    args.cls_model_dir = cls_model_dir
    return args

