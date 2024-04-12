import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_dir = "./images"
result_img_dir = "./results/model"
raw_img_dir = "./results/raw"
cropped_img_dir = "./results/crop"

ocr_model_det_pp_path = './weights/ocr/ch_PP-OCRv3_det_infer'
ocr_model_rec_pp_path = './weights/ocr/ch_PP-OCRv3_rec_infer'
ocr_model_cls_pp_path = './weights/ocr/ch_ppocr_mobile_v2.0_cls_infer'
ocr_model_char_pp_path = './weights/ocr/ppocr_keys_v1.txt'
font_path = './font/STFANGSO.TTF'

saliency_weight_path = "./weights/saliency_detect/u2netp.pth"

llm_path = "baichuan-inc/Baichuan2-13B-Chat"

saliency_ths = 0.5
