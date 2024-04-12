from src.backend_utils import timer

@timer
def run_ocr(ocr_model, warped_img):
    dt_boxes, rec_res = ocr_model(warped_img)
    txt = [text[0] for text in rec_res]
    txt = '\n'.join(txt)
    return txt
