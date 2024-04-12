import imageio
import numpy as np
from PIL import Image

from src.models import load_saliency_detect_model, load_ocr_model, load_llm_model

from src.backend_utils import (
    make_warp_img,
    resize_and_pad,
)

from src.saliency_detect.saliency_infer import run_saliency
from src.kie.kie_infer import run_kie
from src.ocr.ocr_infer import run_ocr

saliency_detect_model = load_saliency_detect_model()
ocr_model = load_ocr_model()
llm_model, tokenizer = load_llm_model()

def infer(img, prompt, ki):
    img = resize_and_pad(img, size=1024, pad=False)

    # saliency detect
    mask_img = run_saliency(saliency_detect_model, img)
    imageio.imwrite("./test_img/mask.jpg", mask_img)

    img[~mask_img.astype(bool)] = 0.0

    # transform and warp image, crop image
    warped_img = make_warp_img(img, mask_img)
    imageio.imwrite('./test_img/crop.jpg', warped_img)

    # ocr
    txt = run_ocr(ocr_model, warped_img)
    print('txt: {}'.format(txt))

    prompt = prompt.format(txt, ki)
    message = []
    message.append({"role": "user", "content": prompt})

    # llm kie
    response = run_kie(llm_model, tokenizer, message)

    return response

if __name__ =='__main__':
    image_path = './test_img/2.jpg'
    image = np.array(Image.open(image_path))

    # Key information to be extracted
    invoice_ki = '单位 电话 车号 证号 日期 上／下车 单价 里程 金额'

    household_ki = '姓名 出生地 籍贯 出生日期 性别 民族 公民身份证件编号 文化程度 婚姻状况 服务处所 职业 登记日期'

    # If you want to use a large model in English, please use english_prompt
    english_prompt = "Your current task is to extract the key information I specified from the OCR text recognition results. " \
                     "The OCR text recognition results are surrounded by symbols, " \
                     "containing the recognized text in order from left to right and from top to bottom in the original image. " \
                     "The key information I specified is surrounded by the [] symbol. " \
                     "Please note that OCR's text recognition results may have issues such as long sentence breaks being cut off, " \
                     "unreasonable word segmentation, and corresponding misalignment. " \
                     "You need to make comprehensive judgments based on contextual semantics to extract accurate key information. " \
                     "Output in JSON format. Here's the official start: OCR text: {}. Key information to be extracted: [{}]."

    chinese_prompt = "你现在的任务是从OCR文字识别的结果中提取我指定的关键信息。OCR的文字识别结果使用符号包围，" \
             "包含所识别出来的文字， 顺序在原始图片中从左至右、从上至下。我指定的关键信息使用[]符号包围。" \
             "请注意OCR的文字识别结果可能存在长句子换行被切断、不合理的分词、 对应错位等问题，" \
             "你需要结合上下文语义进行综合判断，以抽取准确的关键信息。输出为json格式。" \
             "下面正式开始：OCR文字：{}。要抽取的关键信息：[{}]。"

    response_kie = infer(image, chinese_prompt, invoice_ki)



