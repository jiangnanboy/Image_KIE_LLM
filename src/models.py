import torch
import configs as cf
from src.saliency_detect.u2net import U2NETP
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from src.ocr.ocr_model import parse_ocr_opt, OcrModel

def load_ocr_model():
    print('load ocr model ...')
    ocr_opt = parse_ocr_opt(cf.ocr_model_det_pp_path, cf.ocr_model_cls_pp_path, cf.ocr_model_rec_pp_path, cf.ocr_model_char_pp_path)
    ocr_model = OcrModel(ocr_opt)
    return ocr_model

def load_saliency_detect_model():
    print('load saliency detect model...')
    net = U2NETP(3, 1)
    net = net.to(cf.device)
    net.load_state_dict(torch.load(cf.saliency_weight_path, map_location=cf.device))
    net.eval()
    return net

def load_llm_model():
    print('load llm model...')
    tokenizer = AutoTokenizer.from_pretrained(cf.llm_path,
                                              revision="v2.0",
                                              use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cf.llm_path,
                                                 revision="v2.0",
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(cf.llm_path, revision="v2.0")
    model.eval()
    return model, tokenizer