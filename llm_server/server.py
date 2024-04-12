from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
# from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

import uvicorn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Baichuan-13B-Base
model_path = "baichuan-inc/Baichuan2-13B-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(
    model_path
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/stream")
async def create_item_stream(request: Request):
    json_post_raw = await request.json()
    ret = do_stream_chat(json_post_raw)
    return StreamingResponse(ret)


def do_stream_chat(json_post_raw):
    messages = json_post_raw.get('messages')
    for response in model.chat(tokenizer, messages, stream=True):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if response:
            yield response + "\0"


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)