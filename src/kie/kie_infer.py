from src.backend_utils import timer

@timer
def run_kie(model, tokenizer, messages):
    response = model.chat(tokenizer, messages)
    return response
