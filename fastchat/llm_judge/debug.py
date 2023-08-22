import transformers
import torch


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    load_in_half=False,
    gptq_model=False,
    use_fast_tokenizer=False,
    padding_side="left",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, legacy=True)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

        model_wrapper = AutoGPTQForCausalLM.from_quantized(model_name_or_path, device="cuda:0", use_triton=True)
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, load_in_8bit=True)
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                model = model.cuda()
        if load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer


model, tokenizer = load_hf_lm_and_tokenizer(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    load_in_half=True,
)
tid = tokenizer.encode("</s>", add_special_tokens=False)[-1]
print(tid)
tid = tokenizer.encode("<|im_start|>", add_special_tokens=False)[-1]
print(tid)
tid = tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]
print(tid)
tid = tokenizer.eos_token_id
print(tid, tokenizer.eos_token)
