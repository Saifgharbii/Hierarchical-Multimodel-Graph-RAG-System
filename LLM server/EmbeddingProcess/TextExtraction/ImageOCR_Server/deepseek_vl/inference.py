import torch
from transformers import AutoModelForCausalLM, LlamaTokenizerFast

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image



def load_model(model_path="deepseek-ai/deepseek-vl-1.3b-chat"):
    print("Loading model...")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print("Model Loaded")
    return vl_chat_processor, vl_gpt, tokenizer


def inference(message: str, images_names: list[str], pil_images: list[Image], vl_chat_processor: VLChatProcessor,
              vl_gpt: MultiModalityCausalLM, tokenizer:  LlamaTokenizerFast) -> str:
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{message}",
            "images": images_names,
        },
        {"role": "Assistant", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
