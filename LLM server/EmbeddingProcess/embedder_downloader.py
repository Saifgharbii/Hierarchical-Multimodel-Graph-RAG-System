from transformers import AutoTokenizer, AutoModel

model_name = "dunzhang/stella_en_400M_v5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)