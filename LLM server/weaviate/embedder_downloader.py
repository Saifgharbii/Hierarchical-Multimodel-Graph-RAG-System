from transformers import AutoTokenizer, AutoModel

CONFIG = {
    'model_name': 'dunzhang/stella_en_400M_v5'
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
model = AutoModel.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
