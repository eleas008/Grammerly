from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import config

class ModelRegistry:
    def __init__(self):
        print("Loading BART Summarization Model...")
        self.summary_tokenizer = BartTokenizer.from_pretrained(config.BART_NAME)
        self.summary_model = BartForConditionalGeneration.from_pretrained(config.BART_NAME)
        print("BART Model loaded successfully!")

        print("Loading Grammar Correction Model...")
        self.gec_tokenizer = AutoTokenizer.from_pretrained(config.GEC_MODEL_PATH)
        self.gec_model = AutoModelForSeq2SeqLM.from_pretrained(config.GEC_MODEL_PATH)
        print("GEC Model loaded successfully!")

        print("Loading Paraphrasing Model...")
        self.paraphrasing_tokenizer = AutoTokenizer.from_pretrained(
            config.PARAPHRASING_MODEL_PATH, 
            local_files_only=isinstance(config.PARAPHRASING_MODEL_PATH, type(config.BASE_DIR))
        )
        self.paraphrasing_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.PARAPHRASING_MODEL_PATH, 
            local_files_only=isinstance(config.PARAPHRASING_MODEL_PATH, type(config.BASE_DIR))
        )
        print("Paraphrasing Model loaded successfully!")

models = ModelRegistry()