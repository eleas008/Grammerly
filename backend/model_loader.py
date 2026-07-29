from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import config

class ModelRegistry:
    """Singleton-style class to load and hold NLP models in memory."""
    def __init__(self):
        print("Loading BART Summarization Model...")
        self.summary_tokenizer = BartTokenizer.from_pretrained(config.BART_NAME)
        self.summary_model = BartForConditionalGeneration.from_pretrained(config.BART_NAME)
        print("BART Model loaded successfully!")

        print("Loading Local Grammar Correction Model...")
        self.gec_tokenizer = AutoTokenizer.from_pretrained(config.GEC_MODEL_PATH, local_files_only=True)
        self.gec_model = AutoModelForSeq2SeqLM.from_pretrained(config.GEC_MODEL_PATH, local_files_only=True)
        print("Local GEC Model loaded successfully!")

        print("Loading Local Paraphrasing Model...")
        self.paraphrasing_tokenizer = AutoTokenizer.from_pretrained(config.PARAPHRASING_MODEL_PATH, local_files_only=True)
        self.paraphrasing_model = AutoModelForSeq2SeqLM.from_pretrained(config.PARAPHRASING_MODEL_PATH, local_files_only=True)
        print("Local Paraphrasing Model loaded successfully!")

models = ModelRegistry()