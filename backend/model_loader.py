import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import config

class ModelRegistry:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Model System onto Device: {self.device}")


        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)


        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL)

    
        self.model = PeftModel.from_pretrained(self.base_model, config.ADAPTER_PATH)
        self.model.print_trainable_parameters()
        self.model.eval()
        self.model.to(self.device)

        print("Single Multi-Task Model with Adapter Loaded Successfully!")

models = ModelRegistry()