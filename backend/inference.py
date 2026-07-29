from model_loader import models

def generate_summary(text: str) -> str:
    if not text.strip():
        return "Please provide input text to summarize."

    word_count = len(text.split())
    max_len = min(140, max(20, int(word_count * 0.6)))
    min_len = min(30, max(5, int(word_count * 0.2)))

    inputs = models.summary_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    summary_ids = models.summary_model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        early_stopping=True
    )
    return models.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def correct_grammar(text: str) -> str:
    if not text.strip():
        return "Please provide input text to correct."

    inputs = models.gec_tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    corrected_ids = models.gec_model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    return models.gec_tokenizer.decode(corrected_ids[0], skip_special_tokens=True)


def paraphrase_text(text: str) -> str:
    if not text.strip():
        return "Please provide input text to paraphrase."

    inputs = models.paraphrasing_tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    para_ids = models.paraphrasing_model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=5,
        num_return_sequences=3,
        early_stopping=True
    )

    suggestions = [
        models.paraphrasing_tokenizer.decode(g, skip_special_tokens=True) 
        for g in para_ids
    ]

    return "\n\n".join(
        [f"Option {i+1}:\n{suggestion}" for i, suggestion in enumerate(suggestions)]
    )