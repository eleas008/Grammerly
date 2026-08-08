import re
import torch
from model_loader import models


def is_conversation(text: str) -> bool:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    speaker_turn_pattern = re.compile(r"^[A-Z][a-zA-B0-9_\s]{1,15}:")
    speaker_lines_count = sum(1 for line in lines if speaker_turn_pattern.match(line))
    return speaker_lines_count >= 2 or (speaker_lines_count / len(lines)) > 0.2


def generate_summary(text: str) -> str:
    if not text.strip():
        return "Please provide input text to summarize."

    is_dialogue = is_conversation(text)

    if is_dialogue:
        print(" -> Detected Dialogue: Running WITH LoRA Adapter")
        prompt = "summarize: " + text.strip()
        adapter_context = torch.no_grad()
    else:
        print(" -> Detected Normal Paragraph: Disabling LoRA Adapter (Using Base Model)")
        prompt = "summarize the following text into 5 distinct key points:\n" + text.strip()
        adapter_context = models.model.disable_adapter()

    inputs = models.tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(models.device)

    with torch.no_grad():
        with adapter_context:
            if is_dialogue:
                outputs = models.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            else:
                outputs = models.model.generate(
                    **inputs,
                    max_length=200,
                    min_length=60,
                    num_beams=4,
                    length_penalty=2.0,
                    encoder_no_repeat_ngram_size=3,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

    raw_output = models.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not is_dialogue:
        lines = re.split(r'(?<=\.)\s+|(?=\b\d+\.\s*)', raw_output)
        lines = [l.strip() for l in lines if l.strip()]
        if len(lines) >= 5:
            return "\n".join(lines[:5])
        return "\n".join(lines)

    return raw_output


def correct_grammar(text: str) -> str:
    if not text.strip():
        return ""

    prompt = f"Fix spelling and grammar errors: {text.strip()}"

    inputs = models.tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(models.device)

    with torch.no_grad():
        outputs = models.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            do_sample=False,       
            early_stopping=True
        )

    result = models.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    result = re.sub(r'\s+([.,!?])', r'\1', result)
    
    return result

def paraphrase_text(text: str, num_outputs: int = 3) -> str:
    if not text.strip():
        return "Please provide input text to paraphrase."

    prompt = "paraphrase: " + text.strip()
    inputs = models.tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(models.device)

    with torch.no_grad():
        outputs = models.model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            num_return_sequences=num_outputs,
            no_repeat_ngram_size=2
        )

    suggestions = [
        models.tokenizer.decode(g, skip_special_tokens=True).strip()
        for g in outputs
    ]

    return "\n\n".join(
        [f"Option {i+1}:\n{s}" for i, s in enumerate(suggestions)]
    )