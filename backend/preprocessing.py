import re
from importlib.resources import files
import torch
import nltk
from nltk import pos_tag, word_tokenize
from symspellpy import SymSpell, Verbosity
from model_loader import models


REQUIRED_NLTK_PACKAGES = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for package in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}')
    except LookupError:
        nltk.download(package, quiet=True)


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = str(files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt"))
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def smart_spell_check(text: str) -> str:
    if not text.strip():
        return text

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    corrected_words = []

    for word, tag in tagged_tokens:
        if tag in ['NNP', 'NNPS'] or (word[0].isupper() and word.isalpha()):
            corrected_words.append(word)
        elif word.isalpha():
            suggestions = sym_spell.lookup(
                word.lower(), 
                verbosity=Verbosity.CLOSEST, 
                max_edit_distance=2
            )
            if suggestions:
                corrected_word = suggestions[0].term
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    corrected_text = " ".join(corrected_words)
    corrected_text = re.sub(r'\s+([.,!?])', r'\1', corrected_text)
    return corrected_text


def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return text

    deduped_text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

    spell_cleaned_text = smart_spell_check(deduped_text)

    prompt = "fix spelling and grammar: " + spell_cleaned_text.strip()
    
    inputs = models.tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(models.device)

    with torch.no_grad():
        with models.model.disable_adapter():
            outputs = models.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                do_sample=False,
                early_stopping=True
            )

    corrected_text = models.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    corrected_text = re.sub(r'\s+([.,!?])', r'\1', corrected_text)
    corrected_text = re.sub(r'\s+', ' ', corrected_text)

    return corrected_text