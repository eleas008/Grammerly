import re
from spellchecker import SpellChecker

spell = SpellChecker()

def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return text

    deduped_text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

    tokens = re.findall(r'\w+|[^\w\s]|\s+', deduped_text)
    corrected_tokens = []

    for i, token in enumerate(tokens):
        if token.isalpha():
            is_capitalized = token[0].isupper()

            if is_capitalized and i > 0:
                corrected_tokens.append(token)
                continue

            corrected_word = spell.correction(token.lower())
            
            if corrected_word and corrected_word != token.lower():
                if token.isupper():
                    corrected_word = corrected_word.upper()
                elif token[0].isupper():
                    corrected_word = corrected_word.capitalize()
                corrected_tokens.append(corrected_word)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    processed_text = "".join(corrected_tokens)

    capitalized_text = re.sub(
        r'(?:^|(?<=[.!?]\s))([a-z])',
        lambda match: match.group(1).upper(),
        processed_text
    )

    return capitalized_text