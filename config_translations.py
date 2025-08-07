import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import ffmpeg
import subprocess
import re


#   Model Configurations 


# Define model names for AI4Bharat IndicTrans2 and Facebook M2M100 (used in pipeline logic)
ai4bharat_models = {
    'eng_to_indic'  : 'ai4bharat/indictrans2-en-indic-1B',
    'indic_to_eng'  : 'ai4bharat/indictrans2-indic-en-1B',
    'indic_to_indic': 'ai4bharat/indictrans2-indic-indic-1B'
}
facebook_m2m100_model = 'facebook/m2m100_418M'

# Map ISO language codes to logical groups and model-compatible forms
language_group_map = {
    'en': 'eng',
    'hi': 'indic', 'mr': 'indic', 'bn': 'indic', 'ta': 'indic', 'ml': 'indic', 'te': 'indic', 'kn': 'indic', 'ms': 'malay'
}
# FLORES language codes for IndicTrans2
flores_lang_map = {
    'en': 'eng_Latn',
    'hi': 'hin_Deva',
    'mr': 'mar_Deva',
    'bn': 'ben_Beng',
    'ta': 'tam_Taml',
    'ml': 'mal_Mlym',
    'te': 'tel_Telu',
    'kn': 'kan_Knda',
    'ms': 'msa_Latn',  # Malay
}
# M2M100-compatible language codes
m2m100_lang_map = {
    'en': 'en',
    'hi': 'hi',
    'mr': 'mr',
    'bn': 'bn',
    'ta': 'ta',
    'ml': 'ml',
    'te': 'te',
    'kn': 'kn',
    'ms': 'ms',
}
# Default fonts for each language (for subtitle overlay)
subtitle_fonts = {
    'en': 'Arial',
    'hi': 'Noto Sans Devanagari',
    'bn': 'Noto Sans Bengali',
    'mr': 'Noto Sans Devanagari',
    'ta': 'Noto Sans Tamil',
    'ml': 'Noto Sans Malayalam',
    'te': 'Noto Sans Telugu',
    'kn': 'Noto Sans Kannada',
    'ms': 'Noto Sans Malay',
}

# Set device to GPU (cuda) if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#  TEXT TRANSLATION UTILITIES 


def batch_translate(texts, translate_fn, batch_size=4):
    """Translate a list of texts in batches to save RAM and avoid OOM errors."""
    result = []
    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i:i + batch_size]
        result.extend(translate_fn(sub_batch))
    return result

def select_translation_strategy(src_lang, tgt_lang):
    """Choose translation strategy and necessary model(s) based on language pairing."""
    src_group = language_group_map.get(src_lang)
    tgt_group = language_group_map.get(tgt_lang)
    if not src_group or not tgt_group:
        raise ValueError(f"Unsupported language code: {src_lang} or {tgt_lang}")
    if src_group == 'indic' and tgt_group == 'indic':
        return 'direct', ai4bharat_models['indic_to_indic']
    if src_group == 'eng' and tgt_group == 'indic':
        return 'direct', ai4bharat_models['eng_to_indic']
    if src_group == 'indic' and tgt_group == 'eng':
        return 'direct', ai4bharat_models['indic_to_eng']
    if (src_group == 'eng' and tgt_group == 'malay') or \
       (src_group == 'malay' and tgt_group == 'eng') or \
       (src_group == 'malay' and tgt_group == 'malay'):
        return 'direct', facebook_m2m100_model
    if src_group == 'eng' and tgt_group == 'eng':
        return 'direct', None
    if src_group == 'indic' and tgt_group == 'malay':
        return 'pivot_indic_to_malay', (ai4bharat_models['indic_to_eng'], facebook_m2m100_model)
    if src_group == 'malay' and tgt_group == 'indic':
        return 'pivot_malay_to_indic', (facebook_m2m100_model, ai4bharat_models['eng_to_indic'])
    raise ValueError(f"No supported strategy for translation from {src_lang} to {tgt_lang}")

def select_font(lang_code):
    """Returns the recommended font for a given language code; for subtitle overlay."""
    return subtitle_fonts.get(lang_code, 'Arial')

def ai4bharat_translate_batch(sentences, src_lang, tgt_lang, model_name):
    """Translates a batch of sentences using an AI4Bharat IndicTrans2 model (local or HuggingFace)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    ip = IndicProcessor(inference=True)
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translations

def load_m2m100_pipeline(model_name, src_lang, tgt_lang):
    """Returns a function to translate a list of texts with Facebook's M2M100 model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    tokenizer.src_lang = m2m100_lang_map[src_lang]
    def translate(texts):
        results = []
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(m2m100_lang_map[tgt_lang])
            )
            decoded = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            results.append(decoded)
        return results
    return translate

def load_translation_pipeline(model_name, src_lang, tgt_lang):
    """Returns the correct translation function after resolving language code formats."""
    if 'ai4bharat/indictrans2' in model_name:
        return lambda texts: ai4bharat_translate_batch(
            texts, flores_lang_map[src_lang], flores_lang_map[tgt_lang], model_name)
    elif 'facebook/m2m100' in model_name:
        return load_m2m100_pipeline(model_name, m2m100_lang_map[src_lang], m2m100_lang_map[tgt_lang])
    else:
        raise ValueError(f"Unsupported model {model_name}")
