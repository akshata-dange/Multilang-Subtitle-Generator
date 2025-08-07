import whisper
import subprocess
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ffmpeg
from IndicTransToolkit.processor import IndicProcessor

#    -- Model Configurations    -- #
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

   
#    TEXT TRANSLATION UTILITIES     
   

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

   
#    AUDIO, SUBTITLES, AND VIDEO     

def extract_audio(input_video_path, output_audio_path):
    """Extracts only the audio track from a video using FFmpeg."""
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-ar', '44100',
        '-ac', '2',
        output_audio_path,
        '-y'
    ]
    subprocess.run(command, check=True)

def write_srt(segments, srt_path):
    """Writes Whisper segments (dicts) to an SRT file in standard subtitle format."""
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def parse_srt(srt_path):
    """Reads an SRT file and returns a list of (index, timestamp, text) tuples."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)", re.DOTALL)
    entries = pattern.findall(content)
    return entries

def rebuild_srt(entries, translated_texts, output_path):
    """Rewrites an SRT file, swapping in translated text, preserving original timing."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (index, timestamp, _) in enumerate(entries):
            f.write(f"{index}\n{timestamp}\n{translated_texts[i]}\n\n")

def convert_srt_to_ass(srt_file_path, ass_file_path):
    """Converts SRT subtitles to ASS (Advanced SubStation Alpha) format using FFmpeg."""
    command = [
        'ffmpeg',
        '-y',
        '-i',
        srt_file_path,
        ass_file_path
    ]
    subprocess.run(command, check=True)
    print(f"Converted {srt_file_path} to {ass_file_path}")

def burn_ass_subtitles_with_font(input_video_path, ass_subtitle_file, output_video_path, fonts_dir=None):
    """Uses ffmpeg-python to burn styled ASS subtitles into a video, with optional font embedding."""
    filters = {}
    if fonts_dir:
        filters["fontsdir"] = fonts_dir
    try:
        (
            ffmpeg.input(input_video_path)
            .filter("ass", ass_subtitle_file, **filters)
            .output(output_video_path, vcodec="libx264", crf=23, acodec="copy")
            .overwrite_output()
            .run(quiet=False)
        )
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        print("FFmpeg error:", err_msg)
        raise e
    print(f"Hard‑burned subtitle video saved at: {output_video_path}")

   
#       MAIN SCRIPT      

if __name__ == "__main__":
    #     Load Whisper model for automatic speech recognition (ASR)
    model = whisper.load_model("base")

    #     Get input video path/name interactively from user
    input_video = input("Enter input video filename or path (e.g., videoplayback.mp4): ").strip()

    #     Get language codes interactively from user    
    source_lang = input("Enter source language code (e.g., en, hi, ms): ").strip()
    target_lang = input("Enter target language code (e.g., hi, en, ms): ").strip()

    #     File paths and font directory configuration    
    output_audio = "extracted_audio.mp3"
    original_srt_path = "original.srt"
    translated_srt_path = "translated.srt"
    translated_ass_path = "translated.ass"
    output_video_path = "output_subtitled.mp4"
    FONTS_DIR = "/home/neosoft/fonts"  # update this path if needed

    #     Audio extraction and transcription    
    print("Extracting audio...")
    extract_audio(input_video, output_audio)

    print("Transcribing audio with Whisper...")
    result = model.transcribe(output_audio)
    print("Transcription completed.")

    #     Convert ASR result to SRT subtitle format    
    write_srt(result["segments"], original_srt_path)

    #     Parse SRT and prepare for translation    
    entries = parse_srt(original_srt_path)
    texts = [entry[2].replace("\n", " ") for entry in entries]

    #     Translation Strategy/Model Selection    
    strategy, models = select_translation_strategy(source_lang, target_lang)
    print(f"Translation strategy: {strategy}")

    #     Translation logic: direct or pivot (via English) as needed    
    if strategy == 'direct':
        if models is None:
            translated_texts = texts  # No translation needed (SRC = TGT)
        else:
            trans_fn = load_translation_pipeline(models, source_lang, target_lang)
            translated_texts = batch_translate(texts, trans_fn, batch_size=4)
    elif strategy == 'pivot_indic_to_malay':
        # First pass: Indic --> English, Second pass: English --> Malay
        trans_fn_1 = load_translation_pipeline(models[0], source_lang, 'en')
        intermediate_texts = batch_translate(texts, trans_fn_1, batch_size=4)
        trans_fn_2 = load_translation_pipeline(models[1], 'en', target_lang)
        translated_texts = batch_translate(intermediate_texts, trans_fn_2, batch_size=4)
    elif strategy == 'pivot_malay_to_indic':
        # First pass: Malay --> English, Second pass: English --> Indic
        trans_fn_1 = load_translation_pipeline(models[0], source_lang, 'en')
        intermediate_texts = batch_translate(texts, trans_fn_1, batch_size=4)
        trans_fn_2 = load_translation_pipeline(models[1], 'en', target_lang)
        translated_texts = batch_translate(intermediate_texts, trans_fn_2, batch_size=4)
    else:
        raise RuntimeError("Unhandled translation strategy.")

    print("Translation completed.")

    #     Write out translated subtitles as SRT file    
    rebuild_srt(entries, translated_texts, translated_srt_path)
    print(f"Saved translated subtitles to {translated_srt_path}")

    #     Convert SRT to ASS subtitle format for advanced styling/font    
    convert_srt_to_ass(translated_srt_path, translated_ass_path)

    #     Select subtitle font for the target language    
    font_to_use = select_font(target_lang)
    print(f"Selected subtitle font: {font_to_use}")

    #     Burn ASS subtitles onto video with proper font embedding    
    burn_ass_subtitles_with_font(
        input_video,
        translated_ass_path,
        output_video_path,
        fonts_dir=FONTS_DIR
    )
    print("All done. Output video:", output_video_path)
