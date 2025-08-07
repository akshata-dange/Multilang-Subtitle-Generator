import whisper
from config_translations import (
    select_translation_strategy,
    load_translation_pipeline,
    batch_translate,
    select_font,
)
from audio_subtitles import (
    extract_audio,
    write_srt,
    parse_srt,
    rebuild_srt,
    convert_srt_to_ass,
    burn_ass_subtitles_with_font,
)

if __name__ == "__main__":
    # Load Whisper model for automatic speech recognition (ASR)
    model = whisper.load_model("base")

    #     Get input video path/name interactively from user
    input_video = input("Enter input video filename or path (e.g., videoplayback.mp4): ").strip()


    #  Get language codes interactively from user 
    source_lang = input("Enter source language code (e.g., en, hi, ms): ").strip()
    target_lang = input("Enter target language code (e.g., hi, en, ms): ").strip()

    #  File paths and font directory configuration 
    output_audio = "extracted_audio.mp3"
    original_srt_path = "original.srt"
    translated_srt_path = "translated.srt"
    translated_ass_path = "translated.ass"
    output_video_path = "output_subtitled.mp4"
    FONTS_DIR = "/home/neosoft/fonts"  # update this path if needed

    #  Audio extraction and transcription 
    print("Extracting audio...")
    extract_audio(input_video, output_audio)

    print("Transcribing audio with Whisper...")
    result = model.transcribe(output_audio)
    print("Transcription completed.")

    #  Convert ASR result to SRT subtitle format 
    write_srt(result["segments"], original_srt_path)

    #  Parse SRT and prepare for translation 
    entries = parse_srt(original_srt_path)
    texts = [entry[2].replace("\n", " ") for entry in entries]

    #  Translation Strategy/Model Selection 
    strategy, models = select_translation_strategy(source_lang, target_lang)
    print(f"Translation strategy: {strategy}")

    #  Translation logic: direct or pivot (via English) as needed 
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

    #  Write out translated subtitles as SRT file 
    rebuild_srt(entries, translated_texts, translated_srt_path)
    print(f"Saved translated subtitles to {translated_srt_path}")

    #  Convert SRT to ASS subtitle format for advanced styling/font 
    convert_srt_to_ass(translated_srt_path, translated_ass_path)

    #  Select subtitle font for the target language 
    font_to_use = select_font(target_lang)
    print(f"Selected subtitle font: {font_to_use}")

    #  Burn ASS subtitles onto video with proper font embedding 
    burn_ass_subtitles_with_font(
        input_video,
        translated_ass_path,
        output_video_path,
        fonts_dir=FONTS_DIR
    )
    print("All done. Output video:", output_video_path)
