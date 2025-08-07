import subprocess
import re
import ffmpeg

# ----- AUDIO, SUBTITLES, AND VIDEO  ------- #

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
