import sys
import soundfile as sf
import logging
import json, os

def add_duration(line):
    try:
        data = json.loads(line)
        if "audio_path" in data:
            audio_path = data["audio_path"]
            if os.path.exists(audio_path):
                info = sf.info(audio_path)
                data["duration"] = info.duration
            else:
                logging.warning(f"Audio file not found: {audio_path}")
        else:
            logging.warning("Missing 'audio_path' in record")
        return data
    except Exception as e:
        logging.warning(f"Error processing line: {e}")
        return None

with open(sys.argv[1], "r", encoding="utf-8") as f_in, \
     open(sys.argv[2], "w", encoding="utf-8") as f_out:
    for line in f_in:
        res = add_duration(line)
        if res:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
