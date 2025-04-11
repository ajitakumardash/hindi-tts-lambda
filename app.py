from flask import Flask, request, send_file, jsonify
from kokoro import KPipeline
from flask_cors import CORS
import numpy as np
import soundfile as sf
import io

app = Flask(__name__)
CORS(app)

pipeline = KPipeline(lang_code='h')
voices = ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi']

@app.route("/")
def index():
    return jsonify({"message": "Hindi TTS API is running."})

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json

    text = data.get("text", "")
    voice = data.get("voice", "hf_alpha")
    speed = data.get("speed", 1.0)

    if not text.strip():
        return jsonify({"error": "Text is required."}), 400

    try:
        generator = pipeline(
            text=text,
            voice=voice,
            speed=float(speed),
            split_pattern=r'[।.!?\n]+'
        )
        audio_chunks = [audio for _, _, audio in generator]

        if not audio_chunks:
            return jsonify({"error": "No audio generated."}), 500

        final_audio = np.concatenate(audio_chunks)

        buf = io.BytesIO()
        sf.write(buf, final_audio, 24000, format='WAV')
        buf.seek(0)

        return send_file(buf, mimetype="audio/wav", download_name="output.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 👇 This part is for AWS Lambda
from mangum import Mangum
handler = Mangum(app)
