from flask import Flask, request, jsonify
import torch
from vits import utils
from vits.models import SynthesizerTrn
import commons
import json

app = Flask(__name__)

# Load model VITS
device = "cuda" if torch.cuda.is_available() else "cpu"
hps = utils.get_hparams_from_file("configs/config.json")

model = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).to(device)

utils.load_checkpoint("logs/G_latest.pth", model, None)

@app.route("/")
def home():
    return "VITS TTS API is running!"

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Chuyển văn bản thành giọng nói
        stn_tst = utils.get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            audio = model.infer(x_tst, None, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]

        # Lưu file WAV
        output_path = "output.wav"
        utils.save_wav(audio.cpu().numpy(), output_path, hps.data.sampling_rate)

        return jsonify({"message": "TTS Success!", "output": output_path})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
