from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from .utils import decode_base64_to_cv2
from .detector import FaceAnalyzer
from .config import HOST, PORT

app = Flask(__name__)
CORS(app) 

def warm_up_models():
    """Pre-loads AI models into RAM so the first scan is fast."""
    print(">>> AI Engine: Warming up Neural Networks... (Please wait)")
    try:
        # Create a black square
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # We MUST set enforce=False here because there is no face in a black square
        FaceAnalyzer.analyze_face(dummy_img, enforce=False)
        print(">>> AI Engine: Warm-up Complete. System Ready.")
    except Exception as e:
        print(f">>> AI Engine: Warm-up error: {e}")

@app.route('/api/ai/scan', methods=['POST'])
def scan_face():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data"}), 400

    cv2_img = decode_base64_to_cv2(data['image'])
    if cv2_img is None:
        return jsonify({"status": "error", "message": "Bad image format"}), 400

    # Real scan uses enforce=True (default) for accuracy
    result = FaceAnalyzer.analyze_face(cv2_img)
    return jsonify(result)

if __name__ == "__main__":
    warm_up_models()
    print(f">>> Rasoi AI Vision Running on http://{HOST}:{PORT}")
    # debug=False is recommended for AI apps to prevent double-loading models
    app.run(host=HOST, port=PORT, debug=False)