import os

# Set project root so DeepFace knows where to find the .deepface folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['DEEPFACE_HOME'] = project_root

# --- Performance vs Accuracy Settings ---
# 'mtcnn' is highly accurate for age/gender but much faster than 'retinaface'
FACE_DETECTOR_BACKEND = 'mtcnn' 

# API Settings
HOST = "0.0.0.0"
PORT = 5001