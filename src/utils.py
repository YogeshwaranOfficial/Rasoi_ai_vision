import base64
import numpy as np
import cv2

def decode_base64_to_cv2(base64_string):
    """Enhanced decoding with medical-grade contrast and sharpening filters."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None: return None

        # --- STEP 1: SHADOW RECOVERY (CLAHE) ---
        # Converts to LAB color space to boost luminance without ruining color
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        image = cv2.merge((cl,a,b))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        # --- STEP 2: NOISE REDUCTION ---
        # Slight blur to remove 'pixel grain' that AI confuses for wrinkles
        image = cv2.GaussianBlur(image, (3,3), 0)

        # --- STEP 3: FEATURE SHARPENING ---
        # Amplifies facial landmarks (eyes, mouth, wrinkles)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)

        return image
    except Exception as e:
        print(f">>> Processing Error: {e}")
        return None