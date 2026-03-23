from deepface import DeepFace
import numpy as np

class FaceAnalyzer:
    @staticmethod
    def analyze_face(image, enforce=True):
        try:
            # 1. Multi-modal analysis using MTCNN (Best for facial proportions)
            results = DeepFace.analyze(
                img_path=image, 
                actions=['age', 'gender', 'emotion'],
                detector_backend='mtcnn', 
                align=True,
                enforce_detection=enforce,
                silent=True
            )
            
            res = results[0]
            
            # --- FIX 1: GENDER ACCURACY (Probability Weighting) ---
            # DeepFace returns a dict: {'Man': 98.2, 'Woman': 1.8}
            # We lower the threshold for 'Woman' to 35% to counter shadow-based 'Man' bias.
            gender_data = res['gender']
            if gender_data['Woman'] > 35.0:
                final_gender = "Woman"
            else:
                final_gender = "Man"

            # --- FIX 2: CHILD DETECTION HEURISTIC (1-10 Range) ---
            raw_age = res['age']
            
            # Heuristic: If AI sees a 'Young Adult' (18-26) but the face is small/smooth,
            # it is likely a child. We apply a 0.65 multiplier to pull them into the 1-10 range.
            if raw_age < 26:
                # This helps catch the 1-10 range that models usually skip
                processed_age = raw_age * 0.65 
            else:
                processed_age = raw_age

            # --- FIX 3: PROFESSIONAL DECADE SNAPPING ---
            # We snap to the midpoint of the 10-year ranges for React UI consistency
            if processed_age <= 10: 
                refined_age = 5       # UI will show 1-10
            elif processed_age <= 20: 
                refined_age = 15      # UI will show 11-20
            elif processed_age <= 30: 
                refined_age = 25      # UI will show 21-30
            elif processed_age <= 40: 
                refined_age = 35      # UI will show 31-40
            elif processed_age <= 50: 
                refined_age = 45      # UI will show 41-50
            elif processed_age <= 60: 
                refined_age = 55      # UI will show 51-60
            else: 
                refined_age = 75      # UI will show 60+

            return {
                "status": "success",
                "age": int(refined_age), 
                "gender": final_gender,
                "emotion": res['dominant_emotion'],
                "confidence": round(res['face_confidence'] * 100, 2)
            }

        except Exception as e:
            print(f"AI Vision Error: {str(e)}")
            return {"status": "error", "message": "Please ensure your face is well-lit and visible."}