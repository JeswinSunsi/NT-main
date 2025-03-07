from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import parselmouth
import numpy as np
import traceback
import os
from typing import Dict, Any,List
from pdf.report import create_report
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf
import cv2


app = FastAPI()

labels: List[str] = ['Healthy', 'Parkinson']

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return {"message": "Welcome to the Audio Analysis API"}

@app.post("/analyze")
async def analyze_and_predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        upload_directory = 'temp'
        os.makedirs(upload_directory, exist_ok=True)


        file_path = os.path.join(upload_directory, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        analysis_results = analyze_audio(file_path)
        

        with open('a.txt', 'w') as f:
            f.write(str(analysis_results))

        os.remove(file_path)

        input_data = {
            'mean_pitch': round(analysis_results['mean_pitch'], 2),
            'mean_intensity': round(analysis_results['mean_intensity'], 2),
            'f1': round(analysis_results['f1'], 2),
            'f2': round(analysis_results['f2'], 2),
            'f3': round(analysis_results['f3'], 2)
        }

        thresholds = {
            'pitch': 116.09,
            'intensity': 67.89,
            'f1': 1343.93,
            'f2': 1688.41,
            'f3': 1495.40
        }


        prediction = heuristic_model(
            input_data['mean_pitch'],
            input_data['mean_intensity'],
            input_data['f1'],
            input_data['f2'],
            input_data['f3'],
            thresholds
        )

        result = "Parkinson's" if prediction == 1 else "Not Parkinson's"
        detected= "High" if prediction==1 else "Low"
       
        create_report(detected=detected,pitch=input_data['mean_pitch'],
            intensity=input_data['mean_intensity'],
            f1=input_data['f1'],
            f2=input_data['f2'],
            f3=input_data['f3'],)
        print(detected,input_data['mean_intensity'],input_data['mean_pitch'],input_data['f1'],input_data['f2'],input_data['f3'])
        differences = {
            'pitch_diff': round(input_data['mean_pitch'] - thresholds['pitch'], 2),
            'intensity_diff': round(input_data['mean_intensity'] - thresholds['intensity'], 2),
            'f1_diff': round(input_data['f1'] - thresholds['f1'], 2),
            'f2_diff': round(input_data['f2'] - thresholds['f2'], 2),
            'f3_diff': round(input_data['f3'] - thresholds['f3'], 2)
        }

        response_data = {
            'prediction': result,
            'user_values': input_data,
            'thresholds': thresholds,
            'differences': differences
        }
        

        
        with open('b.txt', 'w') as f:
            f.write(str(response_data))
        headers = {"Content-Disposition": "attachment; filename=voice_analysis_report.pdf"} 
        return FileResponse('voice_analysis_report.pdf',media_type="application/pdf",headers=headers)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
async def results():
    try:
        with open('b.txt', 'r') as f:
            data = eval(f.read())
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def heuristic_model(mean_pitch: float, mean_intensity: float, f1: float, f2: float, f3: float, thresholds: Dict[str, float]) -> int:
    if (mean_pitch < thresholds['pitch'] and
        mean_intensity < thresholds['intensity'] and
        f1 > thresholds['f1'] and
        f2 > thresholds['f2'] and
        f3 > thresholds['f3']):
        return 1 
    return 0 

def analyze_audio(file_path: str) -> Dict[str, Any]:
    try:
    
        sound = parselmouth.Sound(file_path)

        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0] 
        mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

     
        intensity = sound.to_intensity()
        intensity_values = intensity.values
        mean_intensity = np.mean(intensity_values)

        
        formants = sound.to_formant_burg()
        midpoint = sound.duration / 2

        
        f1 = formants.get_value_at_time(1, midpoint)
        f2 = formants.get_value_at_time(2, midpoint)
        f3 = formants.get_value_at_time(3, midpoint)

        return {
            'mean_pitch': float(mean_pitch),
            'mean_intensity': float(mean_intensity),
            'f1': float(f1),
            'f2': float(f2),
            'f3': float(f3),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/report')
async def reports():
    headers = {"Content-Disposition": "attachment; filename=voice_analysis_report.pdf"} 
    return FileResponse('voice_analysis_report.pdf',media_type="application/pdf",headers=headers)

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1)  
    
    return img_array

model = load_model('scribble/parkinson_disease_detection.h5')

@app.post("/scribble")
async def scribble(file: UploadFile = File(...)):
    try:

        if not file.content_type.startswith('image/'):
            return {
                "error": "Uploaded file must be an image",
                "status": "error"
            }
        

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
  
        processed_image = preprocess_image(image)
        

        if processed_image.shape != (1, 128, 128, 1):
            return {
                "error": f"Invalid image shape after processing: {processed_image.shape}",
                "status": "error"
            }
        

        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction[0], axis=0)) 
        confidence = float(prediction[0][predicted_class])  
        return {
            "prediction": labels[predicted_class],
            "has_parkinsons": bool(predicted_class == 1),
            "confidence": confidence,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)