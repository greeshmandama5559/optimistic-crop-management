from flask import Flask,redirect ,url_for,render_template,request,jsonify, session
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from dotenv import load_dotenv
import pandas as pd
import os
import random
from model import generate
from img_process import is_leaf
from joblib import load

app = Flask(__name__)

load_dotenv()
app.secret_key = "project_crop_management"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

rf_model = load("random_forest_model.pkl")
label_encoders = load("label_encoders.pkl")

rfg = load("rfg.pkl")
label_enc = load("label_enc.pkl")

my_model_crops = ['Rice', 'Wheat', 'Maize', 'Potato', 'Onion', 'Sugarcane', 'Cotton',
       'Coconut', 'Soybean', 'Groundnut']
my_model_seasons = ['Kharif', 'Rabi', 'Whole Year']

year = 2025

def int_result(result):
    num_str = ""
    decimal_seen = False

    for char in result:
        if char.isdigit():
            num_str += char
        elif char == "." and not decimal_seen:
            num_str += char
            decimal_seen = True

    return float(num_str) if num_str else 0.0

def my_model(crp, ste, yr, rain, sea):
    manual_input = {
        "Crop": crp,
        "State": ste,
        "Year": yr,
        "Annual Rainfall": rain,
        "Season": sea
    }
    
    encoded_input = {
        "Crop": label_encoders["Crop"].transform([manual_input["Crop"]])[0],
        "State": label_encoders["State"].transform([manual_input["State"]])[0],
        "Year": manual_input["Year"],
        "Season": label_encoders["Season"].transform([manual_input["Season"]])[0],
        "Annual Rainfall": manual_input["Annual Rainfall"]
    }
    
    input_df = pd.DataFrame([encoded_input])
    
    predicted_yield = rf_model.predict(input_df)[0]
    
    return predicted_yield

def my_model_2(crp, ste, rain, sea,pes):
    crop_mapping = {
        "Cotton": "Cotton(lint)",
        "Soybean": "Soyabean",
        }
    
    manual_input = {
        "Crop": crop_mapping.get(crp, crp),
        "Season": sea,
        "State": ste,
        "Annual_Rainfall": rain,
        "Pesticide": pes
    }
    
    encoded_input = {
        "Crop": label_enc["Crop"].transform([manual_input["Crop"]])[0],
        "Season": label_enc["Season"].transform([manual_input["Season"]])[0],
        "State": label_enc["State"].transform([manual_input["State"]])[0],
        "Annual_Rainfall": manual_input["Annual_Rainfall"],
        "Pesticide": manual_input["Pesticide"]
    }
    
    input_df = pd.DataFrame([encoded_input])
    
    predicted_yield = rfg.predict(input_df)[0]
    
    return round(float(predicted_yield),7)*1000

class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
                'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 
                'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
                'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 
                'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

class_precautions = {
    'Apple___Apple_scab': 'Use fungicides and plant resistant varieties. Ensure proper air circulation.',
    'Apple___Black_rot': 'Cut off infected branches and use copper-based fungicides. Avoid overhead watering.',
    'Apple___Cedar_apple_rust': 'Remove nearby juniper plants. Apply fungicides during the growing season.',
    'Apple___healthy': 'Maintain regular monitoring and proper fertilization to ensure continued health.',
    'Blueberry___healthy': 'Ensure proper watering and soil acidity for optimal growth.',
    'Cherry_(including_sour)___healthy': 'Regularly cut and inspect for early signs of disease.',
    'Cherry_(including_sour)___Powdery_mildew': 'Improve air circulation and use sulfur-based fungicides.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant varieties and apply foliar fungicides when needed.',
    'Corn_(maize)___Common_rust_': 'Plant resistant hybrids and use appropriate fungicides if infection is severe.',
    'Corn_(maize)___healthy': 'Maintain crop rotation and monitor for early signs of disease.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use disease-resistant hybrids and apply fungicides at early infection stages.',
    'Grape___Black_rot': 'Cut off vines to improve airflow and apply fungicides early in the season.',
    'Grape___Esca_(Black_Measles)': 'Remove infected wood and avoid overwatering. Use balanced fertilization.',
    'Grape___healthy': 'Regularly inspect vines and ensure proper pruning for air circulation.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Remove infected leaves and use copper-based fungicides.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Remove infected trees and control psyllid insect populations.',
    'Peach___Bacterial_spot': 'Use disease-resistant varieties and apply copper-based sprays as needed.',
    'Peach___healthy': 'Cut off trees properly and maintain soil health.',
    'Pepper,_bell___Bacterial_spot': 'Use copper-based sprays and avoid overhead watering.',
    'Pepper,_bell___healthy': 'Monitor for pests and maintain optimal soil conditions.',
    'Potato___Early_blight': 'Use certified disease-free seeds and apply fungicides if needed.',
    'Potato___healthy': 'Rotate crops and keep the soil well-drained to prevent diseases.',
    'Potato___Late_blight': 'Remove infected plants immediately and apply fungicides preventively.',
    'Raspberry___healthy': 'Keep plants well-spaced and cut regularly for airflow.',
    'Soybean___healthy': 'Rotate crops and monitor for early disease symptoms.',
    'Squash___Powdery_mildew': 'Increase air circulation and use sulfur-based fungicides.',
    'Strawberry___healthy': 'Ensure good drainage and inspect regularly for pests or disease.',
    'Strawberry___Leaf_scorch': 'Avoid overhead watering and remove infected leaves promptly.',
    'Tomato___Bacterial_spot': 'Use copper sprays and plant disease-resistant varieties.',
    'Tomato___Early_blight': 'Remove infected leaves and apply fungicides if necessary.',
    'Tomato___healthy': 'Rotate crops and avoid excessive moisture on leaves.',
    'Tomato___Late_blight': 'Use resistant varieties and apply fungicides at first signs of infection.',
    'Tomato___Leaf_Mold': 'Ensure good ventilation in greenhouses and use fungicides if needed.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves and apply fungicides regularly.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil to control mite populations.',
    'Tomato___Target_Spot': 'Apply appropriate fungicides and maintain proper plant spacing.',
    'Tomato___Tomato_mosaic_virus': 'Use virus-free seeds and sanitize tools between uses.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies, as they transmit the virus. Use resistant varieties.'
}

confidence_threshold = 0.5

disease_model = keras.models.load_model('dense.keras')
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) 
    img_array /= 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    
    if disease_model.input_shape[-1] == 1: 
        img_array = np.mean(img_array, axis=-1, keepdims=True)

    return img_array

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/suggestions")
def suggestions():
    return render_template("suggestions.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/crop_yield", methods=["GET"])
def crop_yield():
    return render_template("crop_yield.html")

@app.route("/crop_yield_prediction", methods=["POST"])
def crop_yield_prediction():
    try:
        crop_type = request.json.get("crop-type")
        season = request.json.get("season")
        state = request.json.get("state")
        Annual_Rainfall = float(request.json.get("rainfall"))
        pesticide = request.json.get("pesticides")
        area = float(request.json.get("area"))

        data = [Annual_Rainfall, pesticide, crop_type, season, state]
        prev_data = session.get("prev_data")
        prev_area = session.get("area")



        if prev_data == data and prev_area == area:
            yieldrt = session.get("yieldrt")
            minyield = session.get("minyield")
            maxyield = session.get("maxyield")
        elif prev_data == data and prev_area != area:
            yieldrt = session.get("yieldrt")
            num = int(random.randint(1, 100))
            minyield = area * (yieldrt - num)
            maxyield = area * (yieldrt + num)
        else:
            if crop_type in my_model_crops and season in my_model_seasons and pesticide == '':
                result = my_model(crop_type, state, year, Annual_Rainfall, season)
            else:
                pesticide = float(pesticide)
                prompt = f"""crop type={crop_type}, season={season}, state={state}, Annual Rainfall={Annual_Rainfall}, 
                            pesticide={pesticide}kg/acre. Predict the expected crop yield in kilograms per acre with the highest possible accuracy.
                            Note: Give the output as 'Predicted Yield = [predicted yield] kg/acre in floating value', nothing more than that."""
                
                result = generate(prompt)
                result = int_result(result) if result else None

                if result is None:
                    result = my_model_2(crop_type, state, Annual_Rainfall, season, pesticide)

                # result = my_model_2(crop_type, state, Annual_Rainfall, season, pesticide)

            if result:
                yieldrt = result
                num = int(random.randint(1, 100))
                minyield = area * (yieldrt - num)
                maxyield = area * (yieldrt + num)

                session["prev_data"] = data
                session["area"] = area
                session["yieldrt"] = yieldrt
                session["minyield"] = minyield
                session["maxyield"] = maxyield
            else:
                session["error_output"] = "Error Generating Prompt. Please Try Again Later."
                return redirect(url_for("crop_yield"))

        return jsonify({
            "yieldrt": round(yieldrt, 2),
            "minyield": round(minyield, 2),
            "maxyield": round(maxyield, 2)
        })

    except Exception as e:
        session["error_output"] = f"Error: {e}"
        return jsonify({"error": str(e)}), 400

@app.route('/analyze', methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
    
            result = is_leaf(filename)

            if result is None:
                img_array = load_and_preprocess_image(filename)
                predictions = disease_model.predict(img_array)
                predicted_index = np.argmax(predictions)
            elif "no" not in result.lower():
                img_array = load_and_preprocess_image(filename)
                predictions = disease_model.predict(img_array)
                predicted_index = np.argmax(predictions)
            else:
                # os.remove(filename) 
                return jsonify({'error': f'{result} Try with different image. '}), 400
    
            if predicted_index >= len(class_labels):
                os.remove(filename) 
                return jsonify({'error': 'Prediction does not match any known plant class, '}), 400
    
            predicted_class = class_labels[predicted_index]
            precaution = class_precautions.get(predicted_class, "No precaution available")

            os.remove(filename) 
            return jsonify({
                'prediction': predicted_class,
                'precaution': precaution
            }), 200
        except Exception as e:
            return jsonify({'error': 'Something Went Wrong PLease Try Again Later, '}), 400

@app.route("/crop_disease")
def crop_disease():
    return render_template("crop_disease.html")

if __name__ == "__main__":
    app.run(debug=True)