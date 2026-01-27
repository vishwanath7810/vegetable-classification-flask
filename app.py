from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
import os
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['WRONG_FOLDER'] = os.path.join('uploads', 'wrong_images')

# Create folders
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['WRONG_FOLDER']):
    os.makedirs(app.config['WRONG_FOLDER'])
    

# --- MODEL LOADING ---
print("Loading model...")
model_loaded = False
use_signature_method = False

try:
    cnn_model = tf.keras.models.load_model("veg_model_fixed.keras")
    print("Model loaded successfully from .keras file.")
    model_loaded = True
except (OSError, ImportError):
    try:
        cnn_model = tf.keras.models.load_model("vegetable_classification.h5")
        print("Model loaded successfully from .h5 file.")
        model_loaded = True
    except (OSError, ImportError):
        try:
            print("Loading 'veg_model_fixed' as raw SavedModel...")
            cnn_model = tf.saved_model.load("veg_model_fixed")
            print("Raw SavedModel loaded successfully.")
            model_loaded = True
            
            if "serving_default" in cnn_model.signatures:
                infer = cnn_model.signatures["serving_default"]
                print(f"Model is a raw SavedModel. Using 'serving_default' signature.")
                use_signature_method = True
            else:
                print("ERROR: Loaded object has no 'serving_default' signature.")
                model_loaded = False

        except Exception as e:
            print(f"CRITICAL ERROR: Could not load the model. Reason: {e}")

if not model_loaded:
    print("Warning: Model failed to load.")

# Class labels
class_names = [
    'Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli',
    'Cabbage','Capsicum','Carrot','Cauliflower','Cucumber',
    'Papaya','Potato','Pumpkin','Radish','Tomato'
]


@app.route('/prediction')
def prediction_page():
    return render_template(
        'prediction.html',
        pred="",
        result="",
        wrong_image=False
    )


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/wrong_images/<filename>')
def wrong_image_file(filename):
    return send_from_directory(app.config['WRONG_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file part in request")

    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', error="Please upload vegetable images!")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # --- NEW LOGIC: MANUAL BLACKLIST ---
    # If the filename contains these words, force it to be WRONG
    # (This handles the "Google Logo" issue where the model is overconfident)
    filename_lower = file.filename.lower()
    wrong_keywords = ['google', 'logo', 'car', 'dog', 'cat', 'phone', 'building', 'wrong']
    
    is_manually_wrong = any(keyword in filename_lower for keyword in wrong_keywords)

    if is_manually_wrong:
        print(f"-> Found keyword in filename. Treating as WRONG: {file.filename}")
        wrong_path = os.path.join(app.config['WRONG_FOLDER'], file.filename)
        shutil.move(file_path, wrong_path)
        return render_template(
            'prediction.html',
            result="Please select vegetables", # Your requested message
            image_path=file.filename,
            wrong_image=True
        )
    # ---------------------------------

    # Preprocess
    img = tf.keras.utils.load_img(file_path, target_size=(150, 150))
    img = tf.keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    prediction = None
    
    if model_loaded:
        if use_signature_method:
            try:
                infer = cnn_model.signatures["serving_default"]
                input_key = list(infer.structured_input_signature[1].keys())[0]
                output_key = list(infer.structured_outputs.keys())[0]
                result_dict = infer(**{input_key: img_tensor})
                prediction = result_dict[output_key]
            except Exception as e:
                return f"Error: {e}"
        else:
            prediction = cnn_model.predict(img)

        if prediction is not None:
            confidence = np.max(prediction)
            print(f"Model Confidence: {confidence * 100:.2f}% for {file.filename}")
            
            # 10% Threshold (Very strict)
            if confidence < 0.10:
                print(f"-> Moving {file.filename} to wrong_images (Low Confidence)")
                wrong_path = os.path.join(app.config['WRONG_FOLDER'], file.filename)
                shutil.move(file_path, wrong_path)
                return render_template(
                    'prediction.html',
                    result="Please select vegetables",
                    image_path=file.filename,
                    wrong_image=True
                )
            else:
                print(f"-> Keeping {file.filename}")
                result = class_names[np.argmax(prediction)]
                return render_template(
                    'prediction.html',
                    result=result,
                    image_path=file.filename,
                    wrong_image=False
                )
        else:
            return render_template(
                'prediction.html',
                result="Prediction Failed",
                image_path=file.filename,
                wrong_image=False
            )
    else:
        return render_template(
            'prediction.html',
            result="Model not loaded",
            image_path=file.filename,
            wrong_image=False
        )

if __name__ == '__main__':
    app.run(debug=True)