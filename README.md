🥦 Vegetable Classification using Deep Learning & Flask

A professional end-to-end Vegetable Image Classification Web Application built using TensorFlow (CNN) and Flask.
Users can upload an image of a vegetable and the system predicts the vegetable type with high accuracy.


🚀 Features
📷 Upload vegetable image
🧠 Deep Learning CNN-based classification
⚡ Real-time prediction using Flask
❌ Detects non-vegetable / wrong images
📁 Automatically separates wrong images
🎨 Clean and responsive UI
🧪 Tested with multiple image samples


🛠️ Tech Stack
Backend
Python 3.10
TensorFlow / Keras
Flask
NumPy
Frontend
HTML5
CSS3


Tools
Google Colab (Model Training)
VS Code
Git & GitHub

📂 Project Structure
vegetable-classification-flask/
│
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
│
├── static/
│   └── css/
│       └── style.css       # Styling
│
├── templates/
│   ├── index.html          # Upload page
│   └── prediction.html    # Result page
│
├── uploads/
│   ├── wrong_images/       # Non-vegetable images
│   └── *.jpg
│
└── README.md
⚠️ Note:
Trained model files and large datasets are excluded from GitHub due to size limits.



🧠 Model Details
Model Type: Convolutional Neural Network (CNN)
Input Size: 150 × 150 RGB
Output: 15 Vegetable Classes
Activation: Softmax
Framework: TensorFlow / Keras

Vegetable Classes
Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli,
Cabbage, Capsicum, Carrot, Cauliflower, Cucumber,
Papaya, Potato, Pumpkin, Radish, Tomato


🔄 Application Workflow
User uploads an image
Image preprocessing (resize, normalize)
Model performs prediction
Confidence score is evaluated
Result is displayed on UI
Wrong images are moved automatically

▶️ How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/vishwanath7810/vegetable-classification-flask.git
cd vegetable-classification-flask

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
python app.py

5️⃣ Open Browser
http://127.0.0.1:5000




🎯 Use Cases
Smart agriculture systems
Grocery automation
Learning project for Deep Learning & Flask
College final-year mini project


🚀 Future Enhancements
Mobile app (Flutter / Android)
Fruit & grocery classification
Cloud deployment (Render / AWS)
Confidence-based rejection improvement
Admin dashboard


👨‍💻 Author
Vishwanath Todkar
📍 Android | Flutter | Python Developer
📌 GitHub: vishwanath7810


⭐ Support

If you like this project, please ⭐ the repository
It motivates me to build more projects 💙
