# f1stratpred

🏁 F1 Strategy Prediction
AI-Enabled Tyre Strategy Recommender for Formula 1 Races

This project leverages machine learning to predict the optimal tyre strategy (Soft, Medium, or Hard) for Formula 1 circuits based on various race conditions and track characteristics. By analyzing historical data—including weather, track type, pit stops, and stint performance—the model recommends the best tyre choice to maximize race performance and efficiency.

🔍 Features
Predicts the best tyre strategy for a given F1 race scenario

Trained on curated datasets from past races (sourced and preprocessed)

Interactive web interface built with Flask for easy use

Lap-by-lap analysis potential for future updates

🧠 Tech Stack
Python (Pandas, Scikit-learn)

Streamlit for the web app

HTML/CSS + JavaScript for frontend

Model saved as model.pkl

🚀 How to Use
Clone the repo and install requirements

Run train_model.py to train or update the model

Launch the app using app.py

Input race conditions on the website to get strategy recommendations

📂 Project Structure
bash
Copy
Edit
├── app.py
├── model.pkl
├── train_model.py
├── test_setup.py
├── /templates
│   └── index.html
├── /static
│   ├── style.css
│   └── script.js
└── /venv
📈 Future Improvements
Add lap-wise prediction and pit stop timing

Integrate real-time race data APIs

Deploy on cloud platforms (Heroku, Vercel, etc.)
