## 🌐 Week-3 — Streamlit App Deployment

This week, I deployed the **EV Green Policy Simulator** as a live web application using Streamlit and Render.

### 🚀 Deployment Steps
1. Prepared the final `app.py` with:
   - Multi-page navigation (Home, Analytics, Simulator, AI Assistant)
   - Integrated trained ML model (`ev_policy_best_model.pkl`)
   - Integrated scaler (`scaler.pkl`)
   - Added Groq LLM API for the AI Assistant
2. Added required dependencies in `requirements.txt`
3. Uploaded all files to a GitHub repository
4. Deployed the application using Render (Web Service)
5. Set the environment variable: `GROQ_API_KEY = your_api_key`
6. Linked repository → Auto Build → Deployment successful

### 🔗 Live App URL
https://ev-green-policy-simulator.onrender.com

### 🔐 Application Login Details
✔ **Username:** `admin`  
✔ **Password:** `password`  
*(Evaluation access for faculty/testing only)*

⚠️ **Important Note:**  
After entering the username and password, you need to **click the Login button twice** to enter the application.  
(This is due to an authentication refresh in the Streamlit session state.)


### 📁 Files Used for Deployment
- `app.py`
- `models/ev_policy_best_model.pkl`
- `models/scaler.pkl`
- `requirements.txt`
- `ev_adoption_dataset_clean.csv`

The deployed app provides:
- Interactive EV analytics  
- Policy simulation  
- Predictive modeling  
- AI-powered dataset insights  

## 👨‍💻 Contributor
**Rahul Majumder**  
*Developer & Project Author*

> *“Week-3 transformed this project from a model into an experience — bringing data, intelligence, and interactivity together into a single web application.”*
