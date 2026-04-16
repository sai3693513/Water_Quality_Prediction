# Project Walkthrough: Water Quality Prediction System (Aquify)

Congratulations! Your **Water Quality Prediction System** is now a professional, high-end academic project with a modular backend, detailed documentation, and a premium UI. This walkthrough will guide you through the features and how to use the system for your final-year presentation.

## 🚀 How to Run the Project

1.  **Open Terminal**: Navigate to your project's backend directory:
    `c:\Users\bunny\OneDrive\Desktop\project details\backend`
2.  **Activate Virtual Environment** (if applicable):
    ```powershell
    .\venv\Scripts\activate
    ```
3.  **Install Requirements** (if not already done):
    ```powershell
    pip install flask joblib pandas scikit-learn imbalanced-learn xgboost google-generativeai
    ```
4.  **Run the App**:
    ```powershell
    python app.py
    ```
5.  **Access the Site**: Open your browser and go to `http://127.0.0.1:8000`

---

## 🛡️ Core Features

### 1. Premium Home & Discovery
The new home page features a modern, clean aesthetic tailored for an environmental health application. It clearly outlines the importance of water potability and provides a smooth entry point for users.

### 2. Secure Authentication
A dedicated Login and Registration system ensures that each user has their own private space. This allows for personalized prediction history and profile management.

### 3. AI Training & Evaluation Center
-   **Dataset Upload**: Upload your CSV datasets (e.g., `water_potability.csv`) directly from the browser.
-   **Smart Training**: The system uses **Random Forest** and **XGBoost** with **SMOTE** balancing to ensure maximum accuracy even with skewed data.
-   **Instant Feedback**: Get real-time metrics (Accuracy, Precision, Recall) immediately after training, visualized with professional charts.

### 4. Smart Water Detection (Prediction)
Users can enter specific water parameters such as **pH**, **Hardness**, **Solids**, **Chloramines**, and **Sulfate**. The system then utilizes the trained model to instantly determine if the water is **Safe** or **Unsafe** for drinking.

### 5. Interactive Chatbot: "Aquabot"
Built into the dashboard is an intelligent security assistant.
-   **Educational**: Ask it what "pH" means or why "High Solids" are dangerous.
-   **AI Powered**: When a Gemini API key is provided, it offers deep insights into water quality and safety recommendations.

### 6. Persistence & History
Every prediction made is logged in the **History** section. Users can revisit their past tests, see the values they entered, and track the safety status of different water sources over time.

---

## 📄 Academic Documentation
I've included complete documentation in the `documentation/` folder to help with your report and viva:
-   **`abstract.md`**: A concise summary of your project for your report's introduction.
-   **`problem_statement.md`**: Explains the "Why" and "What" of your project (ideal for the problem definition chapter).
-   **`viva_questions.md`**: A "cheat sheet" of possible examiner questions and expert answers to help you ace your final exam.

## ✅ Accomplishments
-   **Advanced ML Pipeline**: Implemented a robust pipeline including **SimpleImputer**, **StandardScaler**, and **SMOTE** for high-reliability predictions.
-   **Modular Backend**: Logic is cleanly separated into the `backend/` directory, following industry standards for Flask applications.
-   **Premium Aesthetics**: The application features a glassmorphic design, custom iconography, and responsive layouts for a "state-of-the-art" feel.
-   **Chatbot Integration**: Successfully integrated a hybrid (Rule-based + AI) chatbot for enhanced user interaction.

> [!TIP]
> Use the **Performance** page to show your examiners the Confusion Matrix and Model Comparison charts—this demonstrates a deep technical understanding of the Machine Learning process.
