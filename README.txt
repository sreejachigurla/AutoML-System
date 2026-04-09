# AutoML System

## 📌 Overview

The AutoML System is a web-based machine learning application that automates the complete process of data analysis and model building. Users can upload a dataset in CSV format, and the system automatically detects the problem type, preprocesses the data, selects the most suitable algorithm, and generates predictions along with performance metrics and visualizations.

---

## 🎯 Objectives

* Simplify the machine learning workflow for users
* Automate model selection and evaluation
* Provide quick insights from raw datasets
* Reduce the need for manual coding in ML tasks

---

## ⚙️ Features

* 📂 Upload CSV datasets through a user-friendly interface
* 🤖 Automatic problem detection (Regression, Classification, Clustering)
* 🧹 Data preprocessing (handling missing values, encoding, scaling)
* ⚡ Model training using Scikit-learn algorithms
* 📊 Performance metrics and visualization charts
* 📋 Display of predictions and dataset preview

---

## 🛠️ Technologies Used

* **Backend:** Python (Flask)
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib

---

## 🔄 Workflow

1. User uploads a dataset (CSV file)
2. System analyzes the dataset structure
3. Detects problem type automatically
4. Performs preprocessing steps
5. Selects and trains the best model
6. Displays results including metrics and charts

---

## ▶️ How to Run the Project

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
2. Run the Flask application:

   ```
   python app.py
   ```
3. Open your browser and go to:

   ```
   http://localhost:5050
   ```

---

## 📈 Output

* Dataset summary and statistics
* Selected algorithm and model performance
* Visualization charts
* Sample predictions
* Feature importance (if applicable)


---

## 📌 Conclusion

This project demonstrates how machine learning tasks can be automated efficiently using a simple web interface. It is useful for beginners and helps in understanding the end-to-end ML pipeline without deep technical complexity.

---
