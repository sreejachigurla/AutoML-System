AutoML System — Full Stack Web Application
==========================================

QUICK START
-----------
1. Install dependencies:
   pip install flask flask-cors pandas numpy scikit-learn matplotlib

2. Run the server:
   python app.py

3. Open your browser:
   http://localhost:5050

HOW TO USE
----------
1. Open http://localhost:5050 in your browser
2. Drop any CSV file onto the upload zone (or click to browse)
3. Optionally override:
   - Problem Type: Auto-detect / Regression / Classification / Clustering
   - Target Column: type the column name you want to predict
4. Click "Run AutoML Analysis"
5. View results: charts, metrics, predictions table, feature importance

WHAT IT DOES AUTOMATICALLY
---------------------------
✓ Detects problem type (regression / classification / clustering)
✓ Drops high-missing columns (>60% null)
✓ Removes ID-like columns
✓ Label-encodes categorical features
✓ Fills missing values with median
✓ Standardizes all features (z-score)
✓ For classification: tries Logistic Regression, Decision Tree, KNN
  → picks best via 5-fold cross-validation
✓ For regression: Linear Regression with full diagnostic plots
✓ For clustering: K-Means with auto-selected k (elbow + silhouette)
✓ Generates charts: actual vs predicted, confusion matrix, PCA scatter, etc.
✓ Shows feature importance, sample predictions, preprocessing steps

SAMPLE DATASETS (in sample_data/ folder)
-----------------------------------------
house_prices.csv   → Regression  (predict house price)
weather.csv        → Classification (predict RainTomorrow)
mall_customers.csv → Clustering (segment customers)

FILES
-----
app.py           - Flask backend (ML engine + API)
static/index.html - Frontend (HTML/CSS/JS)
sample_data/     - Sample CSV files to test with
