# Customer-Churn-Prediction - Telecom
This project focuses on building a customer churn prediction pipeline, starting from data exploration and preprocessing to model training, evaluation, and interpretability. The ultimate goal is to help businesses identify customers at risk of churn and design effective retention strategies.
<br><br>
<h2>Project Workflow</h2>
 
<h3>1. Data Exploration & Profiling</h3>
<ul>- Analyzed key customer attributes such as Tenure, MonthlyCharges, and TotalCharges.</ul>
<ul>- Visualized churn vs non-churn distributions with Seaborn KDE plots and Plotly histograms/bar charts.</ul>
<ul>- Generated customer risk-type distributions with interactive Plotly bar charts.</ul>
 
<h3>2. Data Preprocessing</h3>
<ul>- Shuffling and resetting dataset index.</ul>
<ul>- One-hot encoding for categorical features.</ul>
<ul>- Normalization of numerical features using StandardScaler.</ul>
<ul>- Splitting into train-test sets using train_test_split.</ul>
 
<h3>3. Feature Engineering</h3>
<ul>- Merged categorical, numerical, and target features into a clean modeling dataset.</ul>
<ul>- Created feature sets X and target y.</ul>
<ul>- Used both df_model.loc[:, ~df_model.columns.isin(target_col)] and X = merged_df[feature_cols] to build predictors.</ul>
 
<h3>4. Model Training & Hyperparameter Tuning</h3>
<ul>- Models used: Logistic Regression, Random Forest, Gradient Boosting.</ul>
<ul>- Hyperparameter optimization via **RandomizedSearchCV**</ul>
<ul>- Used cross_val_score for comparing performance across multiple scoring metrics (accuracy, precision, recall, f1, log-loss).</ul>
 
<h3>5. Evaluation Metrics</h3>
<li>Evaluated models using:
<ul>- Accuracy</ul>
<ul>- Precision, Recall, F1-score</ul>
<ul>- Log Loss (to capture uncertainty in probabilistic predictions)</ul>
<ul>- Custom Cus_log_loss implemented to compare with sklearn.metrics.log_loss.</ul>
<ul>- Explored different thresholds to balance precision vs recall.</ul>
<ul>- Confusion matrix heatmaps plotted for each model.</ul></li>
 
<h3>6. Model Interpretability</h3>
<ul>- Feature importance extracted from Gradient Boosting.</ul>
<ul>- Interpreted model behavior using SHAP (SHapley Additive exPlanations): SHAP values explain the contribution of each feature to churn probability.</ul>
<ul>- Bar plots with Matplotlib and interactive Plotly visualizations.</ul>
 
<h3>7. Customer Churn Probability</h3>
<ul>- Predicted churn probabilities with predict_proba.</ul>
<ul>- Explained that .predict() uses a default threshold = 0.5 (but I tuned custom thresholds).</ul>
<ul>- Visualized probability distributions with Plotly histograms.</ul>
 
<h3>8. Customer Segmentation & Retention Strategy,</h3>
<ul>- Identified high-risk churn customers and visualized their distribution.</ul>
<ul>- Risk types grouped via value_counts() and plotted.</ul>
<ul>- Suggested actionable retention plans (discounts, loyalty rewards, proactive customer service).</ul>
 
<h3>9. Future Enhancements</h3>
<ul>- Incorporate Cohort Analysis to study churn patterns across customer groups (by acquisition date).</ul>
<ul>- Add external data sources:</ul>
<ul>- Customer inquiries & complaint logs</ul>
<ul>- Seasonality in sales</ul>
<ul>- Broader demographics & engagement metrics</ul>
 
 
<h3>Tech Stack</h3>
 
**Python Libraries:**
<br>
<ul><b>Data:</b>Data: pandas, numpy</ul>
<ul><b>Visualization:</b> matplotlib, seaborn, plotly</ul>
<ul><b>ML:</b> scikit-learn, shap</ul>
<ul><b>Approcah:</b> End-to-end ML pipeline with interpretable outputs.</ul>
 
<h3>Example Visuals</h3>
<ul>- Customer churn distribution (churn vs no churn).</ul>
<ul>- Probability distribution of churned vs retained customers.</ul>
<ul>- SHAP feature importance plot for interpretability.</ul>
<ul>- Confusion matrix heatmaps for model comparison.</ul>
