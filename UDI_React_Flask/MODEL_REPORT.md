# UDI Domain Model Report

Each domain has explicitly named reference artifacts. Regression and classification candidates are all trained for each domain; the selected model is recorded in metadata. Forecasting and anomaly models are also trained per domain.

| Domain | Selected Regression | Selected Classification | Forecasting | Anomaly |
|---|---|---|---|---|
| college | `college_regression_linear.joblib` | `college_classification_random_forest.joblib` | `college_forecasting_random_forest.joblib` | `college_anomaly_isolation_forest.joblib` |
| community | `community_regression_gradient_boosting.joblib` | `community_classification_xgboost.joblib` | `community_forecasting_random_forest.joblib` | `community_anomaly_isolation_forest.joblib` |
| company | `company_regression_linear.joblib` | `company_classification_logistic.joblib` | `company_forecasting_random_forest.joblib` | `company_anomaly_isolation_forest.joblib` |
| event_organizer | `event_organizer_regression_gradient_boosting.joblib` | `event_organizer_classification_gradient_boosting.joblib` | `event_organizer_forecasting_random_forest.joblib` | `event_organizer_anomaly_isolation_forest.joblib` |
| hospital | `hospital_regression_linear.joblib` | `hospital_classification_logistic.joblib` | `hospital_forecasting_random_forest.joblib` | `hospital_anomaly_isolation_forest.joblib` |
| municipality | `municipality_regression_linear.joblib` | `municipality_classification_xgboost.joblib` | `municipality_forecasting_random_forest.joblib` | `municipality_anomaly_isolation_forest.joblib` |
| ngo | `ngo_regression_gradient_boosting.joblib` | `ngo_classification_logistic.joblib` | `ngo_forecasting_random_forest.joblib` | `ngo_anomaly_isolation_forest.joblib` |

## Candidate algorithms trained for every domain

Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor.

Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, XGBoost Classifier.

Forecasting: Random Forest lag-based model with time-ordered backtesting.

Anomaly detection: Isolation Forest.

These are reference models for the supplied seven domain datasets. Uploaded real-world datasets are not forced to use them: the runtime ML engine validates the uploaded schema and trains fresh candidate models when the schema differs.