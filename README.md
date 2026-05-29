EDA.ipynb

General exploratory data analysis of the dataset, including target distribution, metadata variables, and basic descriptive summaries.

Feature_EDA.ipynb

Exploratory analysis of the extracted ECG-derived features. This notebook is used to inspect feature distributions, missing values, outliers, and relationships between features and the target variable.

Logistic_regression.ipynb

Training and evaluation of the logistic regression model. This notebook includes preprocessing, transformation of skewed features, L2-regularized logistic regression, threshold selection using the F2-score, and final validation/test evaluation.

Random_forest.ipynb

Initial random forest modelling notebook. It contains the original random forest hyperparameter search and model evaluation.

rf.ipynb

Additional random forest experiment. This notebook tests an alternative random forest setup with fully grown trees and tuning mainly through the feature subset parameter (max_features). It was used to check whether the initial random forest model could be improved.

XGBoost.ipynb

Training and evaluation of the XGBoost model. This notebook includes hyperparameter tuning, threshold selection, feature importance analysis, and final validation/test evaluation. Also includes pr-auc and roc-curves for all models to compare.

src/ directory
src/read_code15.py

Functions for loading metadata and ECG records from the CODE-15 dataset structure.

src/preprocessing.py

ECG preprocessing functions, including filtering, quality checks, usable lead selection, and normalization.

src/feature_extraction.py

Functions for extracting ECG-derived features, including R-peak features, RR-interval features, interval features, frequency-domain features, and wavelet-based features.

src/build_feature_table.py

Main script for applying preprocessing and feature extraction to all ECG records. The script processes records in batches and saves the final feature table..

src/Check_preprocess_res.py

Utility script for checking preprocessing results and inspecting whether ECG records were processed correctly.