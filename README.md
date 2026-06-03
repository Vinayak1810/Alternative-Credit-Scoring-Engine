# Alternative Credit Scoring Engine

A machine learning engine built in PyTorch designed to predict loan default risk for the unbanked and underbanked population using alternative data sources (e.g., telecom behavior, housing stability, and employment metrics). 

This project goes beyond traditional credit scoring metrics to address severe real-world data constraints, specifically focusing on handling highly imbalanced target classes and optimizing decision thresholds for commercial risk appetite.

## Project Architecture & Strategy

### 1. Data Pipeline & Feature Engineering
* **Data Source:** Processed a comprehensive financial lending dataset containing over 300,000 application profiles with 122 starting features.
* **Dimensionality Reduction:** Automated the deletion of columns with a missing data threshold exceeding 60% to mitigate noise from sparse entries. Residual numerical gaps were imputed using robust column medians.
* **Categorical Encoding:** Executed One-Hot Encoding across 16 multi-class string features, expanding the numerical training surface to 211 binary identifiers.
* **Feature Scaling & Isolation:** Transformed all variables using a standard normalization scaler ($z$-score) to guarantee uniform weight activation gradients across varied inputs (e.g., comparing income parameters to family metrics). Implemented stratified splits ($80/20$) to maintain strict class balances without data leakage.

### 2. Deep Learning Architecture
The core classification model is constructed as a structured Deep Neural Network (DNN) utilizing **PyTorch**:

* **Input Layer:** 209 normalized input units matching the engineered feature space dimension.
* **Hidden Structure:** Two fully-connected linear layers (128 $\rightarrow$ 64 nodes) using Rectified Linear Unit (ReLU) activation functions to decipher complex non-linear correlation structures.
* **Regularization:** Layer-specific Dropout mechanisms ($p=0.3$) dynamically applied to restrict node co-dependency and suppress overfitting on historical training metrics.
* **Output Node:** Single binary predictor constrained by a Sigmoid transfer function to output explicit probability vectors between $0.0$ and $1.0$.

### 3. The Precision-Recall Strategic Trade-off
A standard training run optimization using Binary Cross Entropy (BCE) and Adam optimization ($lr=0.001$) initially yielded a high classification baseline accuracy of **92%**. 

However, because the loan default distribution was heavily imbalanced ($92\%$ repaid vs. $8\%$ defaulted), the network prioritized accuracy by predicting "repaid" across the majority of instances—resulting in a catastrophic default **Recall of just 1%**.


--- INITIAL MODEL PERFORMANCE (Standard 0.5 Threshold) ---
precision    recall  f1-score   support
Repaid (0)       0.92      1.00      0.96     56538
Defaulted (1)       0.59      0.01      0.02      4965
accuracy                           0.92     61503


**The Solution:** In alternative financial underwriting, the economic loss of a False Negative (undetected default) is massively higher than a False Positive (flagging a safe applicant for human underwriting). 

By lowering the decision threshold from the standard `0.5` down to `0.08` (matching the operational default base-rate), the engine shifted down the ROC curve to favor strict risk management.

--- OPTIMIZED MODEL PERFORMANCE (Dynamic 0.08 Threshold) ---
precision    recall  f1-score   support
Repaid (0)       0.96      0.68      0.80     56538
Defaulted (1)       0.16      0.69      0.26      4965
accuracy                           0.68     61503


* **Business Impact:** Default classification **Recall surged from 1% to 69%**, successfully capturing **3,404 actual credit defaults** that would have otherwise caused massive unrecoverable credit losses for the lender.

---

## Technical Stack & Libraries
* **Language:** Python
* **Deep Learning Frame:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Data Processing:** Pandas, NumPy
* **Analytics & Evaluation:** Scikit-Learn (`train_test_split`, `StandardScaler`, `classification_report`)
* **Visualization:** Seaborn, Matplotlib

## How to Run the Notebook
1. Clone this repository to your machine or launch a cloud instance in Kaggle/Colab.
2. Ensure you attach or place the Home Credit Default Risk competition dataset (`application_train.csv`) in your relative input path.
3. Install necessary baseline extensions:
```bash
   pip install torch pandas numpy scikit-learn seaborn matplotlib
