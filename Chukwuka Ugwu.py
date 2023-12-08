#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


# In[5]:


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.cluster import MiniBatchKMeans
import os  # Import the 'os' module to set environment variables
from keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples


# In[6]:


# Load data
data = pd.read_excel("chuks.xlsx")
dx = pd.DataFrame(data)
dx


# In[ ]:


# Creating a MinMaxScaler object
scaler = MinMaxScaler()

# Fitting and transforming the data
normalized_data = scaler.fit_transform(data)
dx


# In[ ]:


# Data exploration
dx.info()
dx.hist(color='black', figsize=(15, 15))
plt.show()


# In[ ]:


# Pie chart for heart disease distribution
dx['cardio'].value_counts().plot.pie(autopct='%1.1f%%', title="Heart Disease Distribution")
plt.ylabel('')
plt.show()

# Pie chart for gender distribution
dx['gender'].value_counts().plot.pie(autopct='%1.1f%%', title="Gender Distribution")
plt.show()


# In[ ]:


# Bar plot for cholesterol by gender
plt.figure(figsize=(10, 5))
sns.barplot(data=dx, x='gender', y='cholesterol', palette='viridis')
plt.title('Cholesterol Levels by Gender')
plt.show()


# In[ ]:


# Data preprocessing
dx = dx.astype(float)


# In[ ]:


# BMI calculation
dx['bmi'] = dx['weight'] / ((dx['height'] / 100) ** 2)
dx


# In[ ]:


# Pulse pressure calculation
dx['pulse_pressure'] = dx['ap_hi'] - dx['ap_lo']
dx


# In[ ]:


# Handling missing values
dx.dropna(inplace=True)
dx


# In[ ]:


# Assuming dx is your DataFrame
correlation_matrix = dx.corr()

# Set the size of the figure
plt.figure(figsize=(10, 8))

# Create the heatmap with the 'viridis' colormap
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5)

# Add a title to the heatmap
plt.title('Correlation Plot')

# Display the plot
plt.show()


# In[ ]:


# Correlation analysis
correlation_matrix = dx.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Plot')
plt.show()


# In[ ]:


dx.hist(color='r', figsize=(15, 15))
plt.show()


# In[ ]:


# Feature selection
dx.drop(['id', 'alco'], axis=1, inplace=True)


# In[ ]:


# Age conversion to years
dx["age"] = dx["age"].map(lambda x: x // 365)


# In[ ]:


dx[['age']].describe()


# In[ ]:


# KDE plot for age
plt.figure(figsize=(8, 5))
sns.kdeplot(data=dx, x='age', fill=True, color='blue')
plt.title('Age Distribution')
plt.show()


# In[ ]:


# Outlier handling using Z-scores
z_scores = np.abs(stats.zscore(dx.select_dtypes(include=[np.number])))
dx = dx[(z_scores < 3).all(axis=1)]


# In[ ]:


# Splitting data into features and target
X = dx.drop('cardio', axis=1)
y = dx['cardio']


# In[ ]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"Logistic Regression - Accuracy: {accuracy_lr:.2f}, F1 Score: {f1_lr:.2f}, ROC AUC: {roc_auc_lr:.2f}, MAE: {mae_lr:.2f}")

# Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(logistic_regression, X, y, cv=5, scoring='accuracy')
print(f"Logistic Regression - Average Cross-validation score: {np.mean(cv_scores_lr):.2f}")


# In[ ]:





# In[ ]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
print(f"Decision Tree - Accuracy: {accuracy_dt:.2f}, F1 Score: {f1_dt:.2f}, ROC AUC: {roc_auc_dt:.2f}, MAE: {mae_dt:.2f}")

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(decision_tree, X, y, cv=5, scoring='accuracy')
print(f"Decision Tree - Average Cross-validation score: {np.mean(cv_scores_dt):.2f}")


# In[ ]:


# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, F1 Score: {f1_rf:.2f}, ROC AUC: {roc_auc_rf:.2f}, MAE: {mae_rf:.2f}")

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(random_forest, X, y, cv=5, scoring='accuracy')
print(f"Random Forest - Average Cross-validation score: {np.mean(cv_scores_rf):.2f}")


# In[ ]:


# SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print(f"SVM - Accuracy: {accuracy_svm:.2f}, F1 Score: {f1_svm:.2f}, ROC AUC: {roc_auc_svm:.2f}, MAE: {mae_svm:.2f}")


# In[ ]:



# Ensure you are using a linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Get the coefficients
coefficients = svm_model.coef_

# Coefficients can be used as an indication of feature importance for linear SVM


# In[ ]:


# XGBoost
xgboost_classifier = XGBClassifier()
xgboost_classifier.fit(X_train, y_train)
y_pred_xgb = xgboost_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost - Accuracy: {accuracy_xgb:.2f}, F1 Score: {f1_xgb:.2f}, ROC AUC: {roc_auc_xgb:.2f}, MAE: {mae_xgb:.2f}")

# Cross-validation for XGBoost
cv_scores_xgb = cross_val_score(xgboost_classifier, X, y, cv=5, scoring='accuracy')
print(f"XGBoost - Average Cross-validation score: {np.mean(cv_scores_xgb):.2f}")


# In[ ]:


# Naive Bayes
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
y_pred_nb = naive_bayes_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, y_pred_nb)
mae_nb = mean_absolute_error(y_test, y_pred_nb)

# Cross-validation for Naive Bayes
cv_scores_nb = cross_val_score(naive_bayes_classifier, X, y, cv=5, scoring='accuracy')
print(f"Naive Bayes - Average Cross-validation score: {np.mean(cv_scores_nb):.2f}")
print(f"Naive Bayes - Accuracy: {accuracy_nb:.2f}, F1 Score: {f1_nb:.2f}, ROC AUC: {roc_auc_nb:.2f}, MAE: {mae_nb:.2f}")


# In[ ]:


# Hyperparameter tuning (example for Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)


# In[ ]:


# Predictions from individual models
y_pred_lr = logistic_regression.predict(X_test)
y_pred_dt = decision_tree.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)
y_pred_xgb = xgboost_classifier.predict(X_test)
y_pred_nb = naive_bayes_classifier.predict(X_test)

# Convert predictions to probability space if necessary
# For example, if your models output class labels, you might need to use predict_proba to get probabilities
# Here, we assume all models output probabilities and that you're dealing with a binary classification problem

# Averaging the predictions
y_pred_ensemble = np.mean([y_pred_lr, y_pred_dt, y_pred_rf, y_pred_svm, y_pred_xgb, y_pred_nb], axis=0)

# Converting averaged probabilities to class labels (if your models output probabilities)
# For binary classification, this might look like:
y_pred_ensemble_class = (y_pred_ensemble > 0.5).astype(int)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble_class)
f1_ensemble = f1_score(y_test, y_pred_ensemble_class)
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble_class)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble_class)

print(f"Ensemble Model - Accuracy: {accuracy_ensemble:.2f}, F1 Score: {f1_ensemble:.2f}, ROC AUC: {roc_auc_ensemble:.2f}, MAE: {mae_ensemble:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:


# Save each individual model
joblib.dump(logistic_regression, 'logistic_regression_model.pkl')
joblib.dump(decision_tree, 'decision_tree_model.pkl')
joblib.dump(random_forest, 'random_forest_model.pkl')
joblib.dump(svm_classifier, 'svm_classifier_model.pkl')
joblib.dump(xgboost_classifier, 'xgboost_classifier_model.pkl')
joblib.dump(naive_bayes_classifier, 'naive_bayes_classifier_model.pkl')


# In[ ]:


# Save the best model
# Replace 'best_model' with your chosen model
best_model = random_forest
joblib.dump(best_model, 'best_model.pkl')


# In[ ]:


# Model names
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'XGBoost', 'Naive Bayes', 'Ensemble']

# Performance metrics for each model
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_xgb, accuracy_nb, accuracy_ensemble]
f1_scores = [f1_lr, f1_dt, f1_rf, f1_svm, f1_xgb, f1_nb, f1_ensemble]
roc_aucs = [roc_auc_lr, roc_auc_dt, roc_auc_rf, roc_auc_svm, roc_auc_xgb, roc_auc_nb, roc_auc_ensemble]
maes = [mae_lr, mae_dt, mae_rf, mae_svm, mae_xgb, mae_nb, mae_ensemble]

# Function to create bar plots
def create_bar_plot(data, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(models, data, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Visualize Accuracy
create_bar_plot(accuracies, 'Model Accuracy Comparison', 'Accuracy')

# Visualize F1 Score
create_bar_plot(f1_scores, 'Model F1 Score Comparison', 'F1 Score')

# Visualize ROC AUC
create_bar_plot(roc_aucs, 'Model ROC AUC Comparison', 'ROC AUC')

# Visualize MAE
create_bar_plot(maes, 'Model Mean Absolute Error Comparison', 'MAE')


# In[ ]:


# Build the model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


# In[ ]:


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Early stopping
#early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stop])

# Model summary
model.summary()


# In[ ]:


# Evaluate the model
scores = model.evaluate(X_test, y_test)
print(f'Test loss: {scores[0]}, Test accuracy: {scores[1]}')


# In[ ]:


# Save the model
model.save('my_model.h5')


# In[ ]:


# Predictions for metrics calculation
y_pred = model.predict(X_test)
y_pred = [1 if y > 0.5 else 0 for y in y_pred]

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[ ]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


# Predictions from individual machine learning models
y_pred_lr = logistic_regression.predict(X_test)
y_pred_dt = decision_tree.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)
y_pred_xgb = xgboost_classifier.predict(X_test)
y_pred_nb = naive_bayes_classifier.predict(X_test)


# In[ ]:


# Prediction from deep learning model
y_pred_dl = model.predict(X_test).reshape(-1)
y_pred_dl_class = (y_pred_dl > 0.5).astype(int)  # Convert probabilities to class labels


# In[ ]:


# Averaging the predictions (including deep learning model)
y_pred_ensemble = np.mean([y_pred_lr, y_pred_dt, y_pred_rf, y_pred_svm, y_pred_xgb, y_pred_nb, y_pred_dl_class], axis=0)

# Print the averaged predictions
print("Averaged predictions from the ensemble model:")
print(y_pred_ensemble)


# In[ ]:


# Converting averaged probabilities to class labels
y_pred_ensemble_class = (y_pred_ensemble > 0.5).astype(int)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble_class)
f1_ensemble = f1_score(y_test, y_pred_ensemble_class)
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble_class)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble_class)

print(f"Ensemble Model - Accuracy: {accuracy_ensemble:.2f}, F1 Score: {f1_ensemble:.2f}, ROC AUC: {roc_auc_ensemble:.2f}, MAE: {mae_ensemble:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:


# Train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_

# Sort the feature importances in descending order and plot
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title('Feature Importance (Random Forest)')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


# Train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.title('Feature Importance (XGBoost)')
plt.show()


# In[ ]:



# Example with Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:


# Compute ROC curve for a model
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])

# Plot Precision-Recall curve
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X, y=y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[ ]:


# Evaluate each model with a confusion matrix
models = [logistic_regression, decision_tree, random_forest, svm_classifier, naive_bayes_classifier, xgboost_classifier]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Naive Bayes', 'XGBoost']

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


# In[ ]:





# In[ ]:


# Feature importance (example for Random Forest)
importances = random_forest.feature_importances_
indices = np.argsort(importances)
features = X.columns

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


X = pd.DataFrame(data)

# Visualizing data before standardization
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)
plt.title("Data Before Standardization")
plt.show()

# Standardizing the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

# Visualizing data after standardization
plt.figure(figsize=(10, 6))
sns.boxplot(data=X_standardized)
plt.title("Data After Standardization")
plt.show()


# In[ ]:


# Generate a simple binary classification dataset
X, y = make_classification(n_samples=1000, n_features=12, n_classes=2, n_clusters_per_class=1, flip_y=0.03, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate AUC
auc = roc_auc_score(y_test, y_probs)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# Train the Decision Tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(decision_tree, filled=True, feature_names=X, class_names=['Class1', 'Class2'], proportion=True)
plt.title('Decision Tree')
plt.show()


# In[ ]:




