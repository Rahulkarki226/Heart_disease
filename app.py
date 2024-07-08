import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Drop the specified columns
columns_to_drop = ['slope', 'ca', 'thal']
df.drop(columns=columns_to_drop, inplace=True)

# Heading
st.title("Heart Checkup")
st.sidebar.header('Patient Data')
st.subheader('Training Datasets')
st.write(df.describe())

# Split data into features and target variable
X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def user_report():
    age = st.sidebar.slider('Age', 0, 100, 45)
    sex = st.sidebar.radio('Sex', (0, 1), index=0, format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.slider('Chest Pain Type (cp)', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 0, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (chol)', 0, 600, 200)
    fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (restecg)', 0, 2, 0)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 0, 250, 75)
    exang = st.sidebar.radio('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', 0.0, 6.0, 1.0)

    user_report_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)
user_result = model.predict(user_data)

# Color Function
if user_result[0] == 0:
    color = 'Green'
    output = 'Healthy Heart'
else:
    color = 'red'
    output = 'Not Healthy Heart'

# Result
st.subheader('Your Report')
st.markdown(f"<h6 style='color: {color};'>{output}</h6>", unsafe_allow_html=True)
st.subheader('Accuracy: ')
st.write(str(recall_score(y_test, model.predict(X_test)) * 100) + '%')

# Initialize models
rf_classifier = RandomForestClassifier()
logreg_classifier = LogisticRegression()
svm_classifier = SVC()
tree_classifier = DecisionTreeClassifier()

# Train models
rf_classifier.fit(X_train, y_train)
logreg_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
tree_classifier.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_classifier.predict(X_test)
logreg_predictions = logreg_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
tree_predictions = tree_classifier.predict(X_test)

# Calculate accuracy and recall scores
rf_accuracy = accuracy_score(y_test, rf_predictions)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)

rf_recall = recall_score(y_test, rf_predictions)
logreg_recall = recall_score(y_test, logreg_predictions)
svm_recall = recall_score(y_test, svm_predictions)
tree_recall = recall_score(y_test, tree_predictions)

# Plotting the comparison graph
models = ['Random Forest', 'Logistic Regression', 'SVM', 'Decision Tree']
accuracies = [rf_accuracy, logreg_accuracy, svm_accuracy, tree_accuracy]
recalls = [rf_recall, logreg_recall, svm_recall, tree_recall]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette='viridis', ax=ax)
ax.set_title('Model Accuracies and Recall')
ax.set_ylim([0, 1])  # Assuming accuracies range from 0 to 1
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')

# Display recall as annotations on the bars
for i, (accuracy, recall) in enumerate(zip(accuracies, recalls)):
    ax.text(i, accuracy + 0.02, f'Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}', ha='center', va='bottom', fontsize=10)

st.pyplot(fig)
