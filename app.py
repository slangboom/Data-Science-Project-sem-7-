import pickle
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

# Load the models
with open('svm_model.pkl', 'rb') as f:
    final_svm_model = pickle.load(f)

with open('nb_model.pkl', 'rb') as f:
    final_nb_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    final_rf_model = pickle.load(f)
    
##
# Load your data and encoder
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Assuming you're using the same encoder as in your original code
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

##


# Define symptom index and data dictionary
symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Define predictDisease function
# def predictDisease(symptoms):
#     symptoms = symptoms.split(",")

#     input_data = [0] * len(data_dict["symptom_index"])
#     for symptom in symptoms:
#         index = data_dict["symptom_index"][symptom]
#         input_data[index] = 1

#     input_data = np.array(input_data).reshape(1, -1)

#     rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
#     nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
#     svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

#     final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
#     predictions = {
#         "rf_model_prediction": rf_prediction,
#         "naive_bayes_prediction": nb_prediction,
#         "svm_model_prediction": svm_prediction,
#         "final_prediction": final_prediction
#     }
#     return predictions


def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = final_rf_model.predict(input_data)[0]
    nb_prediction = final_nb_model.predict(input_data)[0]
    svm_prediction = final_svm_model.predict(input_data)[0]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    return final_prediction



# Streamlit app
st.title("Disease Prediction Web App")
st.write("Enter symptoms separated by commas")

user_input = st.text_input("Symptoms")

if st.button("Predict"):
    predictions = predictDisease(user_input)
    st.write("SVM Model Prediction:", predictions["svm_model_prediction"])
    st.write("Naive Bayes Model Prediction:", predictions["naive_bayes_prediction"])
    st.write("Random Forest Model Prediction:", predictions["rf_model_prediction"])
    st.write("Final Combined Prediction:", predictions["final_prediction"])
