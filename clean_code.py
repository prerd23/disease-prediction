from tkinter import *
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import os
import sys


# Get the base directory for the .exe to properly locate files
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller executable """
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# List of Symptoms (Ensure this matches the dataset exactly)
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech',
      'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
      'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma',
           'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
           'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
           'Tuberculosis', 'Common Cold', 'Pneumonia', 'Hemorrhoids (piles)',
           'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
           'Osteoarthritis', 'Arthritis', 'Positional Vertigo', 'Acne', 'Urinary tract infection',
           'Psoriasis', 'Impetigo']

# Load Training Data
df = pd.read_csv(resource_path("Training.csv"))
tr = pd.read_csv(resource_path("Testing.csv"))

# Ensure correct feature selection
X = df[df.columns.intersection(l1)]
y = np.ravel(df[["prognosis"]])

X_test = tr[tr.columns.intersection(l1)]
y_test = np.ravel(tr[["prognosis"]])


# Decision Tree Model
def DecisionTree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    # Calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

    # Predict disease based on user input
    predict_disease(clf)


def predict_disease(model):
    """ Predict disease based on user input and display result. """
    symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Ensure input vector matches training feature set
    feature_names = X.columns
    input_vector = [1 if symptom in symptoms else 0 for symptom in feature_names]

    # Predict disease
    predicted_disease = model.predict([input_vector])[0]  # This returns a string

    # Display result
    result_text.delete("1.0", END)
    result_text.insert(END, predicted_disease)

    # Show accuracy in the GUI
    accuracy_label.config(text=f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")


# GUI Setup
root = Tk()
root.title("Disease Prediction System - Decision Tree")
root.configure(background='lightblue')

# Variables
Symptom1 = StringVar(value="None")
Symptom2 = StringVar(value="None")
Symptom3 = StringVar(value="None")
Symptom4 = StringVar(value="None")
Symptom5 = StringVar(value="None")
Name = StringVar()

# Main Title
Label(root, text="Disease Predictor (Decision Tree)", font=("Arial", 20, "bold"), bg="lightblue").grid(row=0, column=0,
                                                                                                       columnspan=2,
                                                                                                       pady=10)

# Patient Information
Label(root, text="Enter Patient Name:", bg="lightblue").grid(row=1, column=0, sticky=W)
Entry(root, textvariable=Name).grid(row=1, column=1)

# Symptom Selection
symptom_labels = ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", "Symptom 5"]
symptom_vars = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

for i, (label, var) in enumerate(zip(symptom_labels, symptom_vars)):
    Label(root, text=label, bg="lightblue").grid(row=2 + i, column=0, sticky=W)
    OptionMenu(root, var, "None", *sorted(l1)).grid(row=2 + i, column=1)

# Predict Button
Button(root, text="Predict Disease", command=DecisionTree, bg="green", fg="white", font=("Arial", 12)).grid(row=7,
                                                                                                            column=0,
                                                                                                            columnspan=2,
                                                                                                            pady=10)

# Accuracy Label
accuracy_label = Label(root, text="Model Accuracy: ", bg="lightblue", font=("Arial", 10))
accuracy_label.grid(row=8, column=0, columnspan=2)

# Result Field
Label(root, text="Predicted Disease:", bg="lightblue").grid(row=9, column=0, sticky=W)
result_text = Text(root, height=1, width=40, bg="white", font=("Arial", 10))
result_text.grid(row=9, column=1, padx=10, pady=5)

root.mainloop()