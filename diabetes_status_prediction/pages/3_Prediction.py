import streamlit as st
import pandas as pd
import numpy as np
import time

import pickle
import lightgbm
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
import os


@st.cache_data
def load_model():
    model_path = 'D:/Bootcamp/Dibimbing/Final Project/try deploy/model/xgb_clf.pkl'  # Sesuaikan dengan lokasi file model Anda
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.subheader("Fill in your current information below and we'll guess you have diabetes or not!")
#1
HighBP = st.radio('Do you have High Blood Pressure?', ['Yes', 'No'])
if HighBP == 'Yes':
    HighBP = 1
else:
    HighBP = 0
#2
HighChol = st.radio('Do you have High Cholestrol?', ['Yes', 'No'])
if HighChol == 'Yes':
    HighChol = 1
else:
    HighChol = 0
#15
GenHlth = st.radio('Would you say that in general your health is',['Excellent','Very Good','Good','Fair','Poor'])
if GenHlth == 'Excellent':
    GenHlth = 1
elif GenHlth == 'Very Good':
    GenHlth = 2
elif GenHlth == 'Good':
    GenHlth = 3
elif GenHlth == 'Fair':
    GenHlth = 4
elif GenHlth == 'Poor':
    GenHlth = 5
#3
weight=st.number_input('What is your weight in kg?',min_value=0.0, format='%.2f')
height=st.number_input('What is your height in m?',min_value=0.0, format='%.2f')
if height > 0:
    BMI = weight / (height ** 2)
    BMI = round(BMI, 2)
else:
    BMI = 0
    st.write("Height must be greater than zero to calculate BMI.")
#4
# Input pengguna untuk umur
Age_options = [
    'Age 18 to 24',
    'Age 25 to 29',
    'Age 30 to 34',
    'Age 35 to 39',
    'Age 40 to 44',
    'Age 45 to 49',
    'Age 50 to 54',
    'Age 55 to 59',
    'Age 60 to 64',
    'Age 65 to 69',
    'Age 70 to 74',
    'Age 75 to 79',
    'Age 80 or older'
]

Age = st.radio('What is your age?', Age_options)
if Age == 'Age 18 to 24':
    Age = 1
elif Age == 'Age 25 to 29':
    Age = 2
elif Age == 'Age 30 to 34':
    Age = 3
elif Age == 'Age 35 to 39':
    Age = 4
elif Age == 'Age 40 to 44':
    Age = 5
elif Age == 'Age 45 to 49':
    Age = 6
elif Age == 'Age 50 to 54':
    Age = 7
elif Age == 'Age 55 to 59':
    Age = 8
elif Age == 'Age 60 to 64':
    Age = 9
elif Age == 'Age 65 to 69':
    Age = 10
elif Age == 'Age 70 to 74':
    Age = 11
elif Age == 'Age 75 to 79':
    Age = 12
elif Age == 'Age 80 or older':
    Age = 13
#5
DiffWalk =st.radio('Do you have difficulty walking?',['Yes','No'])
if DiffWalk == 'Yes':
    DiffWalk = 1
else:
    DiffWalk = 0
#6
Income_options = ['<$10 K', '$10–$15 K', '$15–$20 K', '$20–$25 K', '$25–$35 K', '$35–$50 K', '$50–$75 K', '>$75 K']
Income = st.radio('How much is your annual income?', Income_options) 
if Income == '<$10 K':
    Income = 1
elif Income == '$10–$15 K':
    Income = 2
elif Income == '$15–$20 K':
    Income = 3
elif Income == '$20–$25 K':
    Income = 4
elif Income == '$25–$35 K':
    Income = 5
elif Income == '$35–$50 K':
    Income = 6
elif Income == '$50–$75 K':
    Income = 7
elif Income == '>$75 K':
    Income = 8
#7
PhysActivity=st.radio('Do you have physical activity or exercise during the past 30 days other than regular work?',['Yes','No'])
if PhysActivity == 'Yes':
    PhysActivity = 1
else:
    PhysActivity=0
#8
HeartDiseaseorAttack=st.radio('Do you Have coronary hear disease?',['Yes','No'])
if HeartDiseaseorAttack=='Yes':
    HeartDiseaseorAttack=1
else :
    HeartDiseaseorAttack=0
#9
Stroke = st.radio('Have you ever had a stroke?', ['Yes', 'No'])
Stroke = 1 if Stroke == 'Yes' else 0
#10
HvyAlcoholConsump = st.radio('Do you consume alcohol heavily?', ['Yes', 'No'])
HvyAlcoholConsump = 1 if HvyAlcoholConsump == 'Yes' else 0
#11
Smoker = st.radio('Have you smoked at least 100 cigarettes in your entire life?', ['Yes', 'No'])
Smoker = 1 if Smoker == 'Yes' else 0
#12
Veggies = st.radio('Do you eat veggies regularly?',['Yes','No'])
Veggies = 1 if Veggies == 'Yes' else 0
#13
MentHlth=st.number_input('Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?',min_value=0, max_value=31, step=1)
#14
PhysHlth=st.number_input('How many days during the past 30 days was your physical health not good?',min_value=0, max_value=31, step=1)
#16
Education_options = [
    'Never attended school or only kindergarten',
    'Grades 1 through 8 (Elementary)',
    'Grades 9 through 11 (Some high school)',
    'Grade 12 or GED (High school graduate)',
    'College 1 year to 3 years (Some college or technical school)',
    'College 4 years or more (College graduate)'
]

Education = st.radio('What is your education level?', Education_options)

if Education == 'Never attended school or only kindergarten':
    Education = 1
elif Education == 'Grades 1 through 8 (Elementary)':
    Education = 2
elif Education == 'Grades 9 through 11 (Some high school)':
    Education = 3
elif Education == 'Grade 12 or GED (High school graduate)':
    Education = 4
elif Education == 'College 1 year to 3 years (Some college or technical school)':
    Education = 5
elif Education == 'College 4 years or more (College graduate)':
    Education = 6

your_diabetes_data = pd.DataFrame({
    'HighBP':[HighBP],
    'HighChol':[HighChol],
    'BMI' :[BMI],
    'Smoker':[Smoker],
    'Stroke':[Stroke],
    'HeartDiseaseorAttack':[HeartDiseaseorAttack],
    'PhysActivity':[PhysActivity],
    'Veggies':[Veggies],
    'HvyAlcoholConsump':[HvyAlcoholConsump],
    'GenHlth':[GenHlth],
    'MentHlth':[MentHlth],
    'PhysHlth':[PhysHlth],
    'DiffWalk':[DiffWalk],
    'Age':[Age],
    'Education':[Education],
    'Income':[Income] 
})
if st.button('Let\'s Predict!'):
    your_diabetes_status = model.predict(your_diabetes_data)
    your_diabetes_status = your_diabetes_status[0]
    diabetes_label = "Diabetes" if your_diabetes_status == 1 else "Non-Diabetes"  
    st.write(f"### Your diabetes status is predicted as: {diabetes_label}")






