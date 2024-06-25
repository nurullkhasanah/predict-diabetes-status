import streamlit as st
import pandas as pd
import numpy as np
import time

import folium
import branca.colormap as cm
from streamlit_folium import folium_static
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Exploratory Data Analysis')


df = pd.read_csv('dataset\diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

@st.cache_data
def show_data():
    st.write(df)
if st.checkbox('Show Data!'):
    show_data()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

st.markdown("""
            ### What is the proportion of people who have diabetes and who do not have diabetes in the data set?
            """)
@st.cache_data
def create_pie_chart(df):
    diabetes_count = df['Diabetes_binary'].value_counts()
    fig = px.pie(
        values=diabetes_count,
        names=['Non-Diabetic', 'Diabetic'],
        title='Proportion of Diabetes',
        hole=0.3
    )
    return fig

# Create and display the pie chart
fig = create_pie_chart(df)
st.plotly_chart(fig)

st.markdown("""
            ### Are people who have high cholesterol and high blood pressure susceptible to diabetes?
            """)
@st.cache_data
def create_bar_chart(df):
    df['group'] = df.apply(lambda row: f"BP:{row['HighBP']}-Chol:{row['HighChol']}", axis=1)
    prevalence = df.groupby('group')['Diabetes_binary'].mean().reset_index()
    prevalence.columns = ['group', 'prevalence']
    fig = px.bar(prevalence, x='group', y='prevalence', 
                 title='Prevalence of Diabetes Based on the Combination of High Blood Pressure and High Cholesterol',
                 labels={'group': 'Combination of High Blood Pressure and High Cholesterol', 'prevalence': 'Prevalence of Diabetes'})
    for index, row in prevalence.iterrows():
        fig.add_annotation(x=index, y=row['prevalence'], text=f"{row['prevalence']*100:.1f}%",
                           showarrow=True, arrowhead=1, ax=0, ay=-30)

    fig.update_layout(showlegend=False)
    return fig
fig_bar = create_bar_chart(df)
st.plotly_chart(fig_bar)

st.markdown("""
            ### Are obese (BMI>30) people at greater risk of diabetes?
            """)
@st.cache_data
def create_bmi_distribution(df):
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="BMI", hue="Diabetes_binary", fill=True, ax=ax)
    ax.set_title('BMI vs Diabetes')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Density')
    return fig
bmi_chart = create_bmi_distribution(df)
st.pyplot(bmi_chart)
