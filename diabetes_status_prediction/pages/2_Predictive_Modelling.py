import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import dalex as dx
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.title('Model Performance Metrics')

# Dataframe for model performance metrics
metrics_data = {
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
    'Logistic Regression': [0.741159, 0.776321, 0.758333, 0.742053],
    'Random Forest Classifier': [0.716640, 0.774327, 0.744368, 0.722742],
    'XGBoost Classifier': [0.800399, 0.730844, 0.764042, 0.742276]
}
metrics_df = pd.DataFrame(metrics_data)
def main():

    # Display metrics table
    st.subheader('Metrics Data')
    st.write(metrics_df)

    # Visualize metrics using a bar chart
    st.subheader('Metrics Visualization')
    fig, ax = plt.subplots(figsize=(12, 10))
    metrics_df.set_index('Metric').plot(kind='bar', ax=ax)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    st.pyplot(fig)

if __name__ == '__main__':
    main()
st.markdown('XGBoost Classifier has the best performance with accuracy 74%')