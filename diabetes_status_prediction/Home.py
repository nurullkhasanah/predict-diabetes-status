import streamlit as st

st.set_page_config(
    page_title="Home",
)

st.markdown("# Diabetes Prediction and Feature Importance Analysis")
st.markdown(
    """
        ### **What is Diabetes?**

    Diabetes is a chronic disease that increases risk for stroke, kidney failure, renal complications, peripheral vascular disease, heart disease, and death (Xie, 2019)

    The International Diabetes Federation(IDF) estimates that at the current growth, 693 million people will have diabetes worldwide by 2045.

    The Centers for Disease Control and Prevention(CDC) recorded that in 2012, 29.1 million people in the United State were diagnosed diabetes. This condition put high financial burden for government because of medical cost and decreased productivity.

    According to IDF data, in 2021, Indonesia is in the fifth position with 19.5 million diabetes cases and estimated will increase to 28.6 million in 2045. 
    
    Based on above condition, it is very important to develop predictive model which could help to facilitate early diagnosis of diagnosis 
  
"""
)
st.markdown("""
    ## Dataset Source
    The dataset that I use is data from a survey that was conducted by CDC through the Behavioral Risk Factor Surveillance System (BRFSS).
    It's a random-digit–dialed telephone survey of noninstitutionalized US adults aged 18 years or older.
    
    Check out the dataset in [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)

"""
)