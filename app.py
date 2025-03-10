import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title('Iris Model Inference')

with st.sidebar:
    st.header('Data Requirements')
    st.caption('Upload a CSV file with the following columns: sepal_length, sepal_width, petal_length, petal_width')
    with st.expander('Data Format'):
        st.markdown(' - utf -8')
        st.markdown(' - csv')
        st.markdown(' - No missing values')
        st.markdown(' - No categorical values')
    st.divider()
    st.caption("<p style = 'text-align:center'>Develped by Hasitha</p>", unsafe_allow_html = True)

# if st.button("Let's get started"):
#     upload_data = st.file_uploader('Choose a file', type = ['csv'])
    

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

    
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    upload_file = st.file_uploader('Choose a file', type = ['csv'])
    if upload_file is not None:
        df = pd.read_csv(upload_file, low_memory=True)
        st.header('Uploaded Data Sample')
        st.write(df.head())
        model = joblib.load('model.joblib')
        pred = model.predict_proba(df)  
        pred = pd.DataFrame(pred, columns = ['setosa', 'versicolor', 'virginica'])
        st.header('Predicted Values')
        st.write(pred.head())
        
        pred = pred.to_csv(index = False).encode('utf-8')
        st.download_button(label = 'Download Predictions', data = pred, file_name = 'predictions.csv', mime = 'text/csv', key='download-csv')
