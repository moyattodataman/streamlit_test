import streamlit as st
import pandas as pd
from io import StringIO
from prophet import Prophet

st.title("hello")
st.write("write")
st.markdown("# aaaaaaaaaaaaaaaaaa")
st.markdown("## Head2")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    m = Prophet()
    m.fit(df) 
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = m.plot(forecast)

    st.pyplot(fig)
