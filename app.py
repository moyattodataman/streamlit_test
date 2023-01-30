import streamlit as st
import pandas as pd
from io import StringIO
from prophet import Prophet

st.title("時系列分析")
st.write("テストアプリです")
st.markdown("## 仕組み")
st.markdown("[prophet](https://facebook.github.io/prophet/)で予測しています")

st.markdown("## ファイルアップロード時の注意点")
st.markdown("以下のイメージでCSVでアップロードしてください")
st.markdown("[![Image from Gyazo](https://i.gyazo.com/feb3ca41487e8a9b487f9c918d907282.png)](https://gyazo.com/feb3ca41487e8a9b487f9c918d907282)")
st.markdown(":green[$\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")

st.markdown("## さあアップロードしよう👇")
uploaded_file = st.file_uploader("")
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

    st.markdown("## あなたのアップロードしたファイルの情報")
    st.write(df)

    m = Prophet()
    m.fit(df) 
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = m.plot(forecast)

    st.markdown("## 予測結果")
    st.pyplot(fig)
