import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.title("👋 Welcome to Streamlit Basics")
st.write(
    "This first exercise covers the core building blocks: **text**, **widgets**, **layout**, and **charts**."
)

st.header("Text Elements")
st.subheader("Write anything")
st.code("st.write('Hello, *Streamlit*!')")
st.write("Below is what the command above would output:")
st.write("Hello, *Streamlit*!")


st.header("Widgets")
if st.button("Press me"):
    st.success(
        "You pressed the button! Streamlit re-runs the script top-to-bottom on every interaction."
    )

number = st.slider("Pick a number", 0, 100, 42)
st.write(f"You selected **{number}**, its square is **{number**2}**.")


st.header("Layout: columns")
col1, col2 = st.columns(2)
with col1:
    st.metric("Current number", number)
with col2:
    progress_placeholder = st.empty()
    for i in range(101):
        progress_placeholder.progress(i)
        time.sleep(0.01)


st.header("Charts")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"],
)

st.write("Here's a line chart of random data:")
st.line_chart(chart_data)

st.write("And the same data as an Altair area chart:")
area_chart = (
    alt.Chart(chart_data.reset_index())
    .transform_fold(["a", "b", "c"], as_=["Series", "Value"])
    .mark_area(opacity=0.3)
    .encode(
        x="index:Q",
        y="Value:Q",
        color="Series:N",
    )
)
st.altair_chart(area_chart, use_container_width=True)

st.caption(
    "🎉 Congratulations! You've just built your first interactive Streamlit app."
)
