import altair as alt
import pandas as pd
import streamlit as st

st.title("📊 CSV Explorer")
st.write("Upload a CSV file to explore its contents interactively.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Preview")
    rows = st.slider("Rows to display", 5, min(100, len(df)), 10)
    st.dataframe(df.head(rows), use_container_width=True)

    st.subheader("Summary statistics")
    summary_df = df.describe(include="all").T
    for col in summary_df.columns:
        if summary_df[col].dtype == "object":
            summary_df[col] = summary_df[col].astype(str)
    st.dataframe(summary_df, use_container_width=True)

    numeric_cols = df.select_dtypes("number").columns.tolist()
    if numeric_cols:
        st.subheader("Histogram")
        col = st.selectbox("Select numeric column", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 30)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
                y="count()",
                tooltip=["count()"],
            )
            .properties(width=600)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns found for histogram.")
else:
    st.info("Awaiting CSV upload.")
