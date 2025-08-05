from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Text Analyzer", layout="wide")

st.title("📝 Interactive Text Analyzer")
st.write(
    "Paste or type any text below and click **Analyze** to see quick statistics and the most common words."
)

text_input = st.text_area("Enter your text here 👇", height=250)
analyze = st.button("Analyze")

if analyze and text_input.strip():
    words = [word.lower() for word in text_input.split()]
    chars = len(text_input)
    lines = len(text_input.splitlines())
    word_counts = Counter(words)
    vocab_size = len(word_counts)

    st.subheader("Basic Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Characters", chars)
    col2.metric("Words", len(words))
    col3.metric("Unique words", vocab_size)
    col4.metric("Lines", lines)

    most_common = pd.DataFrame(word_counts.most_common(20), columns=["word", "count"])

    with st.expander("Top 20 words", expanded=True):
        st.dataframe(most_common, use_container_width=True)

    st.subheader("Frequency plot")
    chart = (
        alt.Chart(most_common)
        .mark_bar()
        .encode(
            x=alt.X("word", sort="-y"),
            y="count",
            tooltip=["word", "count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Awaiting text input and **Analyze** button click.")
