import streamlit as st
from graph_builder import graph

st.title("🧥 StyloGenie - AI Wardrobe Assistant")

user_input = st.text_input("👤 What are you wearing today or need help with?")
if st.button("✨ Get Suggestion") and user_input:
    with st.spinner("Thinking in style..."):
        result = graph.invoke({"input": user_input})
        st.success(result["generate_suggestion"])
