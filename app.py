import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.sql_utils import load_into_sqlite, query_sqlite
from src.eda import plot_sales_distribution
from src.market_basket import run_market_basket
from src.segmentation import kmeans_segmentation
from src.recommender import simple_content_based
from src.groq import ask_groq   # Changed from groq_integration to groq

st.set_page_config(page_title="Supermall Sales Recommender", layout="wide")

st.title("🛍 Supermall Sales & Recommendation System")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.dataframe(df.head())

    # Load into SQLite
    if st.button("Load into SQLite"):
        load_into_sqlite(df, "sales")
        st.success("Data loaded into SQLite!")

    # EDA
    if st.button("Run EDA"):
        st.pyplot(plot_sales_distribution(df))

    # Market Basket Analysis
    if st.button("Run Market Basket Analysis"):
        if 'item' in df.columns:
            rules = run_market_basket(pd.get_dummies(df['item']), min_support=0.05)
            st.dataframe(rules)
        else:
            st.error("Column 'item' not found in dataset.")

    # Segmentation
    if st.button("Run Segmentation"):
        if 'sales' in df.columns:
            segmented_df, model = kmeans_segmentation(df, features=["sales"])
            st.dataframe(segmented_df)
        else:
            st.error("Column 'sales' not found in dataset.")

    # Recommendation
    target_item = st.text_input("Enter item for recommendations")
    if target_item:
        if 'item' in df.columns:
            recs = simple_content_based(df, "item", target_item)
            st.dataframe(recs)
        else:
            st.error("Column 'item' not found in dataset.")

    # Groq Chat
    prompt = st.text_area("Ask Groq about your sales data")
    if st.button("Ask Groq") and prompt:
        try:
            response = ask_groq(prompt)
            st.write(response)
        except Exception as e:
            st.error(f"Error contacting Groq: {e}")
