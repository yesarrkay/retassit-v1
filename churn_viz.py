import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and cache the data
@st.cache_data
def load_data():
    return pd.read_csv('data/updated_customer_data_v15.csv')

# Applying the CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_churn_viz():
    local_css("style.css")

    # Load the data
    data = load_data()

    # Create a year filter
    selected_years = st.multiselect("Select Years for Analysis", sorted(data['Year'].unique()), default=sorted(data['Year'].unique()))
    filtered_data = data[data['Year'].isin(selected_years)]

    # Create three columns for the visualizations
    col1, col2, col3 = st.columns([1,1,1])  # Equally spaced columns

    # Visualization for the first column
    with col1:
        st.subheader("Customer Growth")
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered_data.groupby(pd.PeriodIndex(filtered_data['Year'].astype(str), freq='Q'))[['Active customer base', 'Number of Customer Acquired', 'Customer Churn']].mean().plot(ax=ax)
        ax.set_xlabel("Year (Quarterly)")
        ax.set_ylabel("Active Customers (in Millions)")
        st.pyplot(fig)

    # Visualization for the second column
    with col2:
        st.subheader("Churn vs MOB")
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered_data.groupby(pd.PeriodIndex(filtered_data['Year'].astype(str), freq='Q'))['Churn customer Average Month on Book'].mean().plot(ax=ax)
        ax.set_xlabel("Year (Quarterly)")
        ax.set_ylabel("MOB (in Months)")
        st.pyplot(fig)

    # Heatmap for the third column
    with col3:
        st.subheader("Product Type & Reason")
        churn_cols = ['Churn by Product- Bank %', 'Churn by Product- Cards %', 'Churn Product- Investment %', 'Reason for Churn - Service (%)', 'Reason for Churn - Price (%)']
        heatmap_data = filtered_data.groupby(pd.PeriodIndex(filtered_data['Year'].astype(str), freq='Q'))[churn_cols].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heatmap_data.T, cmap='YlGnBu', ax=ax)
        st.pyplot(fig)
