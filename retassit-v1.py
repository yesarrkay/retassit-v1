import streamlit as st
import pandas as pd
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import plost
import matplotlib.pyplot as plt
from PIL import Image
from churn_viz import show_churn_viz

from dotenv import load_dotenv
load_dotenv()   

openai_api_key=os.environ.get('OPENAI_API_KEY')
# Get the API key from the environment
openai_api_key = os.environ.get('OPENAI_API_KEY')
# Check if the key exists and print it
if openai_api_key:
    print(openai_api_key)
else:
    print("OPENAI_API_KEY not found in environment variables.")

cluster_descriptions = [
        "High Credit Limt, Low Utilization",
        "Moderate Credit Limit, Moderate Utilization",
        "Lower Credit Limit, Higher Utilization",
        "Highly Detractor, Frequent Caller",
    ]


def app():    
    st.set_page_config(page_title="Retention Assit",page_icon="https://www.logo.wine/a/logo/Genpact/Genpact-Logo.wine.svg",layout='wide', initial_sidebar_state='expanded')
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.sidebar.image('Genpact.png')
        st.sidebar.header('Cora Retention App `ver 1`')
        st.sidebar.markdown('Rapid Prototype @CXMAnalytics ')
        st.sidebar.markdown('''
        ---
        ''')

        # st.sidebar.subheader('Heat map parameter')
        # time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

        # st.sidebar.subheader('Donut chart parameter')
        # donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

        # st.sidebar.subheader('Line chart parameters')
        # plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
        # plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

        st.sidebar.write("Upload Customer Data for Prediction.")
        file =  st.sidebar.file_uploader("Upload CSV file",type=["csv"],key="customer data")
        # file='data/Small Set_Customer Data.csv'  


    st.title("Cora Retention Assit.AI")
    st.subheader("Generative AI App - OpenAI \U0001F916 and  LangChain 🦜")
    # st.subheader("Cora Retention Assit.AI")
    
    show_churn_viz()
    st.subheader("Insights")
    st.write("Churn continues to raise for cards product. The primary reason being service delivery issues and mostly impacting new customers. Recommend to optimize retention stratagies for this segment")


    if not file:
        st.stop()
    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head()) 
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 
    query = st.text_input("Enter a query:",key="intial")

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)

    cluster_checkbox = st.checkbox("Run Gen AI RetAssit V1 prediction model to build cluster")
    if cluster_checkbox :
        # Display a new table with 6 columns and 5 rows
        table_data = {
            
            "Number of Customers": [111, 79, 13,],
            "Average Age": [49, 47, 47,],
            "Average Credit Limit": [30064, 10290, 4321,],
            "Average Total Revolving Balance": [1768, 1483, 1534,],
            "Average Utilization Ratio(%)": [6.34, 27.19, 47.14,]
        }
        st.table(pd.DataFrame(table_data))

        st.write("Select the clusters you want to work with:")
        num_clusters = 10
        clusters_selected = st.multiselect(
            "Select clusters",
            cluster_descriptions,
            default=cluster_descriptions
        )        

        if len(clusters_selected) >0:
            st.write("selected clusters:",clusters_selected)
            st.sidebar.write("Upload Current marketing promotions")
            file =  st.sidebar.file_uploader("Upload CSV file",type=["csv"],key="campaign data")
            # file='data/updated_marketing_campaign_promotion.csv' 
            if not file:
                st.stop()
            data = pd.read_csv(file)
            st.write("Data Preview:")
            st.dataframe(data.head()) 

            query = st.text_input("Enter your prompt:",key="final") 
    
            if st.button("Ask LLM"):
                st.write("Answer:")
                # st.write(answer)
                data_path = 'data/updated_unique_output_table.csv'
                df = pd.read_csv(data_path)
                # Display the data in a table format
                st.write("## Retention strategies Customer Cluster and Campaign Data")
                st.write(df)
                # Download link for the data
                st.markdown(f"[Download the data]({data_path})")

if __name__ == "__main__":
    app()   