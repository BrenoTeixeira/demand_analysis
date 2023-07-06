import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from streamlit_card import card

def load_data(path):

    data = pd.read_csv(path, low_memory=True, parse_dates=['date'])

    return data


def sales(data):

    return data['sales'].sum()


def products(data):

    return data.item.nunique()


def stores(data):

    return data.store.nunique()


def monthly_sales(data):

    data_ = data.set_index('date')
    avg_monthly_sales = data_.resample('M').sum()['sales'].mean()

    return avg_monthly_sales

def weekly_sales(data):

    data_ = data.set_index('date')
    avg_weekly_sales = data_.resample('W').sum()['sales'].mean()

    return avg_weekly_sales


if __name__ == '__main__':

    st.set_page_config(
        page_title='Home Page',
        page_icon="school",
        layout='wide'
    )
    st.sidebar.success('Select the page above')

    path='../DATA/train.csv'
    data = load_data(path)

    st.markdown('# Welcome to the Demand Analysis Report')

    st.markdown('## Data')
    st.write(data.head(6))

    

    total_sales, number_of_prodcts, number_of_stores, avg_monthly_sales, avg_weekly_sales = st.columns(5)
    
   

    total_sales.metric(label='   Total Sales', value=f'{sales(data):,.0f}', )
    number_of_prodcts.metric(label='Number of Products', value=f'{products(data)}')
    number_of_stores.metric(label='Number of Stores', value=f'{stores(data)}')
    avg_monthly_sales.metric(label='Avg Monthly Sales', value=f'{monthly_sales(data):,.1f}')

    avg_weekly_sales.metric(label='Avg Weekly Sales', value=f'{weekly_sales(data):,.1f}')


   
    