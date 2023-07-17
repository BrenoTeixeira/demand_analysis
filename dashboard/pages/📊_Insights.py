import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.ticker as mtick


@st.cache_data
def load_data(path):

    data = pd.read_csv(path, low_memory=True, parse_dates=['date'])

    return data


def plots_configs():
     
    plt.rcParams.update({'figure.facecolor': (0.0, 0.0, 0.0, 0.0),
                         'axes.facecolor': (0.0, 0.0, 0.0, 0.0),
                         'savefig.facecolor': (0.0, 0.0, 0.0, 0.0)})
     
    plt.rcParams['font.size'] = 25
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['text.color'], plt.rcParams['axes.labelcolor'], plt.rcParams['xtick.color'], plt.rcParams['ytick.color'] = 'white', 'white', 'white', 'white'
    plt.rcParams['lines.linewidth'] = 6
    plt.rcParams['lines.markersize'] = 10
    #plt.rcParams['font']


def plot_most_sales_by(df: pd.DataFrame, variable: str, target: str, max_categories: int):

    colors = ['teal' if i <5 else 'gray' for i in range(max_categories+1)]

    assert isinstance(df, pd.DataFrame), 'df must be a DataFrame'
    assert isinstance(variable, str), 'variable must be a string'
    assert isinstance(target, str), 'target must be a string'
    assert isinstance(max_categories, int), 'max_categories must be a integer'

    fig, ax = plt.subplots(1, 1, figsize=(25, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plot_df = df.groupby(variable).sum(numeric_only=True)[[target]].sort_values(target, ascending=False).reset_index().loc[:max_categories]

    bar = sns.barplot(data=plot_df, x=variable, y=target, order=plot_df[variable][:max_categories], ax=ax,palette=colors, )
    plt.bar_label(bar.containers[0], fmt='{:,.0f}', label_type='edge')

    ax.ticklabel_format(axis='y', style='plain')
   
    ax.set_ylabel(target, fontdict={'fontsize': 25})


    ax.set_yticklabels('')
    ax.set_yticks([])

    ax.grid(False);
    return fig


def plot_freq_sales(df_eda, freq='M'):


    df_series = df_eda.copy()[['date', 'sales']].set_index('date')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 8))
    ax.plot(df_series.resample(freq).sum(), color='teal', marker='o')
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(left=False, bottom=False,)
    ax.ticklabel_format(axis='y', style='plain')
    
    
    ax.grid(False)
    return fig


def growth(df_eda, freq='M'):


    df_series = df_eda.copy()[['date', 'sales']].set_index('date')
    df_series_ord = df_series.sort_index().copy().resample(freq).sum()
    df_series_ord['prev_sales'] = df_series_ord.shift()#.fillna(0)
    df_series_ord['growth_%'] = ((df_series_ord['sales'] - df_series_ord['prev_sales'])/df_series_ord['prev_sales'])*100
    fig, ax = plt.subplots(1, 1, figsize=(25, 8))
    ax = df_series_ord['growth_%'].plot(marker='o', color='teal')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter());
    return fig

def plot_yearly_sales(df_eda):


    df_series = df_eda.copy()[['date', 'sales']].set_index('date')


def page_sidebar(df):


    st.sidebar.markdown('## Insights')
    products = st.sidebar.multiselect(label='Sales Trend Filter - Select the product', options=df.item.unique(), default=df.item.unique())

    return list(products)


def image_header(image_path):

    with st.container():

        image = Image.open(image_path)
        new_image = image.resize((800, 200))
        st.image(image=new_image, width=800)


def main():

    st.set_page_config(page_title='Insights', page_icon=":bar_chart:")

    plots_configs()

    header_image = 'images/insights.jpg'
    image_header(image_path=header_image)
    

    path = '../DATA/train.csv'
    df_eda = load_data(path)

    st.markdown('# Sales Insights')

    st.write('This page shows the insights obtained through the Exploratory Data Analysis.')

    products = page_sidebar(df_eda)

    st.markdown("<h2 style='text-align: center'>Top 10 most sold products </h2>", unsafe_allow_html=True)

    st.pyplot(plot_most_sales_by(df=df_eda, variable='item', target='sales', max_categories=10))

    st.markdown("<h2 style='text-align: center'>Top 10 stores with most sales </h2>", unsafe_allow_html=True)

    st.pyplot(plot_most_sales_by(df=df_eda, variable='store', target='sales', max_categories=10))

    (st.markdown('# Sales Trend'))

    st.markdown("<h2 style='text-align: center'> Sales per year </h2>", unsafe_allow_html=True)

    st.pyplot(plot_freq_sales(df_eda=df_eda.query(f'item.isin({products})'), freq='A'))

    st.markdown("<h2 style='text-align: center'> Sales per month </h2>", unsafe_allow_html=True)

    st.pyplot(plot_freq_sales(df_eda=df_eda.query(f'item.isin({products})'), freq='M'))

    st.markdown("<h2 style='text-align: center'> Sales Growth % over month </h2>", unsafe_allow_html=True)
    st.pyplot(growth(df_eda=df_eda.query(f'item.isin({products})'), freq='M'))

    st.markdown("<h2 style='text-align: center'> Sales Growth % over year</h2>", unsafe_allow_html=True)
    st.pyplot(growth(df_eda=df_eda.query(f'item.isin({products})'), freq='A'))

if __name__ == '__main__':
    
    main()


