import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as sp
from PIL import Image

# Load Data
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
# Filter
def top_11_items(data):

    sales_df = data.copy()[data.date.dt.year >= 2016]
    sales_by_prod = sales_df.groupby('item').sum(numeric_only=True).sort_values('sales', ascending=False).reset_index().drop(columns='store')


    # Accumulated sales per produc
    sales_by_prod['acc_sales'] = sales_by_prod.sales.cumsum()

    sales_by_prod['sales_perc'] = sales_by_prod['acc_sales']/sales_by_prod.sales.sum()

    sales_by_prod['product_%'] = np.array([i + 1 for i in range(len(sales_by_prod))])/len(sales_by_prod)

    class_a_index = sales_by_prod.query('(sales_perc >= 0.695) & (sales_perc <= 0.71)').index[0]+1
    class_b_index = sales_by_prod.query('(sales_perc >= 0.95) & (sales_perc <= 0.96)').head(1).index[0]+1


    class_a_items = sales_by_prod[:class_a_index].item[:11]

    return class_a_items

# Central Limit Theorem Sampling
def sampling(data, n_samples, sample_size, attribute='sales', seed=12):

    """This function receives a data set and returns `n_samples` with `sample_size` elements each for a given `attribute` and the means of each sample."""

    assert isinstance(data, pd.DataFrame), "`data` Must be a PandasDataFrame."

    np.random.seed(seed)
    samples = []

    for i in range(n_samples):

        samples.append(data.sample(sample_size)[attribute])

    samples_means = np.mean(samples, axis=1)

    return samples, samples_means

# Preparing Data
def central_limit(data, product, target, date_col, return_data_frame=False):

    data = data.loc[data[date_col].dt.year >= 2016]

    df_sales = data.query(f'item == {product}').sort_values(date_col)
    df_daily_sales = df_sales.groupby(date_col).sum()[[target]]


    samples, samples_means = sampling(data=df_daily_sales, n_samples=1500, sample_size=30, attribute='sales')


    if return_data_frame == False:

        return samples, samples_means
    
    else:
        return df_daily_sales

# Question 1
def probability_sale_at_least(target, mean, st_dev):

    assert isinstance(target, float) or isinstance(target, int), '`target` must be a float or int.'
    assert isinstance(mean, float) or isinstance(mean, int), '`mean` must be a float.'
    assert isinstance(st_dev, float) or isinstance(st_dev, int), '`st_dev` must be a float.'

    z_score = (target - mean)/st_dev
    prob = 1 - sp.norm.cdf(z_score)
    return prob


# Question 2
def stocking_out_value(mean, st_dev, percent=0.8):

    value = sp.norm.ppf(1-percent, loc=mean, scale=st_dev)

    #print(f'')
    return round(value)


#  Question 3
def interval_perc(mean, st_dev, percent=0.9):

    upper = 1 - (1 - percent)/2
    lower = (1 - percent)/2
    upper_limit = sp.norm.ppf(upper, loc=mean, scale=st_dev)
    lower_limit = sp.norm.ppf(lower, loc=mean, scale=st_dev)

    st.write(f'Range with `{percent:.0%}` of possible demand: **[{round(lower_limit)} —— {round(upper_limit)}] units**')

    #st.write(f'Lower Limit  ——— Upper Limit')
    #st.write(f'[{round(lower_limit)}    ——————————    {round(upper_limit)}]')

    return lower_limit, upper_limit


##### Plots ######
def hist_means(samples, product, bins=50):

    mean = np.mean(samples)
    stand = np.std(samples)

    #plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    sns.histplot(samples, kde=True, bins=50, ax=ax, color='teal')

    ax.text(x=mean-9, y=81, s=fr'$\mu$={mean:.2f}', fontdict={'fontsize': 25, 'fontname': 'sans-serif', 'color': 'lime'})
    ax.vlines(x=mean, ymin=0, ymax=81, linestyle='--', colors='red')
   

    ax.text(x=mean+stand/2-15, y=42, s=r'$\sigma$/$\sqrt{n}$=' + f'{stand:.2f}', fontdict={'fontsize': 25, 'fontname': 'sans-serif', 'fontstyle': 'italic', 'color': 'lime'})
    ax.arrow(x=mean, y=41, dx=stand, dy=0, color='red', head_length=1,
              width=0.5)
    
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticklabels('')
    ax.set_yticks([])
    ax.xaxis.label.set_color('white')
    ax.grid(False)
    
    #ax.set_title(f'CLT - Item {product} - Distribution of means');
    return fig

def image_header(image_path):

    with st.container():

        image = Image.open(image_path)
        new_image = image.resize((800, 200))
        st.image(image=new_image, width=800)

if __name__ == '__main__':

    st.set_page_config(page_title='Products Demand Analysis', page_icon=":chart_with_upward_trend:")

    plots_configs()

    header_image = 'images/header_demand.jpg'
    image_header(image_path=header_image)
    
    

    path = '../DATA/train.csv'
    data = load_data(path)

    top_10_prodcuts = top_11_items(data)


    st.markdown('# Demand Report')

    st.write('This page shows answers to the CEO\'s questions about the demand of the top 10 most sold products.')

    #st.write(data.head(6))

    st.sidebar.header('Demand')

    product = st.sidebar.selectbox(options=top_10_prodcuts, label='Products')
    #st.write(product)

    # Filters
    target_sale = st.sidebar.slider(label='Target Sales', min_value=800, max_value=1050, value=1050)
    stock = st.sidebar.slider(label='Target Demand', min_value=800, max_value=1050, value=1050)
    range_percent = st.sidebar.slider(label='Interval Percentage', min_value=0.05, max_value=0.95, step=0.05, value=0.90)
    prob_stocking_out = st.sidebar.slider(label='Probability of Stocking out', min_value=0.00, max_value=1.0, step=0.01, value=0.20)


    # CLT
    samples, samples_means = central_limit(data=data, product=product, target='sales', date_col='date', return_data_frame=False)

    dat = central_limit(data=data, product=product, target='sales', date_col='date', return_data_frame=True)

    avg = np.mean(samples_means)
    stand = np.std(samples_means)

    # Produc Stats
    #st.write(avg, stand)

    
    st.markdown(f"### Item {product}")
    st.markdown("<h2 style='text-align: center'> Distribution of Means - CLT </h2>", unsafe_allow_html=True)

    st.pyplot(hist_means(samples_means, product=product, bins=10))

    st.markdown("## CEO Questions")
    st.markdown(f"### 1. What is the probability that I will sell `{target_sale}` units a day?")
    
    prob = probability_sale_at_least(target_sale, mean=avg, st_dev=stand)

    st.write(f'The probability of selling `{target_sale}` units or more is **{prob:.2%}**.')
   

    st.markdown(f"### 2.Given the demand, what is the probability of stock-out if I arrange to have `{stock}` units in stock every day?")

    proba = probability_sale_at_least(stock, mean=avg, st_dev=stand)

    st.write(f'The probability of stocking out if you arrange to have `{stock}` units every day is **{proba:.2%}**.')

    st.markdown(f"### 3. What is the range which contains `{range_percent:.0%} `of my possible demand?")

    lw, up = interval_perc(mean=avg, st_dev=stand, percent=range_percent)

    st.markdown(f"### 4. How much stock should I have if I want `{prob_stocking_out:.0%}` probability of stocking out?")

    result = stocking_out_value(mean=avg, st_dev=stand, percent=1-prob_stocking_out)

    st.write(f'You should have **{result}** units in your stock if you want a `{prob_stocking_out:.0%}` probability of stocking out.')


    

 
