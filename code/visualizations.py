"""
This module is for your final visualization code.
After you have done your EDA and wish to create some visualizations for you final jupyter notebook
A framework for each type of visualization is provided.
"""
# visualization packages
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Standard data manipulation packages
import pandas as pd
import numpy as np

matplotlib_axes_logger.setLevel('ERROR')

# Set specific parameters for the visualizations
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

def plot_tag_trend(tag):
    
    tag_slopes = {}

    q_df = pd.read_csv('./data/QA_all/Questions.csv', encoding='latin1')
    t_df = pd.read_csv('./data/QA_all/Tags.csv', encoding='latin1')
    
    q_df['time'] = pd.to_datetime(q_df['CreationDate'], infer_datetime_format=True)
    q_df['quarter'] = [x.quarter for x in q_df['time']]
    q_df['year'] = [x.year for x in q_df['time']]
    q_df['quarter'] = q_df['year'].apply(str).apply(lambda x: x[-2:]) + '-' + q_df['quarter'].apply(str)
    
    tag_per_id = t_df.groupby('Id').size()
    id_per_tag = t_df.groupby('Tag').size()

    tagid_df = pd.DataFrame()
    tagid_df['Id'] = tag_per_id.index.values
    tagid_df['tag_count'] = tag_per_id.values

    idtag_df = pd.DataFrame()
    idtag_df['tag'] = id_per_tag.index.values
    idtag_df['id_count'] = id_per_tag.values
    idtag_df = idtag_df.sort_values(by='id_count', ascending=True).reset_index().drop('index', axis=1)
    idtag_df = idtag_df.iloc[-256:]
    top_tags = idtag_df.tag.values

    all_quarters = list(q_df['quarter'].unique())
    for tag in top_tags:
        tag_slopes[tag] = {'quarter':['08-3'], 'pct':[], 'slope':[0]}

    i = 0

    for quarter in all_quarters:
        q_df2 = q_df[q_df['quarter']==quarter].copy()
        t_df2 = t_df[t_df['Id'].isin(q_df2['Id'].unique())].copy()
        for tag in top_tags:
            if i:
                tag_slopes[tag]['quarter'].append(quarter)
            this_pct = len(t_df2[t_df2['Tag']==tag])/len(q_df2)
            tag_slopes[tag]['pct'].append(this_pct)
            if i:
                prev_pct = tag_slopes[tag]['pct'][-2]
                tag_slopes[tag]['slope'].append(this_pct-prev_pct)
        i += 1

    fig, ax = plt.subplots(figsize=(15,8))
    data = pd.DataFrame(tag_slopes[tag])
    sns.lineplot(data=data, y='pct', x='quarter', marker='o')
    ax.set_title(f"Change in tag usage for tag: '{tag}'")
    plt.show()
    
    pass






def sample_plot_1():
    """
    This is a sample visualization function to show what one looks like.
    The code is borrowed from https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

    This function takes no arguments and shows a nice visualization without having all your code in the notebook itself.
    """

    # Set size of figure
    fig = plt.figure(figsize=(16, 10), dpi=80)  


    # Import dataset 
    midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

    # Prepare Data 
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

    # create ax element
    fig, ax = plt.subplots(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

    # Draw Plot for Each Category
    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal', 
                    data=midwest.loc[midwest.category==category, :], 
                    s=20, c=colors[i], label=str(category))

    # Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
                  xlabel='Area', ylabel='Population')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
    plt.legend(fontsize=12)
    plt.savefig('./images/viz1.png', transparent = True)
    
    plt.show()  
    
    pass

def sample_plot2():
    """
    This is a sample visualization function to show what one looks like.
    The code is borrowed from https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

    This function takes no arguments and shows a nice visualization without having all your code in the notebook itself.
    """

    plt.figure(figsize=(16, 10), dpi=80)
    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)

    # Decoration
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
    plt.gca().set(xlabel='Mileage per Gallon in the City', ylabel='Kernel Denisty')
    plt.legend()
    plt.savefig('./images/viz2.png', transparent = True)
    plt.show()

    pass


def visualization_one(target_var = None, input_vars= None, output_image_name=None):
    """
    The visualization functions are what is used to create each individual image.
    The function should be repeatable if not generalizable
    The function will call either the boxplot or density plot functions you wrote above

    :param target_var:
    :param input_vars:
    :param output_image_name: the desired name for the image saved
    :return: outputs a saved png file and returns a fig object for testing
    """
    ###
    # Main chunk of code here
    ###

    # Starter code for labeling the image
    plt.xlabel(None, figure = fig)
    plt.ylabel(None, figure = fig)
    plt.title(None, figure= fig)
    plt.legend()

    # exporting the image to the img folder
    plt.savefig(f'images/{output_image_name}.png', transparent = True, figure = fig)
    return fig