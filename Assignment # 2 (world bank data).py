# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 05:36:09 2023

@author: HAMZA
"""

# importing packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

""" Defining the functions to be used in a program """

def read_world_health_data(filename):
    """ Define a function that reads in a world health data Excel file and 
        returns two dataframes
    """
    
    # Read the Excel file into a Pandas dataframe, starting from the 4th row
    df = pd.read_excel(filename, header=3)
    
    # Create a copy of the original dataframe
    df_countries = df.copy()

    # Set the index of the dataframe to be a multiindex with 'Country Name' as 
    # the first level and 'Indicator Name' as the second level
    df_countries.set_index(['Country Name', 'Indicator Name'], inplace=True)
    
    # Drop the 'Country Code' and 'Indicator Code' columns
    df_countries.drop(['Country Code', 'Indicator Code'], axis=1, inplace=True)
    
    # Rename the column levels to 'Years'
    df_countries.columns.names = ['Years']

    # Convert the column headers to datetime format
    df_countries.columns = pd.to_datetime(df_countries.columns, format='%Y')
    
    # Extract the year component from the datetime column
    df_countries.columns = df_countries.columns.year
    
    # Drop rows and columns with all NaN values
    df_countries = df_countries.dropna(axis=0, how='all')
    df_countries = df_countries.dropna(axis=1, how='all')
    
    # Transpose the dataframe to get years as columns
    df_years = df_countries.transpose().copy()

    # Return both dataframes
    return df_years, df_countries

def world_health_stats(countries, indicators):
    """ This function, world_health_stats, takes in two parameters, countries 
        and indicators, and performs various statistical calculations on the 
        world health data.
    """

    # Subset the data for selected countries and indicators
    df_selected_years = df_years.loc[:, (countries, indicators)] 

    df_selected_countries = df_countries.loc[(countries, indicators), :]

    # Calculate summary statistics for each indicator
    summary = df_selected_years.describe()

    # Calculate correlation matrix
    corr_matrix = df_selected_years.corr()

    # Calculate covariance matrix
    cov_matrix = df_selected_years.cov()

    # Calculate Spearman's rank correlation coefficient matrix
    spearman_matrix = df_selected_years.corr(method='spearman')

    # Calculate Kendall's rank correlation coefficient matrix
    kendall_matrix = df_selected_years.corr(method='kendall')

    # Calculate Pearson's correlation coefficient matrix
    pearson_matrix = df_selected_years.corr(method='pearson')

    # Return the summary statistics, correlation matrix, covariance matrix,  
    # Spearman's rank correlation coefficient matrix, Kendall's rank 
    # correlation coefficient matrix, and Pearson's correlation coefficient 
    # matrix
    return summary, corr_matrix, cov_matrix, spearman_matrix, kendall_matrix,\
        pearson_matrix

def plot_heatmap(corr_matrix, title='Correlation Heatmap', figsize=(50,50), \
                 fontsize=16):
    """
    Generate a heatmap from a given correlation matrix using seaborn library

    Parameters:
    corr_matrix (pd.DataFrame): A correlation matrix of indicators
    title (str): Title for the heatmap (default is 'Correlation Heatmap')
    figsize (tuple): Figure size of the heatmap (default is (50, 50))
    fontsize (int): Font size of the labels (default is 16)

    Returns:
    None
    """

    # Set style and figure size
    sns.set(style='white')
    plt.figure(figsize=figsize)

    # Generate heatmap
    heatmap = sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, \
                          fmt='.2f', annot_kws={"size": fontsize-10})

    # Set x and y label font size and rotation
    plt.xticks(fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)

    # Set colorbar font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize+10)

    # Set title and save figure
    plt.title(title, fontsize=fontsize+30)
    plt.savefig('heatmap.png', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

def generate_scatter_matrix(df_years, countries, indicators, figsize=(15, 15)):
    """
    Generate a scatter plot matrix for the specified countries and indicators.
    
    Parameters:
    -----------
    df_years: pandas.DataFrame
        Dataframe containing data for multiple years.
        
    countries: list
        List of countries to include in the scatter plot matrix.
    
    indicators: list
        List of indicators to include in the scatter plot matrix.
    
    figsize: tuple, optional (default=(15, 15))
        Size of the figure to be plotted.
    """
    
    # Select data for the specified countries and indicators
    df_selected_years = df_years.loc[:, (countries, indicators)] 

    # Create scatter plot matrix
    scatter_matrix = pd.plotting.scatter_matrix(df_selected_years, \
                                                diagonal='hist', \
                                                    figsize=figsize)

    # Adjust layout and rotation of xlabels and ylabels
    for ax in scatter_matrix.ravel():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.xaxis.label.set_ha('right')
        ax.yaxis.label.set_ha('right')
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust spacing between subplots to prevent overlapping axis labels
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    # Set title and save figure
    fig = plt.gcf()
    fig.suptitle('Scatter Matrix of China, US and Russia', fontsize=26, y=0.95)
    plt.savefig('Scatter_Matrix.png', bbox_inches='tight', dpi=300)

    # Improve the layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def scatter_plot(df_years):
    """" This function scatter_plot creates a scatter plot of the mean 
        population growth rate (annual %) and the total CO2 emissions 
        per capita (metric tons) for all countries in the input dataframe.
    """"    
        
    # Select data for population growth and CO2 emissions per capita
    df_pop_growth = df_years.loc[:, (df_years.columns.levels[0].tolist(), \
                                     'Population growth (annual %)')]
    df_co2_per_capita = df_years.loc[:, (df_years.columns.levels[0].tolist()\
                                         , 'CO2 emissions (metric tons per \
                                             capita)')]

    # Drop indicator column headers from the dataframes
    df_co2_per_capita.columns = df_co2_per_capita.columns.droplevel(1)
    df_pop_growth.columns = df_pop_growth.columns.droplevel(1)

    # Only keep countries that are present in both dataframes
    df_pop_growth = df_pop_growth[df_co2_per_capita.columns.tolist()]

    # Create a new dataframe with x and y data
    df = pd.concat([df_pop_growth.mean(axis=0), \
                    df_co2_per_capita.sum(axis=0)], axis=1, \
                   keys=['Population Growth', 'CO2 Emissions per Capita'])

    # Create a scatter plot using Pandas' plot function
    ax = df.plot(kind='scatter', x='Population Growth', \
                 y='CO2 Emissions per Capita', figsize=(10, 8), \
                     logx=False, logy=True, s=20)

    # Set the title, x-label, and y-label of the plot
    ax.set_title('Population Growth vs. CO2 Emissions per Capita')
    ax.set_xlabel('Population Growth (Annual %)')
    ax.set_ylabel('CO2 Emissions per Capita (Metric Tons)')

    # Show the grid lines
    ax.grid(True)

    # Show the plot
    plt.show()






