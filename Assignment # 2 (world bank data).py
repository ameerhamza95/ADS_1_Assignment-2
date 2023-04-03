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







