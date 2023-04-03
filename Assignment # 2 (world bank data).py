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
    """ 
    This function scatter_plot creates a scatter plot of the mean 
    population growth rate (annual %) and the total CO2 emissions per capita 
    (metric tons) for all countries in the input dataframe.
    """
      
    # Select data for population growth and CO2 emissions per capita
    df_pop_growth = df_years.loc[:, (df_years.columns.levels[0].tolist(), \
                                     'Population growth (annual %)')]
    df_co2_per_capita = df_years.loc[:, (df_years.columns.levels[0].tolist()\
                                 , 'CO2 emissions (metric tons per capita)')]

    # Drop indicator column header from the dataframes
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
    ax.set_title('Population Growth vs. CO2 Emissions per Capita', fontsize=20)
    ax.set_xlabel('Population Growth (Annual %)', fontsize=16)
    ax.set_ylabel('CO2 Emissions per Capita (Metric Tons)', fontsize=16)

    # Show the grid lines
    ax.grid(True)
    
    # Save figure
    plt.savefig('Scatter_Plot.png', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

def plot_urban_electricity(countries, colors):
    """
    Plots the urban population percentage and access to electricity for 
    selected countries over time.

    Parameters:
    -----------
    countries: list
        A list of country names to plot.
    colors: dict
        A dictionary of colors for each country.

    Returns:
    --------
    None
    """

    # select data for urban population percentage and access to electricity 
    # for selected countries
    df_urban = df_years.loc[:, (countries, \
                                'Urban population (% of total population)')]
    df_electricity = df_years.loc[1990:, (countries, \
                                'Access to electricity (% of population)')]

    # Drop indicator column header from the dataframes
    df_urban.columns = df_urban.columns.droplevel(1)
    df_electricity.columns = df_electricity.columns.droplevel(1)

    # create a new dataframe with data for each country
    df = pd.concat([df_urban, df_electricity], axis=1, \
                   keys=['Urban Population', 'Access to Electricity'])

    # plot line graph for each country in the first subplot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    ax1, ax2 = axs
    
    # plot line graph for each country in the first subplot
    for country in countries:
        df.loc[:, ('Urban Population', country)].\
            plot(ax=ax1, label=('Urban Population, ' + country), \
                 color=colors[country], legend=True)
        
        # add country name annotation to lines
        x_pos = df.index[1] 
        y_urban = df.loc[df.index[1], ('Urban Population', country)]
        ax1.text(x_pos, y_urban, country, color=colors[country], fontsize=10)
    
    # set graph properties for the first subplot
    ax1.set_title('Urban Population Over Time', fontsize=20)
    ax1.set_xlabel('Year', fontsize=16)
    ax1.set_ylabel('Percentage %', fontsize=16)
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot line graph for each country in the second subplot
    for country in countries:
        df.loc[:, ('Access to Electricity', country)].\
            plot(ax=ax2, label=('Access to Electricity, ' + country), \
                 color=colors[country], legend=True)
        
#         # add country name annotation to lines
#         x_pos = df.index[1] 
#         y_electricity = df.loc[df.index[1], ('Access to Electricity', country)]
#         ax2.text(x_pos, y_electricity, country, color=colors[country], fontsize=10)
    
    # set graph properties for the second subplot
    ax2.set_title('Access to Electricity Over Time', fontsize=20)
    ax2.set_xlabel('Year', fontsize=16)
    ax2.set_ylabel('Percentage %', fontsize=16)
    ax2.grid(True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust teh vertical space
    plt.subplots_adjust(hspace=0.5)
    
    # Save figure
    plt.savefig('Urban_pop_vs_acc_to_elec.png', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

def plot_forest_energy(countries, df_years):
    """
    This function plots a line graph for forest area percentage and renewable 
    energy consumption for the selected countries over the years.

    Arguments:

    countries: a list of country names (strings) to plot data for
    df_years: a pandas dataframe containing data for each year and indicator
    
    Returns:

    None
    """
    
    # select data for forest area percentage and renewable energy consumption 
    # for selected countries
    df_forest = df_years.loc[:, (countries, 'Forest area (% of land area)')]
    df_renewable = df_years.loc[:, (countries, \
        'Renewable energy consumption (% of total final energy consumption)')]

    # Drop indicator column header from the dataframes
    df_forest.columns = df_forest.columns.droplevel(1)
    df_renewable.columns = df_renewable.columns.droplevel(1)

    # create a new dataframe with data for each country
    df = pd.concat([df_forest, df_renewable], axis=1, \
                   keys=['Forest Area', 'Renewable Energy Consumption'])

    # plot line graph for each country
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'green', 'orange', 'purple']
    for i, country in enumerate(countries):
        df.loc[:, ('Forest Area', country)].\
            plot(ax=ax, label=('Forest Area, ' + country), color=colors[i], \
                 legend=True)
        df.loc[:, ('Renewable Energy Consumption', country)].\
            plot(ax=ax, label=('Renewable Energy, ' + country), \
                 color=colors[i], legend=True)
        
        # annotate lines with labels
        ax.annotate(('Forest Area, ' + country), \
                    xy=(1995, df.loc[1995, ('Forest Area', country)]), \
                        xytext=(10, 5), color=colors[i], \
                            textcoords='offset points')
        ax.annotate(('Renewable Energy, ' + country), \
                    xy=(2011, df.loc[2011, ('Renewable Energy Consumption', \
                                             country)]), xytext=(10, -3), \
                        color=colors[i], textcoords='offset points')

    # set graph properties
    ax.set_title('Forest Area Percentage and Renewable Energy Consumption', \
                 fontsize=20)
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel('Percentage', fontsize=16)
    ax.grid(True)
    
    # move legend to the right corner
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Save figure
    plt.savefig('Forest_area_vs_renewable_energy.png', bbox_inches='tight', \
                dpi=300)
    
    # show the plot
    plt.show()

def plot_decade_comparison(df, countries, decades):
    """
    Plots a bar chart subplot for each decade, showing the mean mortality 
    rate and mean agriculture value added percentage for each country in 
    the given list of countries.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be plotted.
    countries (list): List of countries to include in the plot.
    decades (list): List of decades to include in the plot.

    Returns:
    None
    """
    # select data for mortality rate under-5 and agriculture value added 
    # percentage for selected countries
    df_mortality = df.loc[:, (countries, \
                        'Mortality rate, under-5 (per 1,000 live births)')]
    df_agriculture = df.loc[:, (countries, \
                'Agriculture, forestry, and fishing, value added (% of GDP)')]

    # Drop indicator column header from the dataframes
    df_mortality.columns = df_mortality.columns.droplevel(1)
    df_agriculture.columns = df_agriculture.columns.droplevel(1)

    # create a new dataframe with data for each country
    df_combined = pd.concat([df_mortality, df_agriculture], axis=1, \
                            keys=['Mortality Rate', 'Agriculture Value Added'])

    # calculate mean values for each decade
    df_mean = pd.DataFrame()
    for decade in decades:
        start_year = int(decade.split('-')[0])
        end_year = int(decade.split('-')[1])
        df_decade = df_combined.loc[str(start_year):str(end_year), :]
        df_mean[decade] = df_decade.mean()

    # create bar subplots for each decade
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    axes = axes.flatten()
    for i, decade in enumerate(decades):
        ax = axes[i]
        x = np.arange(len(countries))
        width = 0.35
        ax.bar(x - width/2, df_mean.loc['Mortality Rate', decade], \
               width, label='Mortality Rate')
        ax.bar(x + width/2, df_mean.loc['Agriculture Value Added', decade], \
               width, label='Agriculture Value Added')
        
        # set titles and labels
        ax.set_title(decade)
        ax.set_xlabel('Country', fontsize=10)
        ax.set_ylabel('Percentage %', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=0)
        ax.legend()

    # set overall title and adjust spacing
    fig.suptitle('Mortality Rate and Agriculture Value Added by Decade', \
                 fontsize=20)
    fig.subplots_adjust(hspace=0.3)

    # Save figure
    plt.savefig('mor_vs_agr_vlue.png', bbox_inches='tight', dpi=300)
    
    # show the plot
    plt.show()

def plot_indicators(countries, indicator1, indicator2):
    """
    Creates a bar plot comparing two indicators for selected countries from 
    2017 to 2021.
    
    Parameters:
    -----------
    countries: list of str
        List of country names to be included in the plot
    indicator1: str
        Name of the first indicator
    indicator2: str
        Name of the second indicator
    
    Returns:
    --------
    None
    """
    
    # select data for the given indicators and countries for the years 
    # 2017-2021
    df_indicator1 = df_years.loc['2017':'2021', (countries, indicator1)]
    df_indicator2 = df_years.loc['2017':'2021', (countries, indicator2)]

    # Drop indicator column header from the dataframes
    df_indicator1.columns = df_indicator1.columns.droplevel(1)
    df_indicator2.columns = df_indicator2.columns.droplevel(1)

    # take mean of the data for each country from 2017 to 2021
    df_indicator1_mean = df_indicator1.mean()
    df_indicator2_mean = df_indicator2.mean()

    # create a new dataframe with mean values for each country
    df = pd.concat([df_indicator1_mean, df_indicator2_mean], axis=1, \
                   keys=[indicator1, indicator2])

    # create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(countries))
    width = 0.35
    ax.bar(x - width/2, df[indicator1], width, label=indicator1)
    ax.bar(x + width/2, df[indicator2], width, label=indicator2)
    
    # set the title and labels
    ax.set_title("Terrestrial and Marine protected areas vs Urban population \
by Country (2017-2021)", fontsize=20)
    ax.set_xlabel('Country', fontsize=16)
    ax.set_ylabel('Percentage %', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=90, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    
    # Save figure
    plt.savefig('terre_vs_urb_pop.png', bbox_inches='tight', dpi=300)
    
    # show the plot
    plt.show()

def protected_areas_vs_urban_population(countries, indicators):
    '''
    Creates a scatter plot of protected areas versus urban population for the 
    given countries and indicators.

    Parameters:
        - countries (list of str): list of country names to include in 
        the plot
        - indicators (list of str): list of indicator names to include in 
        the plot

    Returns:
        - None
    '''

    # Select the data for the given countries and indicators for the years 
    # 2017-2021
    df_selected = df_years.loc['2017':'2021', (countries, indicators)]
    df_selected = df_selected.mean()

    # Calculate the total population of each country
    df_pop = df_years.loc['2017':'2021', (countries, 'Urban population')]
    populations = df_pop.mean(axis=0)

    # Create a scatter plot with dot size proportional to population
    fig, ax = plt.subplots(figsize=(8,6))
    for country in countries:
        x = df_selected.loc[(country, indicators[0])]
        y = df_selected.loc[(country, indicators[1])]
        size = populations[country]/1000000 # divide by a factor for 
                                            # readability
        ax.scatter(x, y, s=size, alpha=0.5, label=country)

    # Set legend position to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Set x and y axis labels and plot title
    ax.set_xlabel(indicators[0], fontsize=16)
    ax.set_ylabel(indicators[1], fontsize=16)
    ax.set_title('Protected Areas by Urbanization', fontsize=20)
    
    # Save figure
    plt.savefig('scatter_terre_vs_urb_pop.png', bbox_inches='tight', dpi=300)
    
    # show the plot
    plt.show()

def calculate_mean_energy_emissions(df_years, countries):
    """
    This function creates a table comparing renewable energy and CO2 
    emissions for selected countries for past 3 decades and save it in excel.
    
    Returns:
    Formatted table with a caption
    """
    
    # select data for renewable energy consumption and CO2 emissions per 
    # capita for selected countries
    df_renewable = df_years.loc[:, (countries, \
        'Renewable energy consumption (% of total final energy consumption)')]
    df_emissions = df_years.loc[:, (countries, \
                                    'CO2 emissions (metric tons per capita)')]

    # Drop indicator column header from the dataframes
    df_renewable.columns = df_renewable.columns.droplevel(1)
    df_emissions.columns = df_emissions.columns.droplevel(1)

    # create a new dataframe with data for each country
    df_combined = pd.concat([df_renewable, df_emissions], axis=1, \
                            keys=['Renewable Energy Consumption', \
                                  'CO2 Emissions per Capita'])
    
    # decades list
    decades = ['1990-1999', '2000-2009', '2010-2019']

    # calculate mean values for each decade
    df_mean = pd.DataFrame()
    for decade in decades:
        start_year = int(decade.split('-')[0])
        end_year = int(decade.split('-')[1])
        df_decade = df_combined.loc[str(start_year):str(end_year), :]
        df_mean[decade] = df_decade.mean()
    
    # reset index and set new index names for df_mean
    df_mean = df_mean.reset_index().set_index(['Country Name', 'level_0'])\
        .sort_index()
    df_mean.index.names = ['Country', 'Indicators']
    
    # save df_mean to Excel file
    df_mean.to_excel('df_mean.xlsx')
    
    # display formatted dataframe with a caption
    return display(df_mean.style.set_caption('Renewable Energy Consumption \
and CO2 Emissions per Capita by Country and Decade'))

""" Main Program """

# Calling the function to read the world bank data file
df_years, df_countries = read_world_health_data\
    ('Data/API_19_DS2_en_excel_v2_4903056.xls')

# print the dataframes extracted from the original file
print(df_countries.head(), "\n")
print(df_years.head(), '\n')

# List of countries to consider in this analysis
countries = ['China', 'United States', 'Russian Federation', 'Japan', \
             'Germany', 'United Kingdom', 'France', 'Italy', 'Brazil', \
                 'Canada', 'Korea, Rep.', 'Australia', 'Spain', 'Mexico', \
                     'Indonesia']

# List of indicators to consider in this analysis
indicators = ['Urban population (% of total population)', 
               'Population, total', 
               'Population growth (annual %)', 
               'Agriculture, forestry, and fishing, value added (% of GDP)', 
               'Mortality rate, under-5 (per 1,000 live births)', 
               'Terrestrial and marine protected areas \
                   (% of total territorial area)', 
               'Renewable energy consumption \
                   (% of total final energy consumption)', 
               'Access to electricity (% of population)', 
               'Forest area (% of land area)', 
               'CO2 emissions (metric tons per capita)']

# Calling the world_health_stats function to extract the stats
summary, corr_matrix, cov_matrix, spearman_matrix, \
    kendall_matrix, pearson_matrix = world_health_stats(countries, indicators)

# printing the stats return by world_health_stats
print("Summary Statistics:")
print(summary)
print("\nCorrelation Matrix:")
print(corr_matrix)
print("\nCovariance Matrix:")
print(cov_matrix)
print("\nSpearman matrix:")
print(spearman_matrix)
print("\nKendall matrix:")
print(kendall_matrix)
print("\nPearson matrix:")
print(pearson_matrix)

# Calling the plot_heatmap function to display and save the heatmap
# provided corr_matrix from world_bank_stats
plot_heatmap(corr_matrix, title='Correlation Heatmap of Indicators')

# Calling the generate_scatter_matrix function to display and save the scatter
# matrix for three countries.
generate_scatter_matrix(df_years, \
                       ['China', 'United States', 'Russian Federation'], \
                            indicators)

# Calling the scatter_plot to display and save the scatter plot of population 
# growth vs CO2 emissions per capita of the whole countries in the dataframe.
scatter_plot(df_years)

# Countries list to be used in plot_urban_electricity function
countries_urb_elec = ['China', 'Brazil', 'Canada', 'Korea, Rep.', \
                      'Australia', 'Spain', 'Mexico', 'Indonesia']

# define colors for each country
colors = {'China': 'blue', 'Brazil': 'green', 'Canada': 'orange', \
          'Korea, Rep.': 'red', 'Australia': 'purple', 'Spain': 'brown', \
              'Mexico': 'pink', 'Indonesia': 'gray'}
    
# Calling the function to plot and save urban pop and access to electricity 
# over time
plot_urban_electricity(countries_urb_elec, colors)

# Calling the function to plot and save forest area and renewable energy 
# consumption over time for selected countries
plot_forest_energy(['China', 'Brazil', 'Mexico', 'Indonesia'], df_years)

# Calling the function to plot and save a bar plot of mortality rate and  
# agricultural value added vs countries for the 4 decades
plot_decade_comparison(df_years, ['China', 'Brazil', 'Mexico', 'Indonesia'], \
                       ['1990-1999', '2000-2009', '2010-2019', '2020-2021'])

# Calling the function to plot and save terrestrial and marine protected areas 
# vs urban population by countries
plot_indicators(countries, 'Terrestrial and marine protected areas \
(% of total territorial area)', 'Urban population (% of total population)')

# Calling the function to scatter plot and save protected areas by urbanization
protected_areas_vs_urban_population(countries, ['Terrestrial protected areas \
(% of total land area)', 'Marine protected areas (% of territorial waters)'])

# Calling the function to make a table for CO2 emissions and renewable energy
# for the past 3 decades and save it in excel file
calculate_mean_energy_emissions(df_years, countries)








