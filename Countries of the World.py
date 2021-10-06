from numpy.ma.core import count
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Data import and overview
data = pd.read_csv('countries of the world.csv', decimal = ".")
print(data.shape)
print(data.head())
print(data.dtypes)

# Removes limit on how many rows are displayed
pd.set_option("display.max_rows", None)

# Displays the number of NULL values in (a) each column and (b) each row
print(data.isnull().sum(axis = 0))
print(data.isnull().sum(axis = 1))

# Deletes rows where all values are NULL
data = data.dropna(how = "all")

# Returns rows with population greater than 1,000,000,000
print(data[data["Population"] > 1000000000])

# Prints the climate column, then deletes it from the dataframe
print(data["Climate"])
data = data.drop(["Climate"], axis = 1)

# Converts the datatypes of Arable, Crops, and Other to floats to prepare for calculation
data["Arable (%)"] = data["Arable (%)"].astype('float64')
data["Crops (%)"] = data["Crops (%)"].astype('float64')
data["Other (%)"] = data["Other (%)"].astype('float64')

# Creates new column 'Land Total' which is the sum of the 3 land types to check they sum to 100%, showing any rows where they do not sum to 100%
data["Land Total (%)"] = data["Arable (%)"] + data["Crops (%)"] + data["Other (%)"]
print(data.head())
print(data[["Arable (%)", "Crops (%)", "Other (%)", "Land Total (%)"]])
print(data.loc[data["Land Total (%)"] != 100])

# Creates new column showing each regions total population
data = data.assign(**{'Region Population' : data.groupby('Region').Population.transform('sum')})

# Scatter plot showing the correlation beween literacy rate and GDP per capita
plt.scatter(x = data["GDP ($ per capita)"], y = data["Literacy (%)"])
plt.show()

# Linear regression plot showing the correlation between population and land area
sns.regplot(x = "Population", y = "Area (sq. mi.)", data = data, scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.show()

# Correlation matrix and corresponding heatmap between every numeric column
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix)
plt.show()
