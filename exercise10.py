#import required modules
import pandas as pd
import numpy as np

#create new data frame from CSV for beer consumption and temperature
tempbeer = pd.read_csv("temp_and_beer_consump.csv")

#describe statistics for consumption and temperature
print(" Descriptive statistics for consumption score: ")
print(tempbeer["Consumption"].describe())
print(" Descriptive statistics for temperature: ")
print(tempbeer["Temp"].describe())

#import python library matplotlib
import matplotlib.pyplot as plt

#create a histogram for the consumption and temperature
plt.hist(x=tempbeer["Temp"], bins=20)

#label the consumption and temperature histogram
plt.xlabel("Temp")
plt.ylabel("Frequency")
plt.title("Frequency during Respective Temperatures")

#display the consumption and temperature histogram
print(plt.show())

#create a scatterplot with respective labels and display it to the user
plt.scatter(tempbeer["Temp"], tempbeer["Consumption"])
plt.xlabel("Temp")
plt.ylabel("Consumption")
plt.title("Consumption during Respective Temperatures")
print(plt.show())

#import the scipy model
from scipy.stats import pearsonr
import scipy
import statsmodels.api as sm

#create a correlation and display it
print(" Correlation coeifficient and p-value: ")
print(pearsonr(tempbeer["Temp"], tempbeer["Consumption"]))

#create two variables for linear regression
y = tempbeer["Consumption"]
x = tempbeer["Temp"]

#add the constant for linear regression
x = sm.add_constant(x)

#create the linear regression model
mod = sm.OLS(y, x)

#estimate the fit of the linear regression model
results = mod.fit()

#print the results
print(results.summary())

