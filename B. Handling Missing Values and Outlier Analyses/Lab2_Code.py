# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
"""
Vikas Dangi
B20238
DS3-Lab 2
"""
#Question 1
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#reading csv to dataframe format
df_original=pd.read_csv("landslide_data3_original.csv")
df_miss=pd.read_csv("landslide_data3_miss.csv")

#finding missing values
NoOfMiss=df_miss.isnull().sum()
attribute_names=df_miss.columns

#plooting bar graph misisng values
plt.bar(attribute_names,NoOfMiss)
plt.title("attribute names (x-axis) vs the number of missing values")
plt.xlabel("Attributes")
plt.ylabel("Number of missing values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
len(df_miss['temperature'])


# %%
#Question 2
#a
#creating dfnew from whcich tuples with missing stationid value are drpped
dfnew=df_miss.dropna(subset=['stationid'])
print("The total number of tuples deleted: ",df_miss['stationid'].size-dfnew['stationid'].size)
#b
#tuples with more than 3 missing attribute are drpped
l1=dfnew['stationid'].size
dfnew=dfnew.dropna(thresh=dfnew.shape[1]-2)
print("The total number of tuples deleted after new deletion: ",l1-dfnew['stationid'].size)

# fidning missing values  in each attribute
#Question 3
print("The total number of missing values in each attributes: ")
print(dfnew.isnull().sum())

#finding total misisng values
TotalMissNew=sum(dfnew.isnull().sum())
print("\nThe total number of missing values in the new Dataframe is: ", TotalMissNew)
dfnew0=dfnew


# %%
#Question 4
"""

"""
#a
#filling  misisng values wiht mean values
meanSeries=pd.Series(dfnew.mean(skipna=True,numeric_only=True),name='mean')
dfnew=dfnew.fillna(meanSeries)

# i)

# Compute the mean, median, mode and standard deviation for each attributes after filling values
medianSeries=pd.Series(dfnew.median(numeric_only=True),name="median")
stdSeries=pd.Series(dfnew.std(skipna=True,numeric_only=True),name='Std Deviation')
modeSeries=dfnew.mode(numeric_only=True)
modeSeries=modeSeries.rename(index={0:"Mode "})
statnew=modeSeries.append(medianSeries)
statnew=statnew.append(meanSeries)
statnew=statnew.append(stdSeries)
print("\nThe values of mean, median and mode for each attribute after filling NA with mean are: ")
print(statnew)


# Compute the mean, median, mode and standard deviation for each attributes for the original file
medianSeries=pd.Series(df_original.median(numeric_only=True),name="median")
stdSeries=pd.Series(df_original.std(skipna=True,numeric_only=True),name='Std Deviation')
modeSeries=df_original.mode(numeric_only=True)
modeSeries=modeSeries.rename(index={0:"Mode "})
statOriginal=modeSeries.append(medianSeries)
statOriginal=statOriginal.append(meanSeries)
statOriginal=statOriginal.append(stdSeries)
print("\nThe values of mean, median and mode for each attribute of original file are: ")
print(statOriginal)

# ii)
#defining fun to calculate rmse
def RMSE(df1,df2,cname):
    oldindex=list(df1.index)
    countmiss=0
    sumsq=0
    for i in oldindex:
        sumsq+=(df1[cname].loc[i]-df2[cname].loc[i])**2
        if((df1[cname].loc[i]-df2[cname].loc[i])!=0):
            countmiss+=1
    if(countmiss==0):
        return 0
    else:
        rmse=math.sqrt(sumsq/countmiss)
    return rmse
cnames=list(attribute_names)[2:10]

#creatign  list of rmse VALUES OF ALL ATTRIBUTES
RMSElist=[]
for i in cnames:
    RMSElist.append(RMSE(dfnew,df_original,i))
RMSEseries=pd.Series(RMSElist,cnames)

#plotting log scaled graph for RMSE value of all attributes
print("\n The plot between Attributes and their RMSE error is: ")
plt.bar(cnames,RMSElist)
plt.xlabel("Attributes")
plt.ylabel("RMSE error")
plt.yscale('log')
plt.xticks(rotation=45)
plt.show()
print("\nThe RMSE between the original and replaced values for each attribute are: ")
print(RMSEseries)



# b

# i)
#filling misisn values of database by interpolation
dfnew0=dfnew0.interpolate()

# Compute the mean, median, mode and standard deviation for each attributes after linear interpolating values
meanSeries=pd.Series(dfnew0.mean(skipna=True,numeric_only=True),name='mean')
stdSeries=pd.Series(dfnew0.std(skipna=True,numeric_only=True),name='Std Deviation')
medianSeries=pd.Series(dfnew0.median(numeric_only=True),name="median")
modeSeries=dfnew0.mode(numeric_only=True)
modeSeries=modeSeries.rename(index={0:"Mode "})
statnew0=modeSeries.append(medianSeries)
statnew0=statnew0.append(meanSeries)
statnew0=statnew0.append(stdSeries)

print("\nThe values of mean, median and mode for each attribute after linear interpolating are:\n ")
print(statnew0)
print("\nThe values of mean, median and mode for each attribute of original file are:\n ")
print(statOriginal)

# ii)
#creatign  list of rmse VALUES OF ALL ATTRIBUTES
RMSElist=[]
RMSElist0=[]
for i in cnames:
    print(RMSE(dfnew0,df_original,i))
    RMSElist0.append(RMSE(dfnew0,df_original,i))
RMSEseries0=pd.Series(RMSElist0,cnames)

print("\nThe RMSE between the original and interpolated values for each attribute are: ")
print(RMSEseries0)

print("\n The plot between Attributes and their RMSE error is: ")
plt.bar(cnames,RMSElist0)
plt.xlabel("Attributes")
plt.ylabel("RMSE error")
plt.yscale('log')
plt.xticks(rotation=45)
plt.show()


# %%
#Question 5
#function to plot boxplots
dfnew0 = dfnew0.reset_index().drop(["index"] , axis = 1)
def plot_boxplot(clm):
    plt.boxplot(dfnew0[clm])
    plt.xlabel("DATA")
    plt.ylabel("Values")
    plt.show()

q1 = dfnew0.quantile(0.25)
q3 = dfnew0.quantile(0.75)

iqr = q3 - q1

# For identifying outliers
def identify_outlier (clm):
    outlier = []
    for i in range(len(dfnew0)):
        if  ((q3[clm] + (1.5 * iqr[clm])) < dfnew0[clm][i] or (dfnew0[clm][i]<(q1[clm] - 1.5 * iqr[clm]))):
            outlier.append(dfnew0[clm][i])
    plot_boxplot(clm)
    return(outlier)

# For replacing outliers
def replacePlot (clm):
    median_value = dfnew0[clm].median()
    for i in range(len(dfnew0)):
        if  (q3[clm] + (1.5 * iqr[clm])) < dfnew0[clm][i] or  (dfnew0[clm][i]<(q1[clm] - 1.5 * iqr[clm]) ):
            dfnew0[clm] = dfnew0[clm].replace([dfnew0[clm][i]], median_value)
    plot_boxplot(clm)

#identifying and replacing the outlier values of temp column
clm = "temperature"
print("\nOutliers in temeperature\n" , identify_outlier(clm))
replacePlot(clm)

#identifying and replacing the outlier values of rain column
clm = "rain" 
print("\nOutliers in Rain\n" ,identify_outlier(clm))
replacePlot(clm)

