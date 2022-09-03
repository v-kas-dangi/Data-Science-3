"""
DS3 Lab 1
Name: Vikas Dangi
Roll Number: B20238
Mobile Number: 9406661661
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading the csv file provided to us
df=pd.read_csv('pima-indians-diabetes.csv')

#Part 1

#Getting the series containing the median of all the columns except the last one
medianSeries=pd.Series(df.median().iloc[0:8],name="median")
#Getting the DataFrame containing the mode of all the columns
modedf=df.mode(axis=0,numeric_only=False)
#dropping the class column
modedf.drop(columns="class",inplace=True)
#getting the static details including mean std min and max of the required columns
details=df.loc[:,"pregs":"Age"].describe()
details=details.loc[["mean","std","min","max"]]
#appending the median and mode to it
details=details.append(medianSeries)
details=details.append(modedf)
details=details.rename(index={0:"Mode 1",1:"Mode 2"})
print(details)


# %%
#Part 2
#A

#1
#Plotting scatter plot using matplot between Age and number of pregnant women in that range
plt.scatter(df['Age'],df['pregs'],c='r',marker=".")
#giving title to the plot
plt.title("Scatter plot: Age (in years) vs. pregs")
#Labelling the X and Y axis
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Number of times pregnant",fontsize=10)
#This enables the gird in the plot
plt.grid(True)
#manages the layout and orientation
plt.tight_layout()
#displays the graph with the above attributes
plt.show()

#2
#Plotting scatter plot using matplot between Age and Plasma glucose concentration of women in that range
plt.scatter(df['Age'],df['plas'],c='b',marker=".")
plt.title("Scatter plot: Age (in years) vs. plas")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Plasma glucose concentration",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#3
#Plotting scatter plot using matplot between Age and Diastolic blood pressure of women in that range
plt.scatter(df['Age'],df['pres'],c='g',marker=".")
plt.title("Scatter plot: Age(in years) vs. pres(in mm Hg)")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Diastolic blood pressure(mm Hg)",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#4
#Plotting scatter plot using matplot between Age and Triceps skin fold thickness women in that range
plt.scatter(df['Age'],df['skin'],c='orange',marker=".")
plt.title("Scatter plot: Age(in years) vs. skin(in mm)")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Triceps skin fold thickness(mm)",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#5
#Plotting scatter plot using matplot between Age and 2-Hour serum insulin of women in that range
plt.scatter(df['Age'],df['test'],c='y',marker=".")
plt.title("Scatter plot: Age(in years) vs. test(in mm U/mL)")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("2-Hour serum insulin",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#6
#Plotting scatter plot using matplot between Age and number of pregnant women in that range
plt.scatter(df['Age'],df['BMI'],c='purple',marker=".")
plt.title(" Scatter plot: Age(in years) vs. BMI(in kg/m2)")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Body mass index",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#7
#Plotting scatter plot using matplot between Age and Diabetes pedigree function of woomen in that range
plt.scatter(df['Age'],df['pedi'],c='k',marker=".")
plt.title("Scatter plot: Age (in years) vs. pedi")
plt.xlabel("Age(in years)",fontsize=10)
plt.ylabel("Diabetes pedigree function",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()


#B
#1
#Plotting scatter plot using matplot between BMI and number of pregnant women in that range
plt.scatter(df['BMI'],df['pregs'],c='c',marker=".")
plt.title("Scatter plot: BMI(in kg/m2) vs. pregs")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Number of times pregnant",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#2
#Plotting scatter plot using matplot between BMI and Plasma glucose concentration of women in that range
plt.scatter(df['BMI'],df['plas'],c='m',marker=".")
plt.title("Scatter plot: BMI(in kg/m2) vs. plas")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Plasma glucose concentration",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#3
#Plotting scatter plot using matplot between BMI and Diastolic blood pressure of women in that range
plt.scatter(df['BMI'],df['pres'],c='r',marker=".")
plt.title("Scatter plot: BMI(in kg/m2) vs. pres(in mm Hg)")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Diastolic blood pressure(mm Hg)",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#4
plt.scatter(df['BMI'],df['skin'],c='pink',marker=".")
plt.title("Scatter plot: BMI(in kg/m2) vs. skin(in mm)")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Triceps skin fold thickness(mm)",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#5
#Plotting scatter plot using matplot between BMI and 2-Hour serum insulin of women in that range
plt.scatter(df['BMI'],df['test'],c='brown',marker=".")
plt.title("Scatter plot: BMI(in kg/m2) vs. test(in mm U/mL)")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("2-Hour serum insulin",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#6
#Plotting scatter plot using matplot between BMI and Diabetes pedigree function of women in that range
plt.scatter(df['BMI'],df['pedi'],c='violet',marker=".",s=10)
plt.title("Scatter plot: BMI(in kg/m2) vs. pedi")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Diabetes pedigree function",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
#7
#Plotting scatter plot using matplot between BMI and Age of women in that range
plt.scatter(df['BMI'],df['Age'],c='blue',marker=".")
plt.title(" Scatter plot: BMI(in kg/m2) vs. Age")
plt.xlabel("Body mass index",fontsize=10)
plt.ylabel("Age(in years)",fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#Part 3
print("\nThe correlation coefficient between the Age and all other attributes (excluding ‘class’) are: \n")
#Calculating correlation coefficient between the Age and all other attributes
print(df.iloc[:,0:8].corrwith(df["Age"]))
print("\nThe correlation coefficient between the BMI index and all other attributes (excluding ‘class’) are: \n")
#Calculating correlation coefficient between the BMI Index and all other attributes
print(df.iloc[:,0:8].corrwith(df["BMI"]))


# %%
#Part 4
#A
#making bins to get the bin size manually
bin=np.arange(0,18)
plt.hist(df.pregs,bins=bin-0.5)
#labelling X axis
plt.xlabel('Number of times pregnant',fontsize=10)
#Labellig Y axis
plt.ylabel('Number of Females',fontsize=10)
#givng the title of the plot
plt.title("Histogram depiction of attribute pregs")
#for keeping a tight layout fixing the border distortion
plt.tight_layout()
#labelling X axis with the bin array
plt.xticks(bin)
#for taking grid in our plot
plt.grid(True)
plt.show()

#B
#Making  Histogram for depiction of attribute skin
plt.hist(df.skin,bins=50)
plt.xlabel('Triceps skin fold thickness(mm)',fontsize=10)
plt.ylabel('Frequency of occurance in Females',fontsize=10)
plt.title("Histogram depiction of attribute skin")
plt.tight_layout()
plt.grid(True)
plt.show()


# %%
#Part 5
#A
#for class 0
pregclass0=df[(df["class"]==0)].pregs
bin=np.arange(0,18)
plt.hist(pregclass0,bins=bin-0.5)
plt.title("Depiction of attribute pregs for class 0")
#labelling X axis
plt.xlabel('Number of times pregnant',fontsize=10)
#labelling Y axis
plt.ylabel('Frequency of Females',fontsize=10)
#for keeping a tight layout fixing the border distortion
plt.tight_layout()
plt.xticks(bin)
plt.grid(True)
plt.show()
#B
#for class 1
pregclass1=df[(df["class"]==1)].pregs
bin=np.arange(0,18)
plt.hist(pregclass1,bins=bin-0.5)
plt.title("Depiction of attribute pregs for class 1")
plt.xlabel('Number of times pregnant',fontsize=10)
plt.ylabel('Frequency of Females',fontsize=10)
plt.tight_layout()
plt.xticks(bin)
plt.grid(True)
plt.show()


# %%
#Part 6
#plotting box plots of different attributes
#1
plt.boxplot(df['pregs'])
plt.ylabel("Number of times pregnant")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute pregs")
plt.show()
#2
plt.boxplot(df['plas'])
plt.ylabel("Plasma glucose concentration")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute plas")
plt.show()
#3
plt.boxplot(df['pres'])
plt.ylabel("Diastolic blood pressure(mm Hg)")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute pres")
plt.show()
#4
plt.boxplot(df['skin'])
plt.ylabel("Triceps skin fold thickness(mm)")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute skin")
plt.show()
#5
plt.boxplot(df['test'])
plt.ylabel("2-Hour serum insulin(mu U/mL) ")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute test")
plt.show()
#6
plt.boxplot(df['BMI'])
plt.ylabel("Body mass index")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute BMI")
plt.show()
#7
plt.boxplot(df['pedi'])
plt.ylabel("Diabetes pedigree function")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute pedi")
plt.show()
#8
plt.boxplot(df['Age'])
plt.ylabel("Age(years)")
plt.xlabel("Box Plot")
plt.title("Boxplot for attribute Age")
plt.show()


# %%
