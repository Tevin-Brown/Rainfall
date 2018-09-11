'''
Created on Jul 26, 2018

@author: tevin
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  linear_model, metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL.ImageFilter import numpy
from scipy.stats import ttest_ind

def view(fname):
    rainfall = pd.read_csv(fname)
    print(rainfall.shape)
    print(rainfall.head(5))
    print(rainfall.describe())
    print()
    
def graphs(fname):
    rainfall = pd.read_csv(fname)
    cat1 = rainfall["Plot 1"]
    cat2 = rainfall["Plot 2"]
    cat3 = rainfall["Actual"]
    time = rainfall["Time"]
    ax = plt.subplot(111)
    ax.bar(time+.25, cat1, width = .25, color = 'blue',  label = "New Forest")
    ax.bar(time, cat2, width = .25, color = "green", label = "Secondary Forest")
    ax.bar(time-.25, cat3, width = .25, color = "red", label = "Open Field")
    plt.legend()
    plt.title("Average Through-all Histogram")
    plt.xlabel("Average (cm3)")
    plt.ylabel("Number")
    plt.show()
    
def t_Test(fname):
    rainfall = pd.read_csv(fname)
    cat1 = rainfall["Plot 1"]
    cat2 = rainfall["Plot 2"]
    plt.hist([cat1,cat2], color = ['blue','red'] , edgecolor = 'black', bins = 25, label = ["New Forest", "Secondary Forest"])
    plt.legend()
    plt.title("Through-all Histogram")
    plt.xlabel("Through-all (cm3)")
    plt.ylabel("Number")
    plt.show()
    print("Significance between averages of two plots")
    print("Average Through-all New Forest:",numpy.average(cat1))
    print("Average Through-all Old Forest:",numpy.average(cat2))
    print(ttest_ind(cat1, cat2))
    print()
    
def difference(fname):
    rainfall = pd.read_csv(fname)
    cat1 = numpy.array(rainfall["Actual"]) - numpy.array(rainfall["Plot 1"])
    cat2 = numpy.array(rainfall["Actual"]) - numpy.array(rainfall["Plot 2"])
    plt.hist([cat1,cat2], color = ['blue','red'] , edgecolor = 'black', bins = 25, label = ["New Forest", "Secondary Forest"])
    plt.legend()
    plt.title("Difference Between Rainfall and Through-all Histogram")
    plt.xlabel("Difference (cm3)")
    plt.ylabel("Number")
    plt.show()
    print("Significance of the difference between control and experiment")
    print("Average Difference New Forest:",numpy.average(cat1))
    print("Average Difference Old Forest:",numpy.average(cat2))
    print(ttest_ind(cat1, cat2))
    print()
    
def SLR(fname):
    rainfall = pd.read_csv(fname)
    X = rainfall.iloc[:,-1:].values #Open
    y1 = rainfall.iloc[:, 1].values #Newly Planted Forest
    y2 = rainfall.iloc[:,2].values #Secondary Forest
    
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=10)    
    regressor = linear_model.LinearRegression()  
    regressor.fit(X_train, y1_train)  
    intercept = regressor.intercept_ 
    coefficient = regressor.coef_[0]
    print("Equation for Newly Planted");
    print("Middle Temperature = (" + str(coefficient) + ")Ground Temperature + (" + str(intercept) + ")");
    print()
    y1_pred = regressor.predict(X_test)  
    print("Metrics")
    print('Correlation Coefficient(R2):', metrics.r2_score(y1_test, y1_pred)) 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
    print()
    x = rainfall["Actual"]
    line = [coefficient*i+intercept for i in x]
    rainfall.plot(x='Actual', y= ['Plot 1'], style='o', markersize = 3)
    plt.plot(x, line, 'r', label='y={:.2f}x+{:.2f}'.format(coefficient,intercept))  
    plt.legend()
    plt.title('Correlation between Open rainfall and Newly Planted Through-all')  
    plt.xlabel('Open Rainfall (cm3)')  
    plt.ylabel('Newly Planted Through-all (cm3)')  
    plt.show()
    
    X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=10)    
    regressor = linear_model.LinearRegression()  
    regressor.fit(X_train, y2_train)  
    intercept = regressor.intercept_ 
    coefficient = regressor.coef_[0]
    print("Equation for Secondary");
    print("Middle Temperature = (" + str(coefficient) + ")Ground Temperature + (" + str(intercept) + ")");
    print()
    y2_pred = regressor.predict(X_test)  
    print("Metrics")
    print('Correlation Coefficient(R2):', metrics.r2_score(y2_test, y2_pred)) 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y2_test, y2_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y2_test, y2_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2_test, y2_pred)))
    print()
    x = rainfall["Actual"]
    line = [coefficient*i+intercept for i in x]
    rainfall.plot(x='Actual', y= ['Plot 2'], style='o', markersize = 3)
    plt.plot(x, line, 'r', label='y={:.2f}x+{:.2f}'.format(coefficient,intercept))  
    plt.legend()
    plt.title('Correlation between Open rainfall and Secondary Through-all')  
    plt.xlabel('Open Rainfall (cm3)')  
    plt.ylabel('Secondary Through-all (cm3)')  
    plt.show()
    
if __name__ == '__main__':
    file1 = "Rainfall_Data.csv"
    file2 = "Rainfall_Only.csv"
    file3 = "Rainfall_Averages.csv"
#     graphs(file3)
#     view(file2)
#     t_Test(file2)
#     difference(file2)
#     SLR(file1)#don't perform well