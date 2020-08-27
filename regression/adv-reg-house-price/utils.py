import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from datetime import datetime
from scipy import stats
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")

class handle_data():
    
    def __init__(self,data):
        """
        Utility functions for Housing Price Prediction Data Handling 
        """
        self.data = data
        self.continuous = []
        self.discrete = []
        self.numerical = []
        self.years_vars = []
            
    def extract_var(self):
        '''
        Function to extract types of variables.
        Input : Dataframe 
        Output : Column list of Numerical, Categorical, Continuous, Discrete, Date-time    
        '''
        self.numerical = [var for var in self.data.columns if self.data[var].dtype != 'O']
        self.categorical = [var for var in self.data.columns if self.data[var].dtype == 'O']
        self.year_vars = [var for var in self.numerical if 'Yr' in var or 'Year' in var]
        
        self.discrete = []
        for var in self.numerical:
            if len(self.data[var].unique()) < 20 and var not in self.year_vars:
                self.discrete.append(var)
        
        self.continuous = [var for var in self.numerical if var not in self.discrete and \
                      var not in ['Id', 'SalePrice'] and var not in self.year_vars]

        return self.numerical, self.categorical, self.continuous, self.discrete, self.year_vars

    def plot_missing(self,png=False):
        """
        Function to plot Missing data in the dataframe
        Input: dataframe
        Output: Bar-Plot
        """
        missing_data = self.data.isnull().sum()
        missing_df = pd.DataFrame(missing_data.drop(missing_data[missing_data == 0].index).sort_values(ascending=False),
                                 columns = ['values'])
        fig = px.bar(data_frame = missing_df, x = missing_df.index, y = missing_df.values, text =missing_df.values,
                  title = "Missing values count")
        if png:
            fig.show('png')
        else:
            fig.show()


    def check_for_missing(self,cat_cont):
        '''
        Function to check number of missing values in dataframe with chosen type of variable
        Input : dataframe, column-list 
        Output : Column null count values
        '''     
        for x in self.data[cat_cont].columns:
            if self.data[x].isnull().sum()>0:
                print("{} : {}/{}".format(x, self.data[x].isnull().sum(), len(self.data[cat_cont]) ))
        print("\n Done Checking !") 


    def Imputation(self, var ,stats=False):
        '''
        Function to Impute data using sklearn Imputer
        Input : Dataframe, variable(continuous/categorical)
        Output : None
        '''
        if var==self.continuous:
            imputer= SimpleImputer(strategy='median')
        if var==self.categorical:
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            print("Imputation over continuous and categorical only")
        
        imputer.fit(self.data[var])
        self.data[var] = imputer.transform(self.data[var])
        print("Imputer Fitted and Transformed!")
        
        if stats==True:
            imputer.statistics_
    
    def find_non_rare_labels(self, variable, tolerance):
        '''
        Function to check cardinality of a feature.
        Args: Dataframe, 
                Feature - Numerical Features,
                Tolerance - Threshold of number of values.
        Output: List of unique values of the feature.
        '''
        temp = self.data.groupby([variable])[variable].count() / len(self.data)
        non_rare = [x for x in temp.loc[temp>tolerance].index.values]
        return non_rare    
    
    def rare_encoding(self, variable, tolerance):
        '''
        Encoding the rare labels to decrease cardinality

        Input: Dataframe, variable, 
                tolerance - Threshold value

        Ouput : dataframe with encoded values
        '''
        self.data = self.data.copy()
        # find the most frequent category
        frequent_cat = self.find_non_rare_labels(variable, tolerance)
        # re-group rare labels
        self.data[variable] = np.where(self.data[variable].isin(frequent_cat), self.data[variable], 'Rare')
        return self.data

    def diagnostic_plots(self, variable):
        '''
        Plotting Histogram, Quartile Plot(Q-Q) and Boxplot 
        Input : Dataframe, feature
        Output : Plot
        '''
        # define figure size
        plt.figure(figsize=(16, 4))

        # histogram
        plt.subplot(1, 3, 1)
        sns.distplot(self.data[variable], bins=30)
        plt.title('Histogram')

        # Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(self.data[variable], dist="norm", plot=plt)
        plt.ylabel('Variable quantiles')

        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=self.data[variable])
        plt.title('Boxplot')

        plt.show()    

        
    def fixing_skewness(self):
        """
        Function takes in a dataframe and return fixed skewed dataframe
        """
        ## Getting all the data that are not of "object" type. 
        numeric = self.data.dtypes[self.data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = self.data[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
        high_skew = skewed_feats[abs(skewed_feats) > 0.5]
        skewed_features = high_skew.index

        for feat in skewed_features:
            self.data[feat] = boxcox1p(self.data[feat], boxcox_normmax(self.data[feat] + 1))
