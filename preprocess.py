import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from pandas.plotting import table

output = '' # global

def miss(df): # count num of missing values
    missing = df.isnull().sum()
    missing_percent = np.true_divide(missing, len(df))
    missing_dict = missing.to_dict()
    miss_dict_sort = dict(sorted(missing_dict.items(), key=lambda x: x[1], reverse=True))

    plt.bar(list(miss_dict_sort.keys())[:5], list(miss_dict_sort.values())[:5])
    plt.title("Top5 Columns")
    plt.xlabel("Column Names")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("miss.png")

    global output
    output+= """
            <h1 id="h1">Missing Values</h1>
            <h2>Descriptive Results</h2>  
            <p> On Average, {:.2f} Percent of Data is Missing </p>
            <p> {:d} cell(s) are Missing</p>
            <p> Total cell(s) {:d} </p>            
            <h2>Visualization</h2>  
            <img src="miss.png" alt="Missing Bar" class="img">         
            """.format(np.average(missing_percent), missing.sum(), df.size)  # the average is (sum of missing cell) / total cells

    if missing.sum() > 0:  # any missing values -> fill
        output+="""
                <h2>How Handled The Missing Values</h2>
                <table class="table_center"> 
                <tr>
                    <th>Types</th>
                    <th>Methods</th>
                </tr>                   
                """
        if len(df) < 5000 or np.average(missing_percent) > 10:
            num_df = df.select_dtypes(exclude=["bool_", "object_"])  # extract only numbers
            bool_df = df.select_dtypes(include=["bool_"])  # include only numbers
            obj_df = df.select_dtypes(exclude=["number", "bool_"])  # include only categorical

            print("Filling Missing Numerical Values by KNN-Imputation")
            if num_df.size > 0:
                scaler = MinMaxScaler()
                num_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)  # normalize the date [0-1] as KNN-Imputation calculates the distance
                imputer = KNNImputer(n_neighbors=5)
                df[num_df.columns] = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)  # replace by KNN
                output+="""
                        <tr>
                        <td>Numerical</td>
                        <td>KNN-Imputation</td>    
                        </tr>
                        """

            print("Filling Missing Bool and Categorical Values by Mode")
            sim_imputer = SimpleImputer(strategy='most_frequent')
            if bool_df.size > 0:
                df[bool_df.columns] = pd.DataFrame(sim_imputer.fit_transform(bool_df), columns=bool_df.columns)   # replace by Mode
                output+="""
                        <tr>
                        <td>Boolean</td>
                        <td>Most Frequent (Mode)</td>    
                        </tr>
                        """
            if obj_df.size > 0:
                df[obj_df.columns] = pd.DataFrame(sim_imputer.fit_transform(obj_df), columns=obj_df.columns)  # replace by Mode
                output+="""
                        <tr>
                        <td>Categorical</td>
                        <td>Most Frequent (Mode)</td>    
                        </tr>
                        """

        elif np.average(missing_percent) <= 10:
            print("Less Than 10% of Data is Missing: Remove Rows that contain Any Missing")
            df = df.dropna(inplace=True)
            output+="""
                    <tr>
                    <td>All Types</td>
                    <td>Dropped (Because Less Than 10% of Data)</td>    
                    </tr>
                    """

        output+="""                
                </table>                    
                """

    return df  # return new dataset

def describe(df): # count num of describe df
    desc = df.describe()
    # create a subplot without frame
    plot = plt.subplot(111, frame_on=False)
    # remove axis
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)
    # create the table plot and position it in the upper left corner
    table(plot, desc, loc='upper right')
    # save the plot as a png file
    plt.title("Descriptive Statistics")
    plt.savefig('desc_plot.png', dpi=1200)
    global output

    output+='''
            <h1 id="h1">Describe</h1>
            <p>*This is only for numerical types and the resolution might be low</p>
            <h2>Visualization</h2> 
            <img src="desc_plot.png" alt="Describe" class="ds"> 
            '''

    return output

def correlation(df, target):
    pass