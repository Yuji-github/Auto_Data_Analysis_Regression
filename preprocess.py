import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf

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
    plt.close()  # for next plot

    global output
    output+="""
            <section class="card">
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
                df[num_df.columns] = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)  # replace by KNN: Return the orignal values
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
                </section>                  
                """
    else:  # No missing values
        output+=""" 
                </section>                  
                """

    return df  # return new dataset

def describe(df, target): # count num of describe df
    desc = df.describe()

    global output
    # creating a table
    if desc.shape[1] > 0:
        # add ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        output+='''
                <section class="card">
                <h1 id="h1">Descriptive Statistics</h1>
                <h2>Results</h2> 
                <table class="table_center"> 
                    <tr>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                        <th>{:s}</th>
                    </tr>    
                '''.format("Column Name",
                           desc.axes[0][0], desc.axes[0][1],
                           desc.axes[0][2], desc.axes[0][3],
                           desc.axes[0][4], desc.axes[0][5],
                           desc.axes[0][6], desc.axes[0][7])

        for idx in range(desc.shape[1]):  # number of columns
            output+='''
                    <tr>
                    <td>{:s}</td>
                    <td>{:5f}</td>
                    <td>{:5f}</td> 
                    <td>{:5f}</td> 
                    <td>{:5f}</td> 
                    <td>{:5f}</td> 
                    <td>{:5f}</td> 
                    <td>{:5f}</td> 
                    <td>{:5f}</td>     
                    </tr>
                    '''.format(desc.axes[1][idx],
                               desc.values.T[idx][0], desc.values.T[idx][1],
                               desc.values.T[idx][2], desc.values.T[idx][3],
                               desc.values.T[idx][4], desc.values.T[idx][5],
                               desc.values.T[idx][6], desc.values.T[idx][7])
        output+="""                
               </table>                   
               """

    else:  # No descriptive statistics
        output+="""
                <section class="card">
                <h1 id="h1">Descriptive Statistics</h1>
                <h2>Sorry, No Numerical Information in the Dataset</h2> 
                <br>
                """

    # create box plot for target
    df.boxplot(target)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("box_target.png")
    plt.close()

    output+='''              
           <h2>Visualization</h2>  
           <p> Box Plot: {:s}</p>
           <img src="box_target.png" alt="boxplot" class="img"> 
           </section>                       
           '''.format(target)

    return output

def correlation(df, target):
    cor = df.corr()
    cor = cor.dropna(0, how='all')  # drop nan "rows" if the row contains nan only
    cor = cor.dropna(1, how='all')  # drop nan "columns" if the row contains nan only

    sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("corr.png")
    plt.close()  # for next plot

    detail = cor[target]  # correlation between given target vs others
    detail = detail.drop(target)  # drop nan "rows" if the row contains nan "any"

    plt.axhline(y=0.7, color='r', linestyle='-', label='Strong')  # Strong line
    plt.axhline(y=0.5, color='b', linestyle='-', label='Moderate')  # moderate line
    plt.axhline(y=-0.5, color='b', linestyle='-')  # moderate line
    plt.axhline(y=-0.7, color='r', linestyle='-')  # Strong line
    detail.plot(kind='bar', color='g', label='Correlation')
    plt.legend(bbox_to_anchor=(1, 0), loc="lower left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("corr_bar.png")
    plt.close()  # for next plot

    global output
    output+='''
            <section class="card">
            <h1 id="h1">Correlation</h1>
            <h2>Visualization</h2>  
            <p> Heat Map: Entire Dataset</p>
            <img src="corr.png" alt="correlation" class="img">  
            <br>
            <p> Bar Plot: Between {:s} and Other Columns</p>
            <img src="corr_bar.png" alt="correlation" class="img">             
            '''.format(target)

    # creating a table
    output+='''
            <h2>Table Results</h2>
            <table class="table_center"> 
            <tr>
            '''
    for itr in detail.axes[0]:
        output+='''
                <th>{:s}</th>   
                '''.format(str(itr))
    output+='''
            </tr>
            <tr>'''

    # inserting values
    corr_list = []  # for machine learning (later)

    for idx, val in enumerate(detail.values):
        output+='''
                <td>{:5f}</td>   
                '''.format(val)
        if val > 0.5 or val < -0.5:
            corr_list.append(detail.axes[0][idx])  # store column name which has moderate/strong correlations

    # closing this section
    output+='''
            </tr>
            </table>
            <h2>Summary of Correlation</h2>
            '''

    if len(corr_list) == 0:
        output+='''
                <p> All columns do not relate to {:s} </p>
                </section>
                '''.format(target)

    else:
        output+='''
                <p> {:d} columns relate to {:s} </p>
                <p> Because those go beyond thresholds (more than moderate) </p>
                </section>
                '''.format(len(corr_list), target)

    return output, corr_list

def outlier(df, target, corr_lis=None):
    df = df.drop(target, 1)  # drop target

    if len(corr_lis) < df.columns.size:  # the columns have correlations
        df = df[df.columns.intersection(corr_lis)]  # extract the intersections

    df.boxplot(column=list(df.columns.values))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("box_columns.png")
    plt.close()

    global output

    output+='''
            <section class="card">
            <h1 id="h1">Outliers</h1>
            <h2>Visualization</h2>  
            <p> Side-by-Side Box Plots</p>
            <img src="box_columns.png" alt="side_by_side" class="img">                   
            '''.format(target)

    # count outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    up = (df > (Q3 + 1.5 * IQR)).sum()
    down = (df < (Q1 - 1.5 * IQR)).sum()

    outers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

    output+='''
            <h2>Descriptive Results</h2> 
            <p> {:.3f} Percent of Data is Outliers </p>
            '''.format((outers.sum()/df.size))

    if (outers.sum()/df.size) > 0.05:  # more than 5% data is outliers
        win = [down.sum()/df.size, up.sum()/df.size]

        for col in df.columns[:-1]:
            df[col] = winsorize(df[col], limits=win)

        output+='''               
              <p> Applied Winsorizing </p>
              <table class="table_center"> 
              <tr>
              <th>Bottom</th>
              <th>Top</th>
              </tr>  
              <tr>
              <td>{:2f} %</td>
              <td>{:2f} %</td>    
              </tr>
              </table>
              '''.format(win[0]*100, win[1]*100)

        df.boxplot(column=list(df.columns.values))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("box_columns_outer.png")
        plt.close()
        output+='''
                <p> After Winsorizing </p>
                <img src="box_columns_outer.png" alt="side_by_side" class="img">
                '''

    output+='''
            </section>  
            '''

    return output, df

def adj_r_squared(df, target, dummy):
    corr_list = []  # to return
    global output
    output+='''
           <section class="card">
           <h1 id="h1">R-Squared with Dummy Variables</h1>
           <h2>Regression Results</h2>
           <table class="table_center"> 
           <tr>
           <th>Column Name</th>
           <th>R-squared</th>
           <th>Adj. R-squared</th>
           <th>F-statistic</th>
           <th>Prob (F-statistic)</th>
           </tr>  
           '''

    for dum in dummy:
        df_with_dummies = pd.get_dummies(data=df[dum], columns=[dum])  # convert onehot
        test = pd.concat([df[target], df_with_dummies], axis=1)

        formula = '{:s} ~'.format(target)
        for itr, val in enumerate(df_with_dummies.columns):  # creating formula
            if itr == 0:
                formula += str(val)
            else:
                formula += " + " + str(val)

        olsr_model = smf.ols(formula=formula, data=test)
        olsr_model_results = olsr_model.fit()
        # olsr_model_results.save('summary_' + dum + 'pkl')

        output += '''
                    <tr>
                    <td>{:s}</td>
                    <td>{:5f}</td>
                    <td>{:5f}</td> 
                    <td>{:5f}</td>
                    <td>{:5f}</td>   
                    </tr>                         
                    '''.format(dum, olsr_model_results.rsquared, olsr_model_results.rsquared_adj, olsr_model_results.fvalue, olsr_model_results.f_pvalue)

        if len(df_with_dummies.columns) <= 2:  # Bivariate
            if olsr_model_results.rsquared_adj > 0.1:  # this value is depends. There is no right/wrong threshold for r-squared
                corr_list.append(dum)
        else:  # ANOVA
            if olsr_model_results.f_pvalue < 0.05: # reject H0: all variables are the same:
                corr_list.append(dum)

    output+='''
            </table>
            </section>
            '''

    return output, corr_list