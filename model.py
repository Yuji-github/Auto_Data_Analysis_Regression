from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
import joblib
from tqdm import tqdm

output = '' # global

def split_df(df, size=0.2):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=0)

    return x_train, x_test, y_train, y_test

def transformer(x_train, x_test, num=1):
    if num == 1:
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        scaler = "StandardScaler"
    elif num == 2:
        rc = RobustScaler()
        x_train = rc.fit_transform(x_train)
        x_test = rc.transform(x_test)
        scaler = "RobustScaler"
    elif num == 3:
        nc = MinMaxScaler()
        x_train = nc.fit_transform(x_train)
        x_test = nc.transform(x_test)
        scaler = "MinMaxScaler"
    elif num == 4:
        mc = MaxAbsScaler()
        x_train = mc.fit_transform(x_train)
        x_test = mc.transform(x_test)
        scaler = "MaxAbsScaler"

    return x_train, x_test, scaler

def estimator(Neural):
    clf1 = LinearRegression()
    clf2 = Ridge(random_state=0)
    clf3 = Lasso(random_state=0)
    clf4 = AdaBoostRegressor(random_state=0)
    clf5 = SVR()
    clf6 = RandomForestRegressor(random_state=0)

    param1 = {}
    param1['fit_intercept'] = [True]

    param2 = {}
    param2['alpha'] = [1e-3, 1e-2, 1e-1, 1, 10]

    param3 = {}
    param3['alpha'] = [1e-3, 1e-2, 1e-1, 1, 10]

    param4 = {}
    param4['n_estimators'] = [10, 50, 100]
    param4['learning_rate'] = [1, 1.2, 1.5]

    param5 = {}
    param5['kernel'] = ['linear', 'rbf', 'sigmoid']
    param5['C'] = [0.1, 1]
    param5['epsilon'] = [0.01, 1]

    param6 = {}
    param6['n_estimators'] = [10, 50, 100, 150]
    param6['max_depth'] = [3, 5, None]

    if Neural == True:
        clf7 = MLPRegressor(random_state=0)
        param7 = {}
        param7['hidden_layer_sizes'] = [7, 12, 50, 100]  # length = n_layers - 2
        param7['activation'] = ["tanh", "relu"]
        param7['solver'] = ["sgd", "adam"]
        param7['max_iter'] = [10, 50, 100, 200, 500]
        param7['early_stopping'] = [True]
        param7['learning_rate_init'] = [1e-3, 1e-2, 1e-1]

        clf = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
        params = [param1, param2, param3, param4, param5, param6, param7]
    else:
        clf = [clf1, clf2, clf3, clf4, clf5, clf6]
        params = [param1, param2, param3, param4, param5, param6]

    return clf, params

def model_test(df, target, Neural):  # df contains only numercial variables
    split_config = [0.3, 0.2, 0.1]
    trans_config = [1, 2, 3, 4]

    global output
    X = "X Variables: "
    for idx, val in enumerate(df.columns):
        if idx == len(df.columns) - 1:  # idx == target
            break

        if idx != len(df.columns) - 2:  # if not the last x variables
            X += val + ", "
        else:  # if the last
            X += val

    Y = "Y Variables: {:s}".format(target)

    output+="""
              <section class="card">
              <h1 id="h1">Model Selection</h1>
              <h2> Data Info</h2>
              <P> {:s} </p>
              <P> {:s} </p>
              <h2>Applied Models</h2>
              <P>Linear Models: LinearRegression, Ridge, Lasso</p>
              <P>Ensemble Models: AdaBoost, Random Forest</p>
              <P>SVM Models: SVR</p>
              <P>Neural Network Models: MLPRegressor *if Neural is True</p>
              """.format(X, Y)

    best_best = 0.0
    best_best_name = ''
    best_data_size = 0
    best_best_parm = {}
    best_best_scaler = ''

    for idx in range(len(split_config)):
        x_train, x_test, y_train, y_test = split_df(df, split_config[idx])  # default: 80% train
        output += """
                    <h2> Training Info: Train Dataset {:.2f}</h2>
                    """.format((1 - split_config[idx]))

        for i in range(len(trans_config)):
            x_train, x_test, scaler = transformer(x_train, x_test, trans_config[i])  # default: StandardScaler
            output+="""
                    <P> Applied Scaler: {:s} </p>  
                    <table class="table_center"> 
                    <tr>
                    <th>Model Name</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>MAPE</th>
                    <th>R2</th>
                    </tr>            
                    """.format(scaler)

            # Finding the Best Model and Parameters
            clf, params = estimator(Neural)
            best_score = 0
            best_model_name = ''
            for itr in tqdm(range(len(clf))):
                gs = GridSearchCV(clf[itr], params[itr], cv=2, n_jobs=-1).fit(x_train, y_train)  #  n_jobs=-1 is using all CPUs
                if gs.best_score_ > best_score:
                    joblib.dump(gs, 'best.pkl')  # save a local best one
                    best_score = gs.best_score_
                    best_model_name = str(clf[itr])

            model = joblib.load("best.pkl")

            # store the best of the best model
            if model.best_score_ > best_best:
                joblib.dump(model, 'best_best.pkl')
                best_best = model.best_score_
                best_best_name = str(model.estimator)
                best_data_size = (1 - split_config[idx])*100
                best_best_parm = model.best_params_
                best_best_scaler = scaler

            eval(y_test, model.predict(x_test), best_model_name.replace('()', ''))

            output+="""                       
                    </table>
                    <h3>Best Parameters: {:s}</h3>  
                    """.format(str(model.best_params_))

    output+="""     
            </section>  
            """

    best_out='''
            <section class="card">
            <h1 id="h1">Best Model</h1>
            <h3>Best Model: {:s}</h3> 
            <h3>Best Scaler: {:s}</h3>
            <h3>Best Parameters: {:s}</h3> 
            <h3>Best Data Size: {:d}</h3> 
            <h3>Best Score: {:.3f}</h3>              
            </section>  
            '''.format(best_best_name, best_best_scaler, str(best_best_parm), int(best_data_size), best_best)

    output = best_out + output  # switch orders

    return output

def eval(y_true, y_pred, modelName):

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    global output
    output+="""           
            <tr>
            <td>{:s}</td>
            <td>{:.3f}</td>
            <td>{:.3f}</td>  
            <td>{:.3f}</td>  
            <td>{:.3f}</td>  
            <td>{:.3f}</td>      
            </tr>                
            """.format(modelName, mse, rmse, mae, mape, r2)
