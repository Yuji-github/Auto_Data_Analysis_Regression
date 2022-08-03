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

def estimator():
    clf1 = LinearRegression()
    clf2 = Ridge(random_state=0)
    clf3 = Lasso(random_state=0)
    clf4 = AdaBoostRegressor(random_state=0)
    clf5 = MLPRegressor(random_state=0)
    clf6 = SVR()
    clf7 = RandomForestRegressor(random_state=0)

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
    param5['hidden_layer_sizes'] = [7, 12, 50, 100]  # length = n_layers - 2
    param5['activation'] = ["tanh", "relu"]
    param5['solver'] = ["sgd", "adam"]
    param5['max_iter'] = [10, 50, 100, 200, 500]
    param5['early_stopping'] = [True]
    param5['learning_rate_init'] = [1e-3, 1e-2, 1e-1]

    param6 = {}
    param6['kernel'] = ['linear', 'rbf', 'sigmoid']
    param6['C'] = [1]
    param6['epsilon'] = [0.1]

    param7 = {}
    param7['n_estimators'] = [10, 50, 100]
    param7['max_depth'] = [3, 5, None]

    clf = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
    params = [param1, param2, param3, param4, param5, param6, param7]

    return clf, params

def model_test(df, target):  # df contains only numercial variables
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
              """.format(X, Y)

    best_best = 0

    for idx in range(3):
        x_train, x_test, y_train, y_test = split_df(df, split_config[idx])  # default: 80% train
        output += """
                    <h2> Training Info: Train Dataset {:.2f}</h2>
                    """.format((1 - split_config[idx]))

        for i in range(4):
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
            clf, params = estimator()
            best_score = 0
            best_model_name = ''
            for itr in tqdm(range(len(clf))):
                gs = GridSearchCV(clf[itr], params[itr], cv=2, n_jobs=-1).fit(x_train, y_train)
                if gs.best_score_ > best_score:
                    joblib.dump(gs, 'best.pkl')  # save best one
                    best_score = gs.best_score_
                    best_model_name = str(clf[itr])

            model = joblib.load("best.pkl")
            if model.best_score_ > best_best:
                joblib.dump(model, 'best_best.pkl')
                best_best = model.best_score_

            eval(y_test, model.predict(x_test), best_model_name.replace('()', ''))

            output+="""                       
                    </table>
                    <h3>Best Parameters: {:s}</h3>  
                    """.format(str(model.best_params_))

    output+="""     
            </section>  
            """
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
