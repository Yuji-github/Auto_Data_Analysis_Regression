from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math

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
    clf4 = GaussianProcessRegressor(random_state=0)
    clf5 = RadiusNeighborsRegressor()
    clf6 = MLPRegressor(random_state=0)
    clf7 = SVR()

    param1 = {}
    param1['fit_intercept'] = [True]

    param2 = {}
    param2['alpha'] = [1e-3, 1e-2, 1e-1, 1, 10]

    param3 = {}
    param3['alpha'] = [1e-3, 1e-2, 1e-1, 1, 10]

    param4 = {}
    param4['kernel'] = [RBF()]
    param4['alpha'] = [1e-10, 1e-5, 1e-1]

    param5 = {}
    param5['radius'] = [1e-3, 1e-2, 1.0, 2.5, 5.0]
    param5['weights'] = ["uniform", "distance"]
    param5['p'] = [1, 2]

    param6 = {}
    param6['hidden_layer_sizes'] = [4, 7, 12, 50, 100]  #   length = n_layers - 2
    param6['activation'] = ["identity", "logistic", "tanh", "relu"]
    param6['solver'] = ["lbfgs", "sgd", "adam"]
    param6['learning_rate_init'] = [1e-3, 1e-2, 1e-1]

    param7 = {}
    param7['kernel'] = ['linear', 'rbf', 'sigmoid']
    param7['C'] = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
    param7['epsilon'] = [1e-3, 1e-2, 1e-1, 1]

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

            clf, params = estimator()

            gs = GridSearchCV(clf[3], params[3], cv=2, n_jobs=-1).fit(x_train, y_train)
            print(gs.best_params_)
            # linear = LinearRegression()
            # linear.fit(x_train, y_train)
            # eval(y_test, linear.predict(x_test), "LinearRegression")
            #
            # ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], cv=10)
            # ridge.fit(x_train, y_train)
            # eval(y_test, ridge.predict(x_test), "Ridge")
            #
            # lasso = LassoCV(cv=10, random_state=0)
            # lasso.fit(x_train, y_train)
            # eval(y_test, lasso.predict(x_test), "Lasso")
            #
            # elastic = ElasticNetCV(cv=10, random_state=0)
            # elastic.fit(x_train, y_train)
            # eval(y_test, elastic.predict(x_test), "ElasticNet")

            output+="""   
                    </table>  
                    """

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
