from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

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
    elif num == 2:
        rc = RobustScaler()
        x_train = rc.fit_transform(x_train)
        x_test = rc.transform(x_test)
    elif num == 3:
        nc = MinMaxScaler()
        x_train = nc.fit_transform(x_train)
        x_test = nc.transform(x_test)

    return x_train, x_test

def model_test(df, target):  # df contains only numercial variables
    split_config = [0.3, 0.2, 0.1]
    trans_config = [1, 2, 3]
    # for idx in range(3):
    x_train, x_test, y_train, y_test = split_df(df, split_config[1])  # default: 80% train
    x_train, x_test = transformer(x_train, x_test, trans_config[0])  # default: StandardScaler


