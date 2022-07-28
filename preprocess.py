import pandas as pd
import numpy as np

def miss(df): # count num of missing values
    missing = df.isnull().sum()
    missing_percent = np.true_divide(missing, len(df))
    missing_dict = missing.to_dict()

    output = """
            <html>
            <head>Missing Values</head>
            <body>
            <p> {:.2f} </p>

            </body>
            </html>
            """.format(np.average(missing_percent))

    return output