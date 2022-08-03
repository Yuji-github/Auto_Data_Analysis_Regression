# Auto Data Analysis Regression

## Motivation
One of the daily challenges for data scientists is selecting an optimal model. To select the optimal model, we need to work on data cleaning, such as handling missing values and outliers. Even if we carefully clean up the data, there is no promise of meeting the optimal model. On top of that, all datasets are unique and require different approaches to discover an optimal model. Several data scientists prefer to use Random Forest as their first choice because the model works well in general. However, the model is expensive, and there might be a better model if we tune it up well. The auto data analysis for regression suggests an optimal model, hyper-parameters, and proportion of train datasets.

## 1. Requirnents
- python ==> 3.9
- py7zr
- Chrome Broweser   

### Install:
```shell
pip install openpyxl
```

## 2. Execution
### Analysis Regression
```shell
python main.py -I demo.csv -T AveragePrice -R 0 Date -D type region
```

For the more details of parameters, please run:
```shell
python main.py --help
```

### Re-Open Browser
Example:
```shell
python decompress.py -F demo.7z
```

For the more details of parameters, please run:
```shell
python decompress.py --help
```

## 3. Explain 
### Missing Values
<img src="./src/missing.png" alt="demo miss" title="auto miss">

Auto Analysis detects missing values as its first step, and it visualizes them with a bar plot.
If any missing values are detected, the model applies KNN-imputation for numerical variables and mode for categorical variables.
