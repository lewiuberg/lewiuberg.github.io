---
title: "Useful snippets"
categories:
  - post
tags:
  - matplotlib
  - plotly
  - decorators
  - pandas
  - comprehensions
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
---

Here are some snippets I like to have laying around for a rainy day :)

## Import

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import chart_studio.plotly as py
import cufflinks as cf
import dash_core_components as dcc
import dash_html_components as html
import country_converter as coco
import geopy

from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
from plotly.offline import init_notebook_mode, download_plotlyjs, plot, iplot
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
```

## Setup

### Matplotlib

```python
pd.set_option("display.max_rows", 85)
pd.set_option("display.max_columns", 85)
```

### Plotly Offline

```python
init_notebook_mode(connected=True)
cf.go_offline()
```

## Templates

### Decorator

```python
import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator
```

## Misc

### Check operating system

```python
def os_check():
    from sys import platform

    if platform == "darwin":
        print("macOS")
    elif platform == "win32":
        print("Windows")
    elif platform == "linux" or platform == "linux2":
        print("Linux")
    else:
        print("Oops! Something went wrong.")
```

### Check if path is a file or directory

```python
def path_checker(path):
    from pathlib import Path
    path = Path(path)

    if path.exists():
        if path.is_dir():
            print(f"'{path}' is directory")
        else:
            if path.is_file():
                print(f"'{path}' is a file")
        return True

    else:
        print(f"'{path}' does not exist.")
        return False
```

### Unpack list and return with Oxford comma

```python
def unpack_list(lst):  # Oxford comma
    if not isinstance(lst, str):
        lst = [str(item) for item in lst]
    if len(lst) == 0:
        return
    if len(lst) == 1:
        return ", ".join(lst)
    if len(lst) == 2:
        return ", and ".join(lst) 
    else:
        first_part = lst[:-1]
        last_part = lst[-1]
        return ", ".join(first_part) + ", and " + last_part
```

### Combine two lists to a dictionary

```python
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
months_alpha = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Des']
dictionary = [{'label': x,'value': y} for x, y in zip(months, months_alpha)]
```

## Pandas

### Word search in a DataFrame

```python
def word_search(df, *words):
    if not words or len(words[0]) < 1:
        return

    col_count: int = 0
    sum_words: int = 0
    found_words: str = []

    if isinstance(words[0], str):
        words = [word for word in words]
    else:
        words = list(*words)
        print(words)

    for word in words:
        col_count = 0
        sum_word = 0
        for column in df:
            if df[column].dtype == object or df[column].dtype == str:
                col_count += 1
                sum_word += df[column].str.contains(f"^{word}$").sum()
                if df[column].str.contains(f"^{word}$").any():
                    if word not in found_words:
                        found_words.append(word)
        sum_words += sum_word
    if len(found_words) == 0:
        found_words = words
    print("Columns of dtype str or object:", col_count)
    print(
        f"Instances of {unpack_list(found_words)} in the dataframe: {sum_words}"
    )
```

### Find missing values in a DataFrame

```python
def find_missing_values(df):
    column_names = (df.columns[df.isnull().any() == True]).format()
    miss_columns = df.isna().any().sum()
    miss_values = df.isna().sum().sum()
    print(f"Instances of missing data: {miss_values}")
    print(f"Columns with missing data: {miss_columns}")
    print(f"Column names with missing data: {unpack_list(column_names)}")
```

### Find location of missing values in a DataFrame

```python
def missing_location(df):
    col_criteria = df.isnull().any(axis=0)
    miss_col = df[col_criteria.index[col_criteria]]

    miss_only = miss_col[miss_col.isnull().any(axis=1)]

    row_criteria = df.isnull().any(axis=1)
    miss_row = df[row_criteria]
    
    return miss_col, miss_row, miss_only
```

### Find location data from a DataFrame column

```python
def df_locatinon_data(df, search_col):
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    geolocator = Nominatim(user_agent="my_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=.1)
    # Find the location.
    df['location'] = df[search_col].apply(geocode)
    # Extract point to its own columns.
    df['point'] = df['location'].apply(lambda loc: tuple(loc.point)
                                       if loc else None)
    # split point column into latitude, longitude and altitude columns.
    df[['latitude', 'longitude',
        'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

    return df
```

### Change name of index or row items, or to datetime. 

```python
def replace_df_ax_name(df, find, replace_with="", axis=0):

    dff = df.copy()

    if axis == 1:  # <-- Columns
        dff = dff.T

    dff_row = dff.index.to_list()
    dff_dict = {i: dff_row[i] for i in range(len(dff_row))}

    change_index: list = []
    change_dict: dict = {}

    for i, v in dff_dict.items():
        if find in v:
            change_index.append(i)
            if replace_with == "d_to_datetime":
                v = pd.to_datetime(v)
            else:
                v = v.replace(find, replace_with)
            change_dict[i] = v

    dff_dict.update(change_dict)
    dff.index = list(dff_dict.values())

    if axis == 1:  # <-- Columns
        dff = dff.T

    return dff
```

## Comprehensions

[Real Python](https://realpython.com/list-comprehension-python/)

### List

```python
squares = [i * i for i in range(10)]
```
```python
sentence = 'the rocket came back from mars'
vowels = [i for i in sentence if i in 'aeiou']

def is_consonant(letter):
    vowels = 'aeiou'
    return letter.isalpha() and letter.lower() not in vowels
consonants = [i for i in sentence if is_consonant(i)]
```
```python
original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
prices = [i if i > 0 else 0 for i in original_prices]
```

### Set

```python
quote = "life, uh, finds a way"
unique_vowels = {i for i in quote if i in 'aeiou'}
```

### Dictionary

```python
squares = {i: i * i for i in range(10)}
squares
```
