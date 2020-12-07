---
published: true
# last_modified_at: 2020-12-06T15:23:23+01:00
update: false
title: "Jupyter Dash in Jupyter Lab for macOS"
excerpt: "How to install Jupyter Dash on macOS"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
categories:
  - Data Science
  - Programming
  - Data Visualization
  - IDE
tags:
  # - Data Science
  - data science
# -----------------------------------------------------------------------------
  # - Programming
  # Python
  - python
  - pip
  - pyenv
# -----------------------------------------------------------------------------
  # - Data Visualization
  # Tools
  # python
  - plotly
  - dash
# -----------------------------------------------------------------------------
  # - IDE
  - jupyter notebooks
# -----------------------------------------------------------------------------
  # Misc
  - others work
---
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X5TVX1RNG8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X5TVX1RNG8');
</script>

## Python Virtual Environment:

1. Open the terminal and ```cd``` into the project main directory, or create a folder where you would like to start developing your project/app.
2. Create virtual environment: 
   ```pyenv virtualenv <python_version> <environment_name>```
3. Activate your new environment:
   ```pyenv local <environment_name>```
4. Install Node using homebrew (NPV included):
   ~~```brew install node```~~ <-- **Cant get that one to work, please advice if able.**
   [Download Node](https://nodejs.org/dist/v14.15.0/node-v14.15.0.pkg)
5. Install required libraries:
   * ```python3 -m pip install numpy```
   * ```python3 -m pip install pandas```
   * ```python3 -m pip install plotly```
   * ```python3 -m pip install dash```
   * ```python3 -m pip install Jupyterlab```
6. To run Dash inside Jupyter lab:
   ```python3 -m pip install jupyter-dash```
7. To run Plotly figures inside jupyter lab:
   ```python3 -m pip install jupyterlab ipywidgets```
8. Add JupyterLab extension for renderer support:
   * Required:
   ```jupyter labextension install jupyterlab-plotly```
   Example: ```fig.show()```
   * Optional widget extension:
   ```jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget```
   Example: ```fig```
9.  Rebuild:
   ~~```jupyter lab build```~~ Strictly not needed since it is done in step 8.
10. Start Jupyterlab
   ```jupyter lab```
11. Save dependencies for later:
    ```python3 -m pip freeze > requirements.txt```
12. Install dependencies:
    ```python3 -m pip install -r requirements.txt```

## Test code

### Inline:

```python
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Load Data
df = px.data.tips()
# Build App
app = JupyterDash(__name__)
app.layout = html.Div([
    html.H1("JupyterDash Demo"),
    dcc.Graph(id='graph'),
    html.Label([
        "colorscale",
        dcc.Dropdown(
            id='colorscale-dropdown', clearable=False,
            value='plasma', options=[
                {'label': c, 'value': c}
                for c in px.colors.named_colorscales()
            ])
    ]),
])
# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("colorscale-dropdown", "value")]
)
def update_figure(colorscale):
    return px.scatter(
        df, x="total_bill", y="tip", color="size",
        color_continuous_scale=colorscale,
        render_mode="webgl", title="Tips"
    )
# Run app and display result inline in the notebook
app.run_server(mode='inline')
```

### External:

```python
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Load Data
df = px.data.tips()
# Build App
app = JupyterDash(__name__)
app.layout = html.Div([
    html.H1("JupyterDash Demo"),
    dcc.Graph(id='graph'),
    html.Label([
        "colorscale",
        dcc.Dropdown(
            id='colorscale-dropdown', clearable=False,
            value='plasma', options=[
                {'label': c, 'value': c}
                for c in px.colors.named_colorscales()
            ])
    ]),
])
# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("colorscale-dropdown", "value")]
)
def update_figure(colorscale):
    return px.scatter(
        df, x="total_bill", y="tip", color="size",
        color_continuous_scale=colorscale,
        render_mode="webgl", title="Tips"
    )
# Run app and display result inline in the notebook
app.run_server(mode='external')
```

### JupyterLab:

```python
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Load Data
df = px.data.tips()
# Build App
app = JupyterDash(__name__)
app.layout = html.Div([
    html.H1("JupyterDash Demo"),
    dcc.Graph(id='graph'),
    html.Label([
        "colorscale",
        dcc.Dropdown(
            id='colorscale-dropdown', clearable=False,
            value='plasma', options=[
                {'label': c, 'value': c}
                for c in px.colors.named_colorscales()
            ])
    ]),
])
# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("colorscale-dropdown", "value")]
)
def update_figure(colorscale):
    return px.scatter(
        df, x="total_bill", y="tip", color="size",
        color_continuous_scale=colorscale,
        render_mode="webgl", title="Tips"
    )
# Run app and display result inline in the notebook
app.run_server(mode='jupyterlab')
```

## Terminal Script:

A script to define a new project and install everything from `init_requirements.txt` with pip, as well as installing the extensions.

**This is my first go at bash scripting, so be kind :)**

```bash
#!/bin/zsh
HEADER="   _  __             ___             _         __\n  / |/ /__ _    __  / _ \_______    (_)__ ____/ /_\n /    / -_) |/|/ / / ___/ __/ _ \  / / -_) __/ __/\n/_/|_/\__/|__,__/ /_/  /_/  \___/_/ /\__/\__/\__/\n                               |___/              "
clear
echo -e $HEADER
echo ""

while true;
do
    read -r -p "Make new project? (y/n): " NEW
    if [[ $NEW =~ ^([yY])$ ]]
    then
        # -----------------------------------------------------------------------------
        # Install version
        # -----------------------------------------------------------------------------
        while true;
        do
            read -r -p "Install new Python version? (y/n): " INSTALL
            if [[ $INSTALL =~ ^([yY])$ ]]
            then
                while true;
                do
                    read -r -p "Select python version to install (3.8.6): " VERSION
                    echo "Version to install:" $VERSION
                    read -r -p "Is this correct? (y/n): " CORRECT
                    if [[ $CORRECT =~ ^([yY][eE][sS]|[yY])$ ]]
                    then
                        VENVVERSION=$VERSION
                        pyenv install $VENVVERSION
                        break
                    else
                        clear
                        echo -e $HEADER
                        echo ""
                    fi
                done
                break
            elif [[ $INSTALL =~ ^([nN])$ ]]
            then
                # -----------------------------------------------------------------------------
                # Define virtual environment
                # -----------------------------------------------------------------------------
                while true;
                do
                    read -r -p "Select python version to use in environment (3.8.6): " VENVVERSION
                    echo "Version to use:" $VENVVERSION
                    read -r -p "Is this correct? (y/n): " CORRECT
                    if [[ $CORRECT =~ ^([yY][eE][sS]|[yY])$ ]]
                    then
                        break
                    else
                        clear
                        echo -e $HEADER
                        echo ""
                    fi
                done
                break
            fi
        done
        clear
        echo -e $HEADER
        echo ""
        echo "[STATUS] Python ready"
        echo ""
        
        while true;
        do
            read -r -p "Select name for environment (venv_name): " VENVNAME
            echo "Enviroment name:" $VENVNAME
            read -r -p "Is this correct? (y/n): " CORRECT
            if [[ $CORRECT =~ ^([yY][eE][sS]|[yY])$ ]]
            then
                break
            else
                clear
                echo -e $HEADER
                echo ""
            fi
        done
        
        pyenv virtualenv $VENVVERSION $VENVNAME
        pyenv local $VENVNAME
        clear
        echo -e $HEADER
        echo ""
        echo "[STATUS] Environment ready"
        echo ""
        break
    else
        break
    fi
done
# # -----------------------------------------------------------------------------
# # Pip Install
# # -----------------------------------------------------------------------------
while true;
do
    read -r -p "Install libraries with pip? (y/n): " LIB
    if [[ $LIB =~ ^([yY])$ ]]
    then
        pip install --upgrade pip
        pip install -r init_requirements.txt
        break
    else
        break
    fi
done

while true;
do
    read -r -p "Install Jupyter extensions? (y/n): " LIB
    if [[ $LIB =~ ^([yY])$ ]]
    then
        filename='init_jupyterlab_extensions.txt'
        # filename=$1
        while read line; do
            $line
        done < $filename
        #jupyter lab build
        break
    else
        break
    fi
done
```

Sources:

[Charming Data](https://drive.google.com/file/d/1ZRtQUie0y2k3dXz_MM8s29WQaSrM9bDn/view)

[Xing Han Lu](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e)
[Plotly](https://plotly.com/python/getting-started/#jupyterlab-support-python-35)

[Real Python](https://realpython.com/intro-to-pyenv/#virtual-environments-and-pyenv)