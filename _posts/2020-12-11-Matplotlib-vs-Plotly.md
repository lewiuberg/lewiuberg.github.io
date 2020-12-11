---
published: true #! <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE
# author: Lewi Lie Uberg # <-- private
show_date: true
# last_modified_at: 2021-01-01T00:00:00+01:00

title: Matplotlib vs Plotly
dark_title: false
excerpt: >
    Graph created with Matplotlib vs embeded graph created with Plotly.
permalink: /matplotlib-vs-Plotly/
collection: posts
canonical_url: "https://uberg.me/posts/matplotlib-vs-plotly"
search: true

layout: single # (see _layout folder)
# classes:
  # - wide
  # - dark_background
entries_layout: list # list (default), grid
# taxonomy: # category/tag name

header:
  teaser: assets/images/posts/Matplotlib-vs-Plotly/1-th.png
  og_image: /assets/images/favicon/icon96x96.png #  useful for setting OpenGraph images on pages that don’t have a header or overlay image.
  image: /assets/images/posts/Matplotlib-vs-Plotly/1.png
  image_description: "Plotly graph"
  # caption: "Photo credit: [**Pixabay**](https://pixabay.com)"
  # overlay_color: "#ffffff" # Solid color
  overlay_image: /assets/images/posts/Matplotlib-vs-Plotly/1.png
  overlay_filter: 0.3 # same as adding an opacity of 0.5 to a black background
  # overlay_filter: rgba(255, 0, 0, 0.5)
  actions:
    - label: "Download dataset"
      url: "/assets/datasets/eLearning_Employee_Satisfaction.csv"
    # - label: "Learn More"
    #   url: "https://pixabay.com"

# feature_row:
#   - image_path: /assets/images/brains/600x400/brains_600x400_17.png
#     alt: "brain 17"
#     title: "Brain 17"
#     excerpt: "This is a description of brain 17"
#     url: "/here_goes_the_permalink/"
#     btn_class: "btn--primary"
#     btn_label: "Learn more"
#   - image_path: /assets/images/brains/600x400/brains_600x400_18.png
#     alt: "brain 18"
#     title: "Brain 18"
#     excerpt: "This is a description of brain 18"
#     url: "/here_goes_the_permalink/"
#     btn_class: "btn--primary"
#     btn_label: "Learn more"

# gallery:
#   - url: /assets/images/brains/1200x800/brains_1200x800_17.png
#     image_path: assets/images/brains/600x400/brains_600x400_17.png # -th
#     alt: "Brain 17"
#   - url: /assets/images/brains/1200x800/brains_1200x800_18.png
#     image_path: assets/images/brains/600x400/brains_600x400_18.png # -th
#     alt: "Brain 18"

read_time: true
words_per_minute: 200

author_profile: true
share: true
comments: true
related: true

toc: true
toc_label: "Table of Contents"
toc_icon: "file-alt" # https://fontawesome.com/icons?d=gallery&s=solid&m=free

# # Custom sidebar
# sidebar:
#   - title: "Role"
#     image: /assets/images/lewi/lewi-uberg-round.png
#     image_alt: "logo"
#     text: "<i>Developer, Designer.</i>"
#   - title: "Responsibilities"
#     text: "<i>Everything from A to Z.</i>"

# # Navigation set in _navigation.yml
# sidebar:
#   nav: "docs"

categories:
  - Data Science
  - Programming
  - Data Visualization

tags:
  # - Data Science
  - data science

# -----------------------------------------------------------------------------
  # - Programming
  # Python
  - python

# -----------------------------------------------------------------------------
  # - Data Visualization
  # Tools
  # python
  - matplotlib
  - plotly
---
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X5TVX1RNG8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X5TVX1RNG8');
</script>

{% comment %}
<!-- {% include feature_row %}

{% include gallery caption="Gallery of my brain." %} -->{% endcomment %}

## Matplotlib vs Plotly (with embeded graph)

```python
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
```

### Data preparation

```python
df = pd.read_csv("Dataset.csv")
#df = pd.read_csv("eLearning_Employee_Satisfaction.csv")
```


```python
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>Content satisfaction</th>
      <th>Experience satisfaction</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019.0</td>
      <td>AUG</td>
      <td>78</td>
      <td>75</td>
      <td>85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>SEP</td>
      <td>78</td>
      <td>72</td>
      <td>85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>OCT</td>
      <td>77</td>
      <td>72</td>
      <td>85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NOV</td>
      <td>76</td>
      <td>71</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>DEC</td>
      <td>75</td>
      <td>73</td>
      <td>85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020.0</td>
      <td>JAN</td>
      <td>74</td>
      <td>71</td>
      <td>85</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>FEB</td>
      <td>72</td>
      <td>70</td>
      <td>85</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>MAR</td>
      <td>70</td>
      <td>68</td>
      <td>85</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>APR</td>
      <td>73</td>
      <td>73</td>
      <td>85</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>MAY</td>
      <td>80</td>
      <td>80</td>
      <td>85</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>JUN</td>
      <td>80</td>
      <td>79</td>
      <td>85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>JUL</td>
      <td>81</td>
      <td>80</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.loc[0:4, "YEAR"] = 2019
df.loc[5:11, "YEAR"] = 2020
df[["YEAR","Content satisfaction", "Experience satisfaction", "Target"]] = df[["YEAR","Content satisfaction", "Experience satisfaction", "Target"]].astype("int")
#df.set_index(["YEAR", "MONTH"], inplace=True)
df["Period"] = df["MONTH"].astype("str") + "\n" + df["YEAR"].astype("str")
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>Content satisfaction</th>
      <th>Experience satisfaction</th>
      <th>Target</th>
      <th>Period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>AUG</td>
      <td>78</td>
      <td>75</td>
      <td>85</td>
      <td>AUG\n2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>SEP</td>
      <td>78</td>
      <td>72</td>
      <td>85</td>
      <td>SEP\n2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>OCT</td>
      <td>77</td>
      <td>72</td>
      <td>85</td>
      <td>OCT\n2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019</td>
      <td>NOV</td>
      <td>76</td>
      <td>71</td>
      <td>85</td>
      <td>NOV\n2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>DEC</td>
      <td>75</td>
      <td>73</td>
      <td>85</td>
      <td>DEC\n2019</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020</td>
      <td>JAN</td>
      <td>74</td>
      <td>71</td>
      <td>85</td>
      <td>JAN\n2020</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>FEB</td>
      <td>72</td>
      <td>70</td>
      <td>85</td>
      <td>FEB\n2020</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020</td>
      <td>MAR</td>
      <td>70</td>
      <td>68</td>
      <td>85</td>
      <td>MAR\n2020</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020</td>
      <td>APR</td>
      <td>73</td>
      <td>73</td>
      <td>85</td>
      <td>APR\n2020</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020</td>
      <td>MAY</td>
      <td>80</td>
      <td>80</td>
      <td>85</td>
      <td>MAY\n2020</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020</td>
      <td>JUN</td>
      <td>80</td>
      <td>79</td>
      <td>85</td>
      <td>JUN\n2020</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020</td>
      <td>JUL</td>
      <td>81</td>
      <td>80</td>
      <td>85</td>
      <td>JUL\n2020</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[5,1:2]
```

```out
MONTH    JAN
Name: 5, dtype: object
```

```python
type(df["Content satisfaction"][0])
```

```out
numpy.int64
```

```python
type(df.iloc[0,0:1])
```

```out
pandas.core.series.Series
```

```python
plt.plot(df.MONTH, df["Content satisfaction"])
plt.show()
```

![image-center](/assets/images/posts/Matplotlib-vs-Plotly/output_9_0.png){: .align-center}

### Graph with Matplotlib

```python
x = df.MONTH
y1 = df["Content satisfaction"]
y2 = df["Experience satisfaction"]
y3 = df["Target"]

y1_color = "steelblue"
y2_color = "lightblue"
y3_color = "black"

markersize = 50
value_fontsize = 14
value_fontsize_title = 22

fontfamily = "Sans"

fig, ax = plt.subplots(figsize=(12, 6))

plt.title("    eLearning Employee Satisfaction",
          loc="left",
          pad=25,
          fontsize=value_fontsize_title,
          alpha=1,
          fontweight="bold")

xs = df.MONTH
ys1 = y1
ys2 = y2
ys3 = y3

ax.plot(x, y1, linewidth=3, color=y1_color)
ax.scatter(xs[[0, 7, 11]], ys1[[0, 7, 11]], color=y1_color, s=markersize)
ax.annotate(f"{ys1[0]:.0f}%", (xs[0], ys1[0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y1_color)

ax.annotate(f"{ys1[7]:.0f}%", (xs[7], ys1[7]),
            textcoords="offset points",
            xytext=(-17, 18),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y1_color)

ax.annotate(f"{ys1[11]:.0f}% Content satisfaction", (xs[11], ys1[11]),
            textcoords="offset points",
            xytext=(10, -3),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y1_color)

ax.plot(x, y2, linewidth=3, color=y2_color)
ax.scatter(xs[[0, 7, 11]], ys2[[0, 7, 11]], color=y2_color, s=markersize)
ax.annotate(f"{ys2[0]:.0f}%", (xs[0], ys2[0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y2_color)

ax.annotate(f"{ys2[7]:.0f}%", (xs[7], ys2[7]),
            textcoords="offset points",
            xytext=(-17, -18),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y2_color)

ax.annotate(f"{ys2[11]:.0f}% Experience satisfaction", (xs[11], ys2[11]),
            textcoords="offset points",
            xytext=(10, -3),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y2_color)

ax.plot(x, y3, linewidth=3, color=y3_color, linestyle=(0, (5, 1)), alpha=.5)
#ax.scatter(xs[11], ys3[11], color=y3_color, s=markersize)
ax.annotate(f"{ys3[11]:.0f}% Target", (xs[11], ys3[11]),
            textcoords="offset points",
            xytext=(10, -3),
            ha="left",
            fontweight="bold",
            fontsize=value_fontsize,
            color=y3_color,
            alpha=.75)

# ax.set_xlim()
ax.xaxis.set_ticks_position('none')
# ax.xaxis.set_tick_params(labelsize=value_fontsize)
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(value_fontsize)
    tick.label1.set_fontweight("bold")
    tick.label1.set_alpha(.75)

plt.figtext(.1401,
            .04,
            "2019                                                 2020",
            fontsize=value_fontsize,
            fontweight="bold",
            color="black",
            fontfamily=fontfamily,
            alpha=.75)

ax.set_ylim(65, 85)
ax.set_yticks([])

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)

plt.savefig("post.png")
```

![image-center](/assets/images/posts/Matplotlib-vs-Plotly/output_11_0.png){: .align-center}

### Graph with Plotly

```python
import plotly.graph_objects as go

x = df.MONTH
y1 = df["Content satisfaction"]
y2 = df["Experience satisfaction"]
y3 = df["Target"]

y1_color = "steelblue"
y2_color = "lightblue"
y3_color = "black"

markersize = 10
value_fontsize = 14
value_fontsize_title = 22

fontfamily = "Arial"

p = [0, 7, 11]

# Create traces
fig = go.Figure()

# Trace 1
fig.add_trace(
    go.Scatter(x=x,
               y=y1,
               mode="lines",
               name="Content satisfaction",
               line=dict(color=y1_color, width=3)))
fig.add_trace(
    go.Scatter(
        x=x.take([0, 7, 11]),
        y=y1.take([0, 7, 11]),
        mode="markers",  # "markers+text"
        #name="Experience satisfaction",
        marker=dict(color=y1_color, size=markersize),
#         text=[f"{y2[p[0]]}%", f"{y2[p[1]]}%"],
#         textposition="top center"
    )
)

# Trace 2
fig.add_trace(
    go.Scatter(x=x,
               y=y2,
               mode="lines",
               name="Experience satisfaction",
               line=dict(color=y2_color, width=3)))
fig.add_trace(
    go.Scatter(
        x=x.take([0, 7, 11]),
        y=y2.take([0, 7, 11]),
        mode="markers",  # "markers+text"
        #name="Experience satisfaction",
        marker=dict(color=y2_color, size=markersize),
#         text=[f"{y2[p[0]]}%", f"{y2[p[1]]}%"],
#         textposition="top center"
    )
)

# Trace 3
fig.add_trace(
    go.Scatter(x=x,
               y=y3,
               mode="lines",
               name="Target",
               line=dict(color=y3_color, width=2, dash="dash")))

fig.update_traces(hovertemplate = '%{y:.2f}<extra></extra>')

# Annotations
annotations = []

# Title
annotations.append(
    dict(xref="paper",
         yref="paper",
         x=0.05,
         y=1.0,
         xanchor="left",
         yanchor="bottom",
         text="<b>eLearning Employee Satisfaction</b>",
         font=dict(size=value_fontsize_title,
                   color="rgb(37,37,37)"),
         showarrow=False))

# Trace 1
annotations.append(
    dict(x=x[0],
         y=y1[0],
         yanchor="bottom",
         text=f"<b>{y1[0].take(0)}%</b>",
         font=dict(color=y1_color),
         showarrow=False,
         xshift=0,
         yshift=10))

annotations.append(
    dict(x=x[7],
         y=y1[7],
         yanchor="bottom",
         text=f"<b>{y1[7].take(0)}%</b>",
         font=dict(color=y1_color),
         showarrow=False,
         xshift=0,
         yshift=18))

annotations.append(
    dict(x=x[11],
         y=y1[11],
         xanchor="left",
         text=f"<b>{y1[11].take(0)}% Content satisfaction</b>",
         font=dict(color=y1_color),
         showarrow=False,
         xshift=10,
         yshift=0))

# Trace 2
annotations.append(
    dict(x=x[0],
         y=y2[0],
         yanchor="bottom",
         text=f"<b>{y2[0].take(0)}%</b>",
         font=dict(color=y2_color),
         showarrow=False,
         xshift=0,
         yshift=10))

annotations.append(
    dict(x=x[7],
         y=y2[7],
         yanchor="top",
         text=f"<b>{y2[7].take(0)}%</b>",
         font=dict(color=y2_color),
         showarrow=False,
         xshift=0,
         yshift=-10))

annotations.append(
    dict(x=x[11],
         y=y2[11],
         xanchor="left",
         text=f"<b>{y2[11].take(0)}% Experience satisfaction</b>",
         font=dict(color=y2_color),
         showarrow=False,
         xshift=10,
         yshift=0))

# Trace 3
annotations.append(
    dict(x=x[11],
         y=y3[11],
         xanchor="left",
         text=f"<b>{y3[11].take(0)}% Target<b>",
         font=dict(color=y3_color),
         showarrow=False,
         xshift=10,
         yshift=0))

fig.update_xaxes(gridcolor="rgba(0,0,0,0)",tickfont=dict(color="black", size=14), tickprefix="<b>",ticksuffix ="</b><br>")
fig.update_yaxes(gridcolor="rgba(0,0,0,0)",showticklabels=False)

fig.update_layout(annotations=annotations,
                  autosize=True,
                  width=1000,
                  height=500,
                  showlegend=False,
                  paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",
                  font_size=value_fontsize,
                  modebar={"bgcolor": "rgba(255,255,255,0.0)"},
                  hovermode="closest",
                  xaxis=dict(
                      tickmode = 'array',
                      tickvals = [x for x in range(0,12)],
                      ticktext = ['AUG<br>2019', 'SEP', 'OCT', 'NOV', 'DES', 'JAN<br>2020', "FEB", "MAR", "APR", "MAY", "JUN", "JUL"]))

fig.layout.font.family = fontfamily

fig.show()
```

![image-center](/assets/images/posts/Matplotlib-vs-Plotly/1.png){: .align-center}

### Exporting Plotly Graph

#### export the graph as a HTML page

```python
import plotly.io as pio
pio.write_html(fig, file="index.html", auto_open=True)
```

#### Export the graph to chart studio

```python
import chart_studio.plotly as py
py.plot(fig, filename="eLearning_Employee_Satisfaction", auto_open=True)
```

```out
https://plotly.com/~lewiuberg/48/
```

### Embedding Plotly Graph

```python
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~lewiuberg/48')
```

```out
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~lewiuberg/48.embed" height="525" width="100%"></iframe>
```

#### The result

{: .notice--info}
<b>Info!</b> This graph is not optimized for mobile view or narrow screens.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~lewiuberg/48.embed" height="525" width="100%"></iframe>
