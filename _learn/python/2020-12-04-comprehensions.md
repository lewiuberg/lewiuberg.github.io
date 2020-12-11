---
published: true #! <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE
# author: Lewi Lie Uberg # <-- private
show_date: true
# last_modified_at: 2021-01-01T00:00:00+01:00

title: Comprehensions
dark_title: false
excerpt: >
    "Python comprehensions"
permalink: /python-comprehensions/
collection: learn
canonical_url: "https://uberg.me/learn/python/python-comprehensions"
search: true

layout: single # (see _layout folder)
classes:
  # - wide
  # - dark_background
entries_layout: grid # list (default), grid
# taxonomy: # category/tag name

header:
  teaser: assets/images/learn/learn-600x400.png
  og_image: /assets/images/favicon/icon96x96.png #  useful for setting OpenGraph images on pages that don’t have a header or overlay image.
  # image: /assets/images/learn/learn-1200x200.png
  # image_description: "A description of the image"
  # caption: "Photo credit: [**Pixabay**](https://pixabay.com)"
  #overlay_color: "#333" # Solid color
  #overlay_image: /assets/images/about/whoami.png
  #overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  #overlay_filter: rgba(255, 0, 0, 0.5)
  # actions:
  #   - label: "Learn More"
  #     url: "https://unsplash.com"
  #   - label: "Learn More"
  #     url: "https://pixabay.com"

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

# Navigation set in _navigation.yml
sidebar:
  nav: "learn"

# categories:
#   - add
# tags:
#   - add
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

## List

```python
squares = [i * i for i in range(10)]
sentence = 'the rocket came back from mars'
vowels = [i for i in sentence if i in 'aeiou']

def is_consonant(letter):
    vowels = 'aeiou'
    return letter.isalpha() and letter.lower() not in vowels
consonants = [i for i in sentence if is_consonant(i)]
original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
prices = [i if i > 0 else 0 for i in original_prices]
```

## Set

```python
quote = "life, uh, finds a way"
unique_vowels = {i for i in quote if i in 'aeiou'}
```

## Dictionary

```python
squares = {i: i * i for i in range(10)}
squares
```