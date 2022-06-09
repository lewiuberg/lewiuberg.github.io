---
published: true #! <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE
# author: Lewi Lie Uberg # <-- private
show_date: false
# last_modified_at: 2021-01-01T00:00:00+01:00

title: Final Degree Project
dark_title: true
excerpt: >
    "An increasing rate of deadly brain tumors in humans also sees the increasing needfor highly educated medical personnel like neurologists and radiologists for diagno-sis and treatment. Thus, to reduce the workload and the time from initial suspi-cion of disease to diagnosis and a suitable treatment plan, there is a need to im-plement a Computer-Aided-Disease-Diagnosis (CADD) system for brain tumor clas-siﬁcation. By studying the types of tumors involved, how the convolutional neuralnetwork functions, the evolution of its pre-deﬁned architectures, models using pre-trained weights, and their application in brain tumor classiﬁcation, the likelihood ofproducing a promising CADD system increases heavily. The outcome of the re-search conducted in this project presents the starting point of an open-source projectto further develop a CADD system for brain tumor classiﬁcation with reliable results.The project includes all components of a working CADD system, including the datapreprocessing pipeline, the pipeline for deﬁning and training CNN classiﬁcation mod-els, and a user interface in the form of an API as the backend and a website asthe frontend. The project is intended to be open to the general public—however, itsprimary focus is on facilitating medical imaging researchers, medical students, radi-ologic technologists, and radiologists."
permalink: /final-degree/
# collection: pypi
canonical_url: "https://uberg.me/final-degree/"
search: true

layout: final-degree
# classes:
  # - wide
  # - dark_background
# entries_layout: grid # list (default), grid
# taxonomy: # category/tag name

header:
  teaser: assets/images/portfolio/final_degree/final_degree_01-th.png
  # og_image: /assets/images/favicon/icon96x96.png #  useful for setting OpenGraph images on pages that don’t have a header or overlay image.
  # image: /assets/images/brains/1600x1200/brains_1600x1200_18.png
  # image_description: "A description of the image"
  # caption: "Photo credit: [**Pixabay**](https://pixabay.com)"
  # overlay_color: "#333" # Solid color
#   overlay_image: /assets/images/portfolio/pypi/pypi_01.png
  # overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  # overlay_filter: rgba(255, 0, 0, 0.5)
#   actions:
#     - label: "<i class='fab fa-python'></i> Visit my PyPi"
#       url: "https://pypi.org/user/lewiuberg/"

# feature_row:
#   - image_path: /assets/images/portfolio/pypi/pypi_02.png
#     alt: "visual-automata"
#     title: "Visual Automata"
#     excerpt: "Visual Automata is a Python 3 library I built as a wrapper for the Automata library to add more visualization features."
#     url: "https://pypi.org/project/visual-automata/"
#     btn_class: "btn--primary"
#     btn_label: "Visit"
#   - image_path: /assets/images/portfolio/pypi/pypi_04.png
#     alt: "confprint"
#     title: "ConfPrint"
#     excerpt: "ConfPrint provides a simple way to make predefined printer configurations."
#     url: "https://pypi.org/project/confprint/"
#     btn_class: "btn--primary"
#     btn_label: "Visit"

# gallery:
#   - url: /assets/images/brains/1200x800/brains_1200x800_17.png
#     image_path: assets/images/brains/600x400/brains_600x400_17.png # -th
#     alt: "Brain 17"
#   - url: /assets/images/brains/1200x800/brains_1200x800_18.png
#     image_path: assets/images/brains/600x400/brains_600x400_18.png # -th
#     alt: "Brain 18"

# read_time: false
# words_per_minute: 200

# author_profile: true
# share: true
# comments: true
# related: true

# toc: true
# toc_label: "Table of Contents"
# toc_icon: "file-alt" # https://fontawesome.com/icons?d=gallery&s=solid&m=free

# Custom sidebar
# sidebar:
#   - title: "Role"
#     image: /assets/images/lewi/lewi-uberg-round.png
#     image_alt: "logo"
#     text: "<i>Developer, Designer.</i>"
#   - title: "Responsibilities"
#     text: "<i>Everything from A to Z.</i>"

# Navigation set in _navigation.yml
# sidebar:
#   nav: "docs"

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

<!-- {% include feature_row %} -->
{{ content }}

<!-- {% include gallery caption="Gallery of my brain." %} -->
