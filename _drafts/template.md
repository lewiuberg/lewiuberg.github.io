---
published: false #! <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE <-- HERE
# author: Lewi Lie Uberg # <-- private
show_date: true
# last_modified_at: 2021-01-01T00:00:00+01:00

title: Title of Your Post
dark_title: false
excerpt: >
    "A unique line of text to describe this post that will display in an archive listing and meta description with SEO benefits."
permalink: /template/
collection: template
canonical_url: "https://uberg.me/custom-canonical-url"
search: true

layout: splash # (see _layout folder)
classes:
  - wide
  - dark_background
entries_layout: grid # list (default), grid
# taxonomy: # category/tag name

header:
  teaser: assets/images/brains/600x400/brains_600x400_18.png
  og_image: /assets/images/favicon/icon96x96.png #  useful for setting OpenGraph images on pages that don’t have a header or overlay image.
  image: /assets/images/brains/1600x1200/brains_1600x1200_18.png
  image_description: "A description of the image"
  caption: "Photo credit: [**Pixabay**](https://pixabay.com)"
  overlay_color: "#333" # Solid color
  overlay_image: /assets/images/about/whoami.png
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  overlay_filter: rgba(255, 0, 0, 0.5)
  actions:
    - label: "Learn More"
      url: "https://unsplash.com"
    - label: "Learn More"
      url: "https://pixabay.com"

feature_row:
  - image_path: /assets/images/brains/600x400/brains_600x400_17.png
    alt: "brain 17"
    title: "Brain 17"
    excerpt: "This is a description of brain 17"
    url: "/here_goes_the_permalink/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_18.png
    alt: "brain 18"
    title: "Brain 18"
    excerpt: "This is a description of brain 18"
    url: "/here_goes_the_permalink/"
    btn_class: "btn--primary"
    btn_label: "Learn more"

gallery:
  - url: /assets/images/brains/1200x800/brains_1200x800_17.png
    image_path: assets/images/brains/600x400/brains_600x400_17.png # -th
    alt: "Brain 17"
  - url: /assets/images/brains/1200x800/brains_1200x800_18.png
    image_path: assets/images/brains/600x400/brains_600x400_18.png # -th
    alt: "Brain 18"

read_time: true
words_per_minute: 200

author_profile: true
share: true
comments: true
related: true

toc: true
toc_label: "Table of Contents"
toc_icon: "file-alt" # https://fontawesome.com/icons?d=gallery&s=solid&m=free

# Custom sidebar
sidebar:
  - title: "Role"
    image: /assets/images/lewi/lewi-uberg-round.png
    image_alt: "logo"
    text: "<i>Developer, Designer.</i>"
  - title: "Responsibilities"
    text: "<i>Everything from A to Z.</i>"

# Navigation set in _navigation.yml
sidebar:
  nav: "docs"

categories:
  - add
tags:
  - add
---
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X5TVX1RNG8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X5TVX1RNG8');
</script>

{% include feature_row %}

{% include gallery caption="Gallery of my brain." %}
