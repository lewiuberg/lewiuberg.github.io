---
layout: splash
permalink: /
hidden: true
read_time: false
# classes:
#   - dark_background
dark_title: true
header:
  #overlay_color: "#ffffff"
  overlay_image: /assets/images/home/home_03.png
  # actions:
  #   - label: "<i class='fas fa-download'></i> Install now"
  #     url: "/docs/quick-start-guide/"

  # <br />
  # <small><a href="https://github.com/mmistakes/minimal-mistakes/releases/tag/4.21.0">Latest release v4.21.0</a></small>

title: Lewi Uberg
excerpt: >
  Personal website, portfolio and blog.
feature_row:
  - image_path: /assets/images/brains/600x400/brains_600x400_15.png
    alt: "about"
    title: "About"
    excerpt: "Information about me."
    url: "/about/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_4.png
    alt: "posts"
    title: "Posts"
    excerpt: "Blog posts."
    url: "/posts/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_5.png
    alt: "portfolio"
    title: "Portfolio"
    excerpt: "Portfolio of some of my projects."
    url: "/portfolio/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_17.png
    alt: "learn"
    title: "Learn"
    excerpt: >
      Learn different subject related to programming or data science with me.
      This will be a reference for myself, that I share with you.
    url: "/learn/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_1.png
    alt: "medium"
    title: "Medium"
    excerpt: "A collection of Medium articles."
    url: "/medium/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /assets/images/brains/600x400/brains_600x400_9.png
    alt: "hobbies"
    title: "Hobbies"
    excerpt: "Hobbies I like to use my spare time on."
    url: "/hobbies/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  
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
