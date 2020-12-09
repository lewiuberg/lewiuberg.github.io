---
published: true
# author: Lewi Lie Uberg
# last_modified_at: 2020-01-01T00:00:00+01:00
title: "Comprehensions"
excerpt: "Comprehensions"
permalink: /python-comprehensions/
# classes: wide
sidebar:
  title: "Python"
  nav: learn-python
toc: true
toc_label: "Table of Contents"
toc_icon: "file-alt"
---
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X5TVX1RNG8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X5TVX1RNG8');
</script>

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
