---
# published: true
# last_modified_at: 2020-01-01T00:00:00+01:00
permalink: /about/
title: "About"
excerpt: "About me"
read_time: false
show_date: true
# toc: true
toc_label: "On this page"
toc_icon: "cogs"
---
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X5TVX1RNG8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X5TVX1RNG8');
</script>

## Information

```python
me = Person(
    name="Lewi",
    surname="Uberg",
    birthdate=datetime.date(1984, 5, 11),
    education="Final year student"
              "Bachelor of Applied Science"
              "BASc, Data Science",
    address="Smedplassveien 18 B",
    zip_code="4848",
    city="Arendal",
    telephone="95002946",
    email="lewi@uberg.me"
)
```

{: .container}
<div>
Hello!

I will write a little bio here.
</div>

## Contact me

<i class="fas fa-hand-point-left"></i> use these links or contact me directly here.

<!-- modify this form HTML and place wherever you want your form -->
<form
  action="https://formspree.io/f/mgepljqa"
  method="POST"
>
  <label>
    Your name:
    <input type="text" name="name">
  </label>
  <label>
    Your email:
    <input type="text" name="_replyto">
  </label>
  <label>
    Your message:
    <textarea name="message"></textarea>
  </label>

  <!-- your other form fields go here -->

  <button type="submit">Send</button>
</form>
