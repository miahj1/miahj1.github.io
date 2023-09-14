# Scraping Seasonal Anime Data from MyAnimeList with Beautiful Soup

Anime is released every season i.e. Spring, Winter, Fall, and Summer. The seasons spring and fall showcase huge swaths of multifarious releases unlike winter and summer which have very meager offerings. 
My goals for this project is to scrape data from MAL to be used for an event I run every season with friends to watch all the new shows that release every new anime season to then finally scope out which shows 
are worth watching. When I wasn't into scraping, I used to manually scan the website and copy each title and post them in Notion which was rather frustrating and time-consuming: this script has saved me so much time. I can
spend time with my friends without needing to break a sweat before the event.

**Disclaimer:** Let's assume MAL doesn't have an API, at the time of this project I didn't know since I know now what I didn't know back then I can't be bothered to use the API after coding this program.

## Tools
1. Beautiful Soup 4 Python Module to scrape and gather the data.
2. Pandas to put a dataframe together using the scraped data.

## Scraping
If there’s a need to capture large amount of data, messaging the website owner would make a lot of sense; however, this isn’t the case for this project. 
When it comes to scraping websites, BS4 module in Python is the best. Let’s load up the relevant modules—requests library is required to fetch content 
from the website of our choice. The variable `website` is assigned the information fetched from using the requests' function 
`get()` and then a bs4 object e.g. `soup` is declared to parse the HTML information. `lxml` argument works really well with this page, depending
on the page this may need to be changed for bs4 to scrape properly.

```python
from bs4 import BeautifulSoup
import requests

website = requests.get('https://myanimelist.net/anime/season').text
soup = BeautifulSoup(website, 'lxml')
```

The seasonal anime page is displayed using a parent class named `seasonal-anime-list js-seasonal-anime-list js-seasonal-anime-list-key-1`. In the code below,
a `tv_new` object is decalred and assigned the content found by bs4's `find()` function where the first function argument is a `div` and the second argument
is the class of the element discussed before. This function will search inside the entire HTML to find the specific `div` tag and the coressponding class: the
coressponding frame contains the cards of each show.

``` python
tv_new = soup.find('div', class_ = 'seasonal-anime-list js-seasonal-anime-list js-seasonal-anime-list-key-1')
```

We now have the entire base template where all the anime information is stored. When looking through the page with inspect element, there's a class
that each anime card has i.e. `js-anime-category-producer seasonal-anime js-seasonal-anime js-anime-type-all js-anime-type-1` which when iterated
through should give information on each individual show. 

Before we loop through each card, let's declare an empty list to store the scraped content.

```python
animes = []
```

Looping through each element of the `anime_body` gives us each card.

```python
for anime_body in tv_new.find_all('div', class_='js-anime-category-producer seasonal-anime js-seasonal-anime js-anime-type-all js-anime-type-1'):
    pass
```

The classname or css selectors seperated by spaces are multiples classes: there's a common misconception where beginner developers tend to think the entire string is a singular class
without knowing that classes cannot have spaces. `pass` is used in the body of the for loop as a placeholder: a special keyword in Python.

There's multiple elements to each individual anime card such as title, rating, members, synopsis, genres, release dates, studios, and number of episodes.
The member field is a total of all the users that have added the show to their list. There's a few more categorizes that I have not mentioned since
they are not useful for what I ultimately want which the name of the show, the air date of the show, and the rating of the show. I also want
shows that are only on their first season not second, third, or forth. All the avaliable parameters are shown in Fig. 1.

<p align="center">
  <img src="https://github.com/miahj1/miahj1.github.io/assets/84815985/e4ee8f20-07fb-4aba-86c8-159bd6eb5f16" alt="Anime card from myanimelist.">
</p>

<p align="center"><strong>Figure 1:</strong> <i>Anime card template used for each show on the MAL seasonal anime section of their website.</i></p><br>

We'll from now on work inside the scope of the for loop, to get the title of the show inside each anime card is a class aptly named `title`. 
Below is the corressponding `html` from the website.

```html
<div class="title">
  <div class="title-text">
    <h2 class="h2_anime_title">
      <a
        href="https://myanimelist.net/anime/31964/Boku_no_Hero_Academia"
        class="link-title"
        >Boku no Hero Academia</a
      >
    </h2>
  </div>
  ----- snip -----
</div>
```

The `div` with class `title` contains an `h2` and `a` tag; we'll have to dig down to get the title of the show. 
Let's use the trust `find()` function to help us by passing in the type of tag which is in this case a `div` 
and the name of the class which is `title`. However, this only gets the element of the parent class: what we want
is the child class, so let's declare a variable `final_title` which uses `find()` on the `anime_title` element.
We'll pass in the `a` tag and class `link-title`. The member function `text` extracts the string inside the tag.

```python
    anime_title = anime_body.find('div', class_ = 'title')
    final_title = anime_title.find('a', class_= 'link-title').text
```

Now, let's grab the airing date for each show. The html structure on the website looks like:

```html
<div class="info">
  <span class="item">Apr 3, 2016</span>
  ----- snip -----
</div>
```

The parent `div` with class `info` contains the air date of the show in the `span` tag that has the child class `item`. 
Let's first, use `find()` to get the parent element.

```python
    air_date_body = anime_body.find('div', class_='info')
```

We can now search inside the `air_date_body`.

```python
    air_date = air_date_body.find('span', class_='item').text
```

If you notice, this is the same method we used to grab the title of the show. Moving on, let's try
to grab the rating of each show. Our tried and true method won't work for the snippet below.

```html
<div class="scormem-container">
  <div class="scormem-item score score-label score-7" title="Score">
    <i class="fa-regular fa-star mr4"></i>7.88
  </div>
  ----- snip -----
</div>
```

Why wouldn't it work? The child class of `scormem-container` which is `scormem-item score score-label score-7` is not static: 
it is a dynamic class where the value `7` changes based on the rating of the show. `7` can be assumed as being *n* 
that corresponds to any integer from 1 to 10. If there is no rating, the website falls back on using `na`. The function `find()`
just doesn't work for classes that have different variants that's where `select()` comes to the rescue.

`select()` allows choosing tags that match two or more css classes. Let's declare a `rating` variable where we assign 
the variable the select function's return value. The function is passed the name of the class; however, spaces need
to be replaced with periods for the function to work.

```python
    rating = anime_body.select("div.scormem-item.score.score-label")
```

If we tried doing `rating.text`, we would get an error since the select function returns a `ResultSet` object. 
The object contains html snippets of every rating as shown below.

```html
<div class="scormem-item score score-label score-7" title="Score">
  <i class="fa-regular fa-star mr4"></i>7.62
</div>
```

To pre-process the rating and get it to the formatting, we want we can use the built-in `split()` function.
Before we begin, let's cast the `rating` variable to type string otherwise `split()` can't do its magic.
Splitting on the closing italics tag `</i>` gives us `['[<div class="scormem-item score score-label score-7" title="Score">\n<i class="fa-regular fa-star mr4">', '7.33\n        </div>]']`
which is a list of two items since we only want the last item we can use the subscript `[-1]` to access it.

```python
    preprocess_rating = (str(rating).split('</i>'))[-1].split('\n')[-2]
```

The rating is now in the format `7.33\n        </div>]` to fix this: let's split again on the newline character, `\n`.
Next, we'll use subscript `[-2]` on the result which is `['7.38', '        </div>]']` to retrieve the rating.

We're almost done; we were able to get the title, air date, and rating: the only thing left is to figure out a way to filter out seasons that aren't the first season. It's a 
bit of a complicated approach, anime series tend to use different variations on the season either its the final season, the second stage, or the last part. What I
found to work for most releases is to just look for the string `season` in the synopsis section; however, this is not a fool proof method. 

Let's take a look at the html we're working with regarding the synopsis section.

```html
<div class="synopsis js-synopsis">
  <p class="preline">
    The appearance of "quirks," newly discovered super powers, has been steadily
    increasing over the years, with 80 percent of humanity possessing various
    abilities from manipulation of elements to shapeshifting. This leaves the
    remainder of the world completely powerless, and Izuku Midoriya is one such
    individual. Since he was a child, the ambitious middle schooler has wanted
    nothing more than to be a hero. Izuku's unfair fate leaves him admiring
    heroes and taking notes on them whenever he can. But it seems that his
    persistence has borne some fruit: Izuku meets the number one hero and his
    personal idol, All Might. All Might's quirk is a unique ability that can be
    inherited, and he has chosen Izuku to be his successor! Enduring many months
    of grueling training, Izuku enrolls in UA High, a prestigious high school
    famous for its excellent hero training program, and this year's freshmen
    look especially promising. With his bizarre but talented classmates and the
    looming threat of a villainous organization, Izuku will soon learn what it
    really means to be a hero. [Written by MAL Rewrite]
  </p>
  ----- snip -----
</div>
```

The parent class `synopsis js-synopsis` is a `div` that contains a `p` or paragraph tag--everything in that `p` tag is what we want.

Let's declare a variable named `synopsis_body` which uses the `find()` member function: I won't give much of an explanation for the code. 
We have already performed the same method previously twice.

```python
    synopsis_body = anime_body.find('div', class_='synopsis js-synopsis')
    synopsis = synopsis_body.find('p', class_='preline').text
```
To filter out shows that aren't the first season, checking if `season` is not in the synopsis will work in this case.

```python
    if 'season' not in synopsis:
        animes.append(f'{air_date} - {final_title} - {preprocess_rating}')
```

The code will run the condition and if that condition is true: the code block inside the condition executes where
the list `animes` is inserted with the `air_date`, `final_title` and `preprocess_rating` values. `f-strings` are
used to allow combining these variables into one string.

If we use a loop to print out the contents of the list, we get:

```
Apr 1, 2016 - Kagewani: Shou - 6.41
Apr 1, 2016 - Mayoiga - 5.49
Apr 1, 2016 - Neko mo, Onda-ke - 5.07
Apr 1, 2016 - Uchuu Patrol Luluco - 7.54
Apr 1, 2016 - Ushio to Tora (TV) 2nd Season - 7.90
Apr 10, 2016 - Concrete Revolutio: Choujin Gensou - The Last Song - 6.99
Apr 10, 2016 - Flying Witch - 7.51
Apr 10, 2016 - High School Fleet - 7.31
Apr 10, 2016 - Tonkatsu DJ Agetarou - 7.13
Apr 11, 2016 - Sansha Sanyou - 7.10
Apr 12, 2016 - Usakame - 5.76
Apr 12, 2016 - Wagamama High Spec - 5.38
Apr 16, 2016 - Big Order (TV) - 5.36
  ----- snip -----
Apr 9, 2016 - Tanaka-kun wa Itsumo Kedaruge - 7.83
Jun 7, 2016 - Honobono Log - 7.33
Mar 14, 2016 - Ji Jia Shou Shen: Baolie Feiche - N/A
May 3, 2016 - Muzumuzu Eighteen - N/A
May 6, 2016 - Sore Ike! Sabuibo Mask - N/A
```

This is awesome, but none of it is in a specific order. If we look at the output, there are two shows that shouldn't 
be on the list Ushio to Tora (TV) 2nd Season, and Concrete Revolutio: Choujin Gensou - The Last Song. I did say 
that method before wasn't fool proof.

Organizing the data could be performed just by using a Pandas dataframe. Let's import the module and change
the values appended to the animes list to a list. The list would look like `[air_date, final_title, preprocess_rating]`.
This approaches allows us to send in the data to `pd.DataFrame` for Pandas to populate with data. The `columns`
argument is used to name each item in the list as follows Air Date, Title, and Rating.

```python
import pandas as pd

    if 'season' not in synopsis:
        animes.append([air_date, final_title, preprocess_rating])

anime_df = pd.DataFrame(animes, columns=['Air Date', 'Title', 'Rating'])
print(anime_df)
```

When the dataframe is completed, we can print it out and this is what we have now.

```
        Air Date                                              Title Rating
0    Apr 3, 2016                              Boku no Hero Academia   7.88
1    Apr 4, 2016              Re:Zero kara Hajimeru Isekai Seikatsu   8.23
2    Apr 7, 2016                                  Bungou Stray Dogs   7.82
3    Apr 2, 2016  JoJo no Kimyou na Bouken Part 4: Diamond wa Ku...   8.50
4    Apr 8, 2016                            Koutetsujou no Kabaneri   7.27
5    Apr 8, 2016                                  Sakamoto desu ga?   7.55
6    Apr 9, 2016                                          Kiznaiver   7.38
7    Apr 7, 2016       Netoge no Yome wa Onnanoko ja Nai to Omotta?   6.70
8    Apr 6, 2016                                 Sousei no Onmyouji   7.30
9   Apr 16, 2016                        Magi: Sinbad no Bouken (TV)   7.84
10   Apr 9, 2016                      Tanaka-kun wa Itsumo Kedaruge   7.83
  ----- snip -----
54   May 6, 2016                             Sore Ike! Sabuibo Mask    N/A
55  Mar 14, 2016                    Ji Jia Shou Shen: Baolie Feiche    N/A
56   May 3, 2016                                  Muzumuzu Eighteen    N/A
```

Looks a lot better than what we had before. Let's sort the each anime based on its air date, we'll need to
convert the column to `datetime` to make this easy. We'll use the `apply()` function from Pandas which
executre a function we give on to every data point in the column `Air Date`. The argument we're sending in to 
do that conversion is `pd.to_datetime`.

```python
anime_df["Air Date"] = anime_df["Air Date"].apply(pd.to_datetime)
anime_df = anime_df.sort_values(by="Air Date")
```

After that, we can sort the dataframe using `sort_values()` where the `by` argument is set to the column that'll
be sorted. In our case, that column is `Air Date`. The df is sorted: let's see the results.

```
     Air Date                                              Title Rating
55 2016-03-14                    Ji Jia Shou Shen: Baolie Feiche    N/A
53 2016-04-01                                   Neko mo, Onda-ke   5.07
15 2016-04-01                                            Mayoiga   5.49
21 2016-04-01                      Ushio to Tora (TV) 2nd Season   7.90
17 2016-04-01                                Uchuu Patrol Luluco   7.54
43 2016-04-01                                     Kagewani: Shou   6.41
20 2016-04-02        Gyakuten Saiban: Sono "Shinjitsu", Igi Ari!   6.50
3  2016-04-02  JoJo no Kimyou na Bouken Part 4: Diamond wa Ku...   8.50
11 2016-04-02                  Gakusen Toshi Asterisk 2nd Season   7.00
  ----- snip -----
9  2016-04-16                        Magi: Sinbad no Bouken (TV)   7.84
14 2016-04-16                                     Big Order (TV)   5.36
56 2016-05-03                                  Muzumuzu Eighteen    N/A
54 2016-05-06                             Sore Ike! Sabuibo Mask    N/A
28 2016-06-07                                       Honobono Log   7.33
```

That's all I really wanted. If there's a challenge you want to partake in, you can try
sorting by the rating number which would require imputing the missing values 
and converting the values to Pandas floating point type. You can also try
making a more robust algorithm for ignoring anime that aren't from this respective
season. 

Thank you for following along with me this far. :)
