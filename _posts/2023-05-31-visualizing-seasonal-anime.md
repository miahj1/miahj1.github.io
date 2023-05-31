## Visualizing Seasonal Anime Data

Anime is released every season i.e. Spring, Winter, Fall, and Summer. The seasons spring and fall showcase huge swaths of multifarious releases unlike winter and summer which have very meager offerings. 
My goals for this project is to visualize the data after scraping every release for each season for the year of 2022. 

# Tools
1. Beautiful Soup 4 Python Module to scrape and gather the data.
2. Pandas to put a dataframe together using the scraped data.

# Metrics
1. On what day most shows release.
2. The highest rated show.
3. The genre of each show being released per season.
4. The release amount per season.
5. Word cloud of synopsis. 
6. Which studios released how many anime.

# Scraping
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
is the class of the element discussed before. This function will search inside the entire HTML to find the specific `div` tag and the coressponding class.

``` python
tv_new = soup.find('div', class_ = 'seasonal-anime-list js-seasonal-anime-list js-seasonal-anime-list-key-1')
```

We now have the entire base template where all the anime information is stored. When looking through the page with inspect element, there's a class
that each anime card has i.e. `js-anime-category-producer seasonal-anime js-seasonal-anime js-anime-type-all js-anime-type-1` which when iterated
through should give information on each individual show. 

There's multiple elements to each individual anime card such as title, rating, members, synopsis, genres, release dates, studiod, and number of episodes.
The member field is a total of all the users that have added the show to their list. There's a few more categorizes that I have not mentioned since
they are not useful for the analysis: all the avaliable parameters are showing in Fig. 1.

<figure>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/84815985/242366608-40a9afdb-c4ac-436b-981c-9504f7203133.png" alt="anime_card" style="width:50%">
  <figcaption><strong>Fig. 1</strong> Anime card template used for each show on the MAL seasonal anime section of their website.</figcaption>
</figure>
