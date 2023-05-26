## Visualizing My Spotify or Last.fm Listening Data
I'm starting a project based on my listening data.

# Goal and Plans
There are graphics that show up at the end of the year on Spotify from an event called wrapped. 
Wrapped puts together meaningful statistics about the user's listening activity. The issue with 
the current state of the graphics is that they are not as meaningful as sought out to be
which we'll see more on later, and there's this really odd ball limitation of only allowing mobiles users 
to generate or even lookat the graphics--users who actively use the desktop or web player are ignored. 
After the event is over, there's no way to access those graphics again. If let's say a user wanted to see the ones 
from past years of active listening, they aren't able to. These are many of the pitfalls that come with
this event, so my mission is to make actual meaningful visualizations using my Spotify or Last.fm data depending
on which API is less strict to me abusing them.

Let's take a look at some of the statistics wrapped shows:
1. Top 5 Genres
2. [Listening Personality](https://engineering.atspotify.com/2023/01/whats-a-listening-personality/) is a calculated visual that takes abitrary parameters.
3. Minutes Listened, Top Artists, Top Songs, and Top Genre
4. [Audio Day](https://techcrunch.com/wp-content/uploads/2022/11/Audio-Day-Share.png) which consists of three sections: nights, afternoons, and mornings. 
Atleast three or four random adjectives are used to describe how the listener was feeling during each section of the day.
6. [Top Song](https://techcrunch.com/wp-content/uploads/2022/11/Top-Song-Share.png), Date the song was most listened to, Amount of streams for this one song
7. Amount of Artists Listened To
8. [Audio Aura](https://newsroom.spotify.com/2021-12-01/learn-more-about-the-audio-aura-in-your-spotify-2021-wrapped-with-aura-reader-mystic-michaela/) is a gradient
   style mixture where each color represents a mood e.g. purple represents an energetic listener, green represents a mindful listener, pink represents an optimistic listener, orange
   represents a rebellious listener, yellow represents a motivated listener, and blue represents an emotional listener.

Some of the statistics such as audio aura displays a meshed visual of colors. If a listener just looked at it, they wouldn't understand what any of the colors
meant--perhaps this is why it was removed from next year's wrapped. Overall, the team is having a lot of fun seeing what sticks to the wall and removing what doesn't
in later years. However, I won't try to do every statistic on this list, but it would be an interesting sub-goal to achieve.

# Tools
+ Python
+ R
+ Spotify Python API Wrapper
+ Last.FM Python API Wrapper

Python will not be used to finalize any visualizations. The matplotlib module is frustrating to use so much that the spotify dev team 
came up with their own module called [chartify](https://github.com/spotify/chartify) instead of falling into that awful rabbit hole I will use R to finalize visuals.
