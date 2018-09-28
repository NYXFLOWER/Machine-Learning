from imdb import IMDb
import pandas as pd
from IPython.display import YouTubeVideo
YouTubeVideo('GX8VLYUYScM')

ia = IMDb()
movies = pd.read_csv('./R-vs-Python-master/Deadliest movies scrape/code/film-death-counts-Python.csv')

for movie in ia.search_movie('python'):
    print(movie)
