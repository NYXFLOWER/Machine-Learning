from random import shuffle
import pods
import zipfile
import pandas as pd # import the pandas library into a namespace called pd
# this ensures the plot appears in the web browser
# %matplotlib inline
import pylab as plt # this imports the plotting library in python

"""
Compute the probability for the number of deaths being over 40 
for each year we have in our film_deaths data frame. Store the 
result in a numpy array and plot the probabilities against the 
years using the plot command from matplotlib. Do you think the 
estimate we have created of  P(y|t)  is a good estimate? 
Write your code and your written answers in the box below.
"""

print("This is the Jupyter notebook")
print("It provides a platform for:")
words = ['Open', 'Data', 'Science']
for i in range(3):
    shuffle(words) # random sort "words" list.
    print(' '.join(words))


# pods.util.download_url('https://github.com/sjmgarnier/R-vs-Python/archive/master.zip')
#
# zip = zipfile.ZipFile('./master.zip', 'r')
# for name in zip.namelist():
#     zip.extract(name, '.')

film_deaths = pd.read_csv('./R-vs-Python-master/Deadliest movies scrape/code/film-death-counts-Python.csv')

film_deaths.describe()

print(film_deaths['Year'])
print(film_deaths['Body_Count'])

plt.plot(film_deaths['Year'], film_deaths['Body_Count'], 'rx')

film_deaths[film_deaths['Body_Count']>200]

film_deaths[film_deaths['Body_Count']>200].sort_values('Body_Count', ascending=False)

film_deaths['Body_Count'].hist(bins=20) # histogram the data with 20 bins. It counts the number of film in each section.
plt.title('Histogram of Film Kill Count')

plt.plot(film_deaths['Year'], film_deaths['Body_Count'], 'rx')
ax = plt.gca() # obtain a handle to the current axis
ax.set_yscale('log') # use a logarithmic death scale
# give the plot some titles and labels
plt.title('Film Deaths against Year')
plt.ylabel('deaths')
plt.xlabel('year')

deaths = (film_deaths.Body_Count>40).sum()  # number of positive outcomes (in sum True counts as 1, False counts as 0)
total_films = film_deaths.Body_Count.count()
print(deaths)
print(total_films)
prob_death = float(deaths)/float(total_films)
print("Probability of deaths being greather than 40 is:", prob_death)


print(deaths / total_films)

for year in [2000, 2002]:
    deaths = (film_deaths.Body_Count[film_deaths.Year==year]>40).sum()
    total_films = (film_deaths.Year==year).sum()

    prob_death = float(deaths)/float(total_films)
    print("Probability of deaths being greather than 40 in year", year, "is:", prob_death)

