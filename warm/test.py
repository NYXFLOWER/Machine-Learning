import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pods
import zipfile

# pods.util.download_url('https://github.com/sjmgarnier/R-vs-Python/archive/master.zip')
# zip = zipfile.ZipFile('./master.zip', 'r')
# for name in zip.namelist():
#     zip.extract(name, '.')
film_deaths = pd.read_csv(
    './R-vs-Python-master/Deadliest movies scrape/code/film-death-counts-Python.csv')

year_max = film_deaths['Year'].max()
year_min = film_deaths['Year'].min()
year_num = year_max + 1 - year_min

year = np.array(range(year_min, year_max + 1))
probability = np.array(np.zeros(year_num))

i = 0
while i < year_num:
    i_year = year[i]
    total = (film_deaths.Year == i_year).sum()
    if total > 0:
        deaths = (film_deaths.Body_Count[film_deaths.Year == i_year] > 40).sum()
        if deaths > 0:
            probability[i] = float(deaths) / float(total)
        i += 1
    else:
        year = np.delete(year, i)
        year_num -= 1
        probability = np.delete(probability, i)
        continue

plt.plot(year, probability, 'ro')
plt.title('Probabilities of deaths being more than 40 against the years')
plt.ylabel('Probability')
plt.xlabel('year')