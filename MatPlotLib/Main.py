import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
index = pokemon['type_1'].value_counts().index
sb.countplot(data = pokemon, y = 'type_1', order = index)

mp.show()