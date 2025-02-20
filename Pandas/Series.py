import pandas as pd
import numpy as np

# groceries = pd.Series([30, 1, 'No', 'Yes'], ['Apples', 'Eggs', 'Milk', 'Bread'])
# print(groceries)
# print(groceries.size)
# print(groceries.shape)
# print(groceries.ndim)
# print(groceries.index)
# print(groceries.values)
# print('Eggs' in groceries)

# print(groceries['Eggs'])
# print(groceries[['Eggs', 'Bread']])
# print(groceries.loc[['Apples', 'Milk']])
# print(groceries.iloc[[0, 2]])
# groceries['Eggs'] = 50
# print('----------------------')
# print(groceries)
# groceries = groceries.drop('Milk')
# print(groceries)

fruits= pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])
print(fruits)
print('Original grocery list of fruits:\n ', fruits)

# We perform basic element-wise operations using arithmetic symbols
print()
print('fruits + 2:\n', fruits + 2) # We add 2 to each item in fruits
print()
print('fruits - 2:\n', fruits - 2) # We subtract 2 to each item in fruits
print()
print('fruits  *2:\n', fruits*  2) # We multiply each item in fruits by 2
print()
print('fruits / 2:\n', fruits / 2) # We divide each item in fruits by 2
print()


# We print fruits for reference
print('Original grocery list of fruits:\n', fruits)

# We apply different mathematical functions to all elements of fruits
print()
print('EXP(X) = \n', np.exp(fruits))
print()
print('SQRT(X) =\n', np.sqrt(fruits))
print()
print('POW(X,2) =\n',np.power(fruits,2)) # We raise all elements of fruits to the power of 2

# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)
print()

# We add 2 only to the bananas
print('Amount of bananas + 2 = ', fruits['bananas'] + 2)
print()

# We subtract 2 from apples
print('Amount of apples - 2 = ', fruits.iloc[0] - 2)
print()

# We multiply apples and oranges by 2
print('We double the amount of apples and oranges:\n', fruits[['apples', 'oranges']] * 2)
print()

# We divide apples and oranges by 2
print('We half the amount of apples and oranges:\n', fruits.loc[['apples', 'oranges']] / 2)

