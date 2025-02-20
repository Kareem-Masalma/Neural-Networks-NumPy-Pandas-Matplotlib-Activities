import pandas as pd

# items = {
#     'Bob': pd.Series([245, 212, 30]),
#     'Alice': pd.Series([40, 110, 15, 20])
# }

items = {
    'Bob': pd.Series([245, 212, 30], ['Bikes', 'Pants', 'Watches']),
    'Alice': pd.Series([40, 110, 15, 20], ['Books', 'Glasses', 'Bikes', 'Pants'])
}

cart = pd.DataFrame(items)
print(cart)
# print(cart.index)
# print(cart.columns)
# print(cart.values)

print(cart.loc[['Bikes']])



# # We create a list of Python dictionaries
# items2 = [{'bikes': 20, 'pants': 30, 'watches': 35},
#           {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]
#
# # We create a DataFrame  and provide the row index
# store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])
#
# # We display the DataFrame
# print(store_items)
#
# items2 = [{'bikes': 20, 'pants': 30, 'watches': 35},
#           {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]
#
# # We create a DataFrame
# store_items = pd.DataFrame(items2)
#
# # We display the DataFrame
# print(store_items)

