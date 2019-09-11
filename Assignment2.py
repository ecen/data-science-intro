# %% md
# # Assignment 2
_First_ we must be **bold**
* Students:
    * Davíð
    * Eric

# %% md
# ## 1. A company is considering using a system that will allow management to track staff behaviour in real-time, gather data on when and who they email,
# ## who accesses and edits files, and who meets whom when. The HR (human resources, or personnel) department at the company is in favour of
# ## introducing this new system and believes it will benefit the staff at the company. The company's management is also in favour.
# ## Discuss whether introducing this system raises potential ethical issues.
One useful approach to this issue is to look
at the benefits and costs that the stakeholders of the company would incur.
This system would benefit HR and the company's management in several ways.
Good communication within a company is vital for a company's success.
The HR department could see if there are individuals or groups within
the company that should be communicating but are not. They could then
potentially rectify the situation before the lack of communication has adverse
consequences.

This data collection would also beneficial for getting some
indication of who's productive and who's not. Workers who are productive
tend to access more files, email and meet more people. But this is an imperfect
indicator of performance so more in-depth analysis of an individuals performance
would have to be performed as well.

But from the standpoint of the individual worker, the introduction of this
system may not seem like a step in the right direction. The feeling of someone
constantly looking over your shoulder can give the impression that the company
doesn't trust you to do a good enough job on your own. Put in another way; if the comapny's interest don't align well enough with yours, you will probably feel like you are not on the same team then as the management and HR. That would have a bad effect on the company's morale and performance.

It then seems that HR and management have only something to gain and others
within the company have only something to lose. But this all depends on how
this system will be implemented and used. The company's management and HR could
have a policy of trusting the workers to do a good job (i.e. not looking and
analyzing the data on a regular basis) but using it only in cases of dispute,
where it's not clear who's responsible. This would benefit HR and management
but also employees, as more data is beneficial in the case of disputes.

The final outcome of the introduction of this system comes down to whether or not we can trust the HR department and the company's management
not to misuse the data. As such, there needs to be clear definitions of how the data will be used or employees could come to see unreasonable many negative effects in the future.

# %% Collect data.
# All properties with their room count and living area listed are included.

from lxml import html
from lxml import etree
import requests
from io import StringIO, BytesIO
import re

houseSizes = []
housePrices = []
houseRooms = []

parser = etree.HTMLParser(encoding="UTF-8")

# Iterate over the pagination. How many pages to go through is set manually.
for pageIndex in range(1, 11):
    # Request the current page and parse it
    url = 'https://www.hemnet.se/salda/bostader?location_ids%5B%5D=474180&page=' + str(pageIndex) + '&sold_age=all'
    data = requests.get(url).text

    tree = etree.parse(StringIO(data), parser)

    # List of all property divs on the current page
    propertyListings = tree.xpath('//div[@class="sold-property-listing"]')

    for property in propertyListings:
        # Select element that has sizes and rooms
        sizesRooms = property.xpath('./div[@class="sold-property-listing__size"]/div[@class="clear-children"]/div[@class="sold-property-listing__subheading sold-property-listing--left"]/text()')
        # Some properties doesn't have this exact html layout
        # because they have no houses built on them. Skip those.
        if (len(sizesRooms) == 0):
            continue
        sizesRooms = list(map(lambda x: re.sub("\xa0", "", x), sizesRooms))
        # Get the living area and convert to integer
        size = list(map(lambda x: re.findall("([0-9]*,*[0-9]*)m²", x), sizesRooms))[0][0]
        size = round(float(size.replace(",", ".")))

        def calcRoomCount(element):
            finds = re.findall("([0-9]*)rum", element)
            if (len(finds) == 0):
                 return "0"
            return finds[0]

        # Get room count
        rooms = list(map(calcRoomCount, sizesRooms))[0]
        rooms = int(round(float(rooms)))
        # If there were 0 rooms, lets ignore this data point
        if (rooms == 0):
            continue

        # Get the price of this property
        price = property.xpath('./div[@class="sold-property-listing__price"]/div[@class="clear-children"]/span[@class="sold-property-listing__subheading sold-property-listing--left"]/text()')
        price = price[0]
        price = re.sub("\xa0", "", price)
        price = re.sub("\n(.*)Slutpris ", "", price)
        price = re.sub(" kr\n", "", price)

        housePrices.append(int(price))
        houseSizes.append(int(size))
        houseRooms.append(int(rooms))

if (len(housePrices) != len(houseSizes) or len(housePrices) != len(houseRooms)):
    print("Assert array lengths failed.", len(housePrices), len(houseSizes), len(houseRooms))

# %% md
a. What are the values of the slope and intercept of the regression line?

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a pandas dataframe
data = {'Area': houseSizes, 'Price': housePrices}
df = pd.DataFrame(data)
df

# %%
# Prepare the data
df['Price'] = df['Price']/1000000
X = df['Area'].values.reshape(-1, 1)
y = df['Price'].values.reshape(-1, 1)


# Perform linear regression using Scikit-learn
regressor = LinearRegression()
regressor.fit(X, y)

print("The value of the slope is", round(regressor.intercept_[0],2))
print("The value of the coefficient is", round(regressor.coef_[0][0],2)

# Let's predict the prices based on the living area
y_pred = regressor.predict(X)
plt.scatter(X, y, color = 'gray');
plt.plot(X, y_pred, color = 'red', linewidth = 2);
plt.title('Living area vs price')
plt.xlabel('Living area ($m^2$)')
plt.ylabel('Price (Million SEK)')

# %% md
b. Use this model to predict the selling prices of houses which have living area
100 $m^2$, 150 $m^2$ and 200 $m^2$.

# %%
print("The predicted selling price (millions of SEK) of a house with a living area 100 m^2 is",
round(regressor.predict([[100]])[0][0]), 2)

print("The predicted selling price (millions of SEK) of a house with a living area 150 m^2 is",
round(regressor.predict([[150]])[0][0]), 2)

print("The predicted selling price (millions of SEK) of a house with a living area 200 m^2 is",
round(regressor.predict([[200]])[0][0]), 2)

#%% md
c. Draw a residual plot.

#%%
y_residual = y - y_pred
plt.scatter(X, y_residual)
plt.title('Living area vs residual')
plt.xlabel('Living area ($m^2$)')
plt.ylabel('Residual (Million SEK)')

frame = pd.DataFrame({'X': X.squeeze(), 'res': y_residual.squeeze()})
frame.describe()

print(frame[frame.res > 0].shape[0])
print(frame[frame.res < 0].shape[0])

#%% md
d. Discuss the results, and how the model could be improved.

Paying for location could mean that large houses are more isolated.

#%% md
3. Use a confusion matrix and 5-fold cross-validation to evaluate the use logistic regression to classify the iris data set.

#%%
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
