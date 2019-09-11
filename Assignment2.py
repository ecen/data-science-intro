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

The final outcome of the introduction of this system depends on several
assumptions. Should we trust the HR department and the company's management
not to misuse the data? Is this sort of data collection compliant with laws on
individual privacy? We believe this system should only be implemented with these
questions answered in addition to having clear restrictions on how the
data may be used.

The final outcome of the introduction of this system comes down to whether or not we can trust the HR department and the company's management
not to misuse the data. As such, there needs to be clear definitions of how the data will be used. Without such restrictions

# %%
from lxml import html
from lxml import etree
import requests
from io import StringIO, BytesIO
import re

for i in range(1, 11):
url = 'https://www.hemnet.se/salda/bostader?location_ids%5B%5D=474180&page=1&sold_age=all'
data = requests.get(url).text

parser = etree.HTMLParser(encoding="UTF-8")
tree = etree.parse(StringIO(data), parser)

addresses = tree.xpath('//div[@class="sold-property-listing"]/div[@class="sold-property-listing__location"]/h2[@class="sold-property-listing__heading"]/span[@class="item-result-meta-attribute-is-bold item-link"]/text()')

prices = tree.xpath('//div[@class="sold-property-listing"]/div[@class="sold-property-listing__price"]/div[@class="clear-children"]/span[@class="sold-property-listing__subheading sold-property-listing--left"]/text()')
prices = list(map(lambda x: re.sub("\xa0", "", x), prices))
prices = list(map(lambda x: re.sub("\n(.*)Slutpris ", "", x), prices))
prices = list(map(lambda x: re.sub(" kr\n", "", x), prices))
prices

size = tree.xpath('//div[@class="sold-property-listing__subheading sold-property-listing--left"]/text()')
size
