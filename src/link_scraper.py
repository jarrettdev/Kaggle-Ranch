# Script to fetch dataset links from Kaggle
#%%
import pandas as pd
import requests
import zipfile
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
# %%
kaggle_links = []
with open('kaggle.txt') as f:
    kaggle_links = list(set(f.readlines()))
# %%
kaggle_links
# %%
kaggle_links = [link for link in kaggle_links if 'kaggle.com/datasets/' in link]
kaggle_links = [link.replace('\n', '') for link in kaggle_links]
# %%
kaggle_links
random.shuffle(kaggle_links)
#%%

# %%
dataset_links = []
for link in tqdm(kaggle_links[-100:]):
    res = requests.get(link)
    content = BeautifulSoup(res.text, 'lxml')
    try:
        dataset_link = str(content).split('contentUrl":"')[1].split('"')[0]
    except:
        continue
    dataset_links.append(dataset_link)
# %%
dataset_links
# %%
link = kaggle_links[0]
res = requests.get(link)
content = BeautifulSoup(res.text, 'lxml')
# %%
str(content)
# %%
str(content).split('contentUrl":"')[1]
# %%
content
# %%
'contentUrl' in str(content)
# %%
link
# %%
#!pip install wget
#%%
#import wget
# %%
dataset_link = dataset_links[0]
# %%
#wget.download(dataset_link)
# %%
request_url = 'https://www.kaggle.com/api/i/datasets.DatasetService/GetTopicalDatasets'
#%%
headers = {}

#%%
payload = {"topicType":"TRENDING_DATASET","sortBy":"PUBLISHED","offset":"24","count":"24"}
#%%
headers['content-length'] = '72'
headers['accept'] = 'application/json'
headers['origin'] = 'https://www.kaggle.com'
headers['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
headers['referrer'] = 'https://www.kaggle.com/datasets?topic=trendingDataset'
#headers['cookie'] = 'ka_sessionid=0205a449c1a57e2eb01ab206cb440aed; __Host-KAGGLEID=CfDJ8CpxVlrQp8xAogku-5T5qgjU7lyGiPzbTqxFi5IcuyI7CC-KWYPxHIdR5hcMI0JPwdqglSjXd1M8OXxrmKGUXOj72_tAcVGyhb5uHs450c0M9aOa-wQ9naLG; CSRF-TOKEN=CfDJ8CpxVlrQp8xAogku-5T5qgjbyxpQw6TgZ8biznGdluuHQZHIEX6o25NjuiTgtnwnQhpVJWZ2__Swryi_1Szfeh8xbv0WgI_nU0nJYs58yQ; XSRF-TOKEN=CfDJ8CpxVlrQp8xAogku-5T5qgj4Bfm6rAhY7XsLIVWQc2UzMzu1dkfZr6NCP1xjSAx5omncqvtBjBjFGsFj0T_Sdl8wCL4mxM2bNBydvLQzYd-f1RwFgqBnGCJsAsCpBfyZ8F-m0JGwjUOd1EDpKTwWKJQ; CLIENT-TOKEN=eyJhbGciOiJub25lIiw…wiRnJvbnRlbmRFcnJvclJlcG9ydGluZ1NhbXBsZVJhdGUiOiIwLjAxIiwiRW1lcmdlbmN5QWxlcnRCYW5uZXIiOiJ7IH0iLCJDbGllbnRScGNSYXRlTGltaXQiOiI0MCIsIkZlYXR1cmVkQ29tbXVuaXR5Q29tcGV0aXRpb25zIjoiMzUzMjUsMzcxNzQsMzM1NzksMzc4OTgsMzczNTQsMzc5NTkiLCJBZGRGZWF0dXJlRmxhZ3NUb1BhZ2VMb2FkVGFnIjoiZGF0YXNldHNNYXRlcmlhbERldGFpbCJ9LCJwaWQiOiJrYWdnbGUtMTYxNjA3Iiwic3ZjIjoid2ViLWZlLWNhbmFyeSIsInNkYWsiOiJBSXphU3lBNGVOcVVkUlJza0pzQ1pXVnotcUw2NTVYYTVKRU1yZUUiLCJibGQiOiJjNzA3OTk1YTRiZDdkYmI1NjNhY2Y5YjQwY2EzOTQxMDUxNjZhZTczIn0.; GCLB=CM3t4N-xkPvmeQ'.encode()
#%%

res = requests.post(dataset_link, headers=headers, data=payload)
# %%
res
# %%
res.text
# %%
res.content
# %%
#for every link in dataset_links open the link in the browser
# %%
import webbrowser
for link in dataset_links:
    webbrowser.open(link)

# %%
