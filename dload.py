import requests

url = 'https://raw.githubusercontent.com/rapha18th/reddit-ManUtd/master/reddit.csv'
res = requests.get(url, allow_redirects=True)
with open('reddit.csv', 'wb') as file:
    file.write(res.content)

url = 'https://raw.githubusercontent.com/rapha18th/reddit-ManUtd/master/reddit_subset.csv'
res = requests.get(url, allow_redirects=True)
with open('reddit_subset.csv', 'wb') as file:
    file.write(res.content)
