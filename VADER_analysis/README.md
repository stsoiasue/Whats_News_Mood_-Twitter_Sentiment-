
Observed Trends

- None of the news sources are overwhelmingly positive or negative based on the Vader scale (-1 to 1)
- CBS's 100 tweets are more positive than the other news orgs.
- The Nytimes's 100 tweets are more negative than the other news orgs.


```python
# import dependencies
import pandas as pd
import seaborn as sns
import numpy as np
import tweepy
import matplotlib.pyplot as plt
from datetime import datetime
from api_keys import Twitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# estalish twitter API keys
consumer_key = Twitter.consumer_key
consumer_secret = Twitter.consumer_secret
access_token = Twitter.access_token
access_token_secret = Twitter.access_token_secret
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# twitter accounts to search
target_accounts = ('@BBC', '@CBS', '@CNN', '@FoxNews', '@nytimes')
```


```python
# analyze recent 100 tweets from news accounts
# create list to hold sentiments
sentiments = []

for account in target_accounts:
    
    # gather 100 recent tweets from news accounts
    public_tweets = api.user_timeline(id=account, count =100)
    
    # set tweet counter
    tweet_number = 100
    
    # loop through tweets
    for tweet in public_tweets:
        
        # retrive timestamp of tweet
        timestamp = tweet['created_at']
        converted_timestamp = datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y').date()
        
        # retrive tweet text
        text = tweet['text']
        
        # analyze tweet
        scores = analyzer.polarity_scores(text)
        
        # add news account to scores dictionary
        scores['News_Source'] = account
        
        # add date to scores dictionary
        scores['Date'] = converted_timestamp
        
        # add tweet Number to scores dictionary
        scores['Tweet_Number'] = tweet_number
        tweet_number -= 1
        
        # add scores dictionary to sentiments list
        sentiments.append(scores)

# create dataframe with tweet data
news_sentiment_df = pd.DataFrame(sentiments)
news_sentiment_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>News_Source</th>
      <th>Tweet_Number</th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-12-08</td>
      <td>@BBC</td>
      <td>100</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-12-08</td>
      <td>@BBC</td>
      <td>99</td>
      <td>0.1027</td>
      <td>0.104</td>
      <td>0.769</td>
      <td>0.126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-12-08</td>
      <td>@BBC</td>
      <td>98</td>
      <td>0.5719</td>
      <td>0.000</td>
      <td>0.844</td>
      <td>0.156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-12-08</td>
      <td>@BBC</td>
      <td>97</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-12-08</td>
      <td>@BBC</td>
      <td>96</td>
      <td>-0.3182</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save data to csv
news_sentiment_df.to_csv('news_sentiments.csv')
```


```python
# pivot dataframe for plotting
pivoted_df = news_sentiment_df.pivot(index='Tweet_Number', columns='News_Source', values='compound')
pivoted_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>News_Source</th>
      <th>@BBC</th>
      <th>@CBS</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
    </tr>
    <tr>
      <th>Tweet_Number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.7713</td>
      <td>0.5994</td>
      <td>0.0000</td>
      <td>0.8360</td>
      <td>-0.6705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3400</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0258</td>
      <td>-0.8126</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6588</td>
      <td>0.0000</td>
      <td>-0.6214</td>
      <td>0.0000</td>
      <td>0.2500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.6486</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2235</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>0.4939</td>
      <td>0.0000</td>
      <td>0.4927</td>
      <td>-0.6486</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot compound scores on scatter plot

x_values = np.arange(100)

fig = plt.figure(figsize=(10, 7))

for news_source in target_accounts:
    
    plt.scatter(x_values, pivoted_df[news_source])

# add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.13, 1))

# change fontsize of x and y ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add title and axis labels
plt.title('VADER Sentiment Analysis of Popular News Account Tweets', fontsize=18)
plt.xlabel('Tweets', fontsize=15)
plt.ylabel('Compound Score of Tweet\n(1=positive, -1=negative)', fontsize=15)

# save plot
plt.savefig('sentiment_analysis_scatter.png')

plt.show()
```


![png](png_files/sentiment_analysis_scatter.png)



```python
# group by News_Source
grouped_df = news_sentiment_df.groupby('News_Source')

# calculate mean on compound scores
grouped_compound = grouped_df['compound'].mean()

grouped_compound
```




    News_Source
    @BBC        0.130120
    @CBS        0.370750
    @CNN       -0.010047
    @FoxNews    0.043585
    @nytimes   -0.143070
    Name: compound, dtype: float64




```python
# plot mean compound scores
x_values = np.arange(len(target_accounts))

plot_data = zip(x_values, target_accounts)

fig = plt.figure(figsize=(10, 7))

for x, news_source in plot_data:
    
    y = grouped_compound[news_source]
    
    plt.bar(x, y)
    
    plt.text(x, y/2, '{:.2}'.format(y),
             horizontalalignment='center', color='black',
             fontsize=13, weight='bold')
    
# change xticks to news sources
plt.xticks(x_values, target_accounts)

# change fontsize of x and y ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add title and axis labels
plt.title('Average Tweet Sentiment of Popular News Account', fontsize=18)
plt.xlabel('News Twitter Account', fontsize=15)
plt.ylabel('Average Compound Score\n(1=positive, -1=negative)', fontsize=15)

# save plot
plt.savefig('average_sentiment_bar.png')

plt.show()
```


![png](png_files/average_sentiment_bar.png)

