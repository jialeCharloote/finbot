import torch
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline, BitsAndBytesConfig, EarlyStoppingCallback
from concurrent.futures import ProcessPoolExecutor
from transformers import MarianMTModel, MarianTokenizer
from peft import LoraConfig, get_peft_model

import os
import gc
from sklearn.utils import resample
import numpy as np
import pandas as pd
import json
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict


## Step1 Dataset collect From API
github_url = "https://raw.githubusercontent.com/jialeCharloote/econ_chatbot/main/Dataset/stock_tickers.txt"

# GitHub raw URL
url = "https://raw.githubusercontent.com/jialeCharloote/econ_chatbot/main/Dataset/stock_tickers.txt"

tickers = pd.read_csv(url, header=None)[0].tolist()
print(len(tickers))
API_KEY = os.getenv("NEWS_KEY")


news_list=[]
for i in range (len(tickers)):
    url = f"https://api.polygon.io/v2/reference/news?ticker={tickers[i]}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("results", [])
        if isinstance(articles, list):
            news_list+=articles
        if isinstance(articles, dict):
            news_list.append(articles)
    else:
        continue
        #print("Error fetching data:", response.status_code)



# remove duplicated articles based on url 
finan_news = (pd.DataFrame(news_list).drop_duplicates(subset=['article_url'])).drop(columns=['amp_url'])
finan_news = finan_news.dropna(subset=['insights'])
print(f'Data length is {len(finan_news)}')

file_path = "/kaggle/working/finan_news.json"

with open(file_path, "w") as f:
    json.dump(finan_news.to_dict(orient='records'), f)

print(f"JSON saved to {file_path}")

file_path = "/kaggle/input/financial-news-with-ticker-level-sentiment/polygon_news_sample.json"

with open(file_path, "r") as f:
    loaded_data = json.load(f)

news_sample = pd.DataFrame(loaded_data).drop(columns=['amp_url'])
print(len(news_sample))

merge_df = pd.concat([finan_news, news_sample], ignore_index=True)
print(f'Data length is {len(merge_df)}')

file_path = "/kaggle/working/merged_news.json"

with open(file_path, "w") as f:
    json.dump(merge_df.to_dict(orient='records'), f)

print(f"JSON saved to {file_path}")



## Step2 Preprocess merged dataset

#--- load data from github----
github_url = "https://raw.githubusercontent.com/jialeCharloote/econ_chatbot/main/Dataset/merged_news.json"

json_filename = "merged_news.json"
response = requests.get(github_url)

if response.status_code == 200:
    with open(json_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"File '{json_filename}' downloaded successfully.")
else:
    print("Failed to download the file. Check the URL or permissions.")

with open(json_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

news_data =  Dataset.from_list(data)
news_data = news_data.remove_columns(['id', 'publisher','author','published_utc', 'article_url','image_url', 'keywords' ])
del response, data
news_data



#--- change the format to one-to-one: one news one ticker----
processed_data = []

for news in news_data:
    title = news['title']
    description = news['description']
    
    for insight in news['insights']:
        ticker = insight['ticker']
        sentiment = insight['sentiment']
        sentiment_reasoning = insight['sentiment_reasoning']
        
        processed_data.append({
            "title": title,
            "ticker": ticker,
            "description": description,
            "sentiment": sentiment,
            "sentiment_reasoning": sentiment_reasoning
        })
del news, title, insight, ticker, sentiment, sentiment_reasoning, news_data
processed_data = Dataset.from_list(processed_data)
processed_data




#--- remove undetermined sentiment label----
df = pd.DataFrame(processed_data)
print(f'Original sentiment distribution is')
print(df['sentiment'].value_counts())

remove_sentiments = ['mixed', 'NA', 'neutral/positive']
df_filtered = df[~df['sentiment'].isin(remove_sentiments)]

print(f'Filtered sentiment distribution is')
print(df_filtered['sentiment'].value_counts())

del df, remove_sentiments, processed_data

