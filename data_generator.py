import requests
import openai
import wikipediaapi
import json
import time
import random
from bs4 import BeautifulSoup
from string import ascii_uppercase
import pandas as pd
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import re
from collections import Counter


def count_words(text):
    words = text.split()
    return len(words)


def generate_text(title, length, key):
    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f'Write a text about the topic of "{title}" in around {length} words.'}],
        temperature=1,
        seed=587078
    )
    text_response = response.choices[0].message.content
    return text_response


def get_wikipedia_text(title, user_agent, num_paragraphs=3, language='en'):
    wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)
    page = wiki.page(title)
    if not page.exists():
        print(f"{title} not found")

    paragraphs = page.text.split("\n")
    paragraphs = [p for p in paragraphs if p.strip()] # remove empty lists
    output_text = "\n\n".join(paragraphs[:num_paragraphs])
    words = output_text.split()
    return output_text, len(words)


def store_data_from_list(titles, key, user_agent, num_paragraphs=3):
    data = []
    times = []
    for title in titles:
        start = time.time()
        print(title)
        wiki_output = get_wikipedia_text(title, user_agent, num_paragraphs)
        wikipedia_text = wiki_output[0]
        ai_text = generate_text(title, wiki_output[1], key)
        data.append({
            "title": title,
            "wikipedia_text": wikipedia_text,
            "ai_text": ai_text,
            "wiki_characters": len(wikipedia_text),
            "wiki_words": count_words(wikipedia_text),
            "ai_characters": len(ai_text),
            "ai_words": count_words(ai_text)
        })
        stop = time.time()
        print(stop - start)
        times.append(stop - start)
    with open("text_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(sum(times))
    print(times)


def store_data_from_random(num_articles, key, user_agent, num_paragraphs=3):
    data = []
    article_titles = []
    with open('vital_articles_1.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                article_titles.append({'title': row[0].strip(), 'topic': row[1].strip(), 'section': row[2].strip()})

    selected_articles = set()
    attempts = 0
    max_attempts = len(article_titles)

    while len(selected_articles) < num_articles and attempts < max_attempts:
        random_article = random.choice(article_titles)
        # Make sure article not already in list
        if random_article['title'] in selected_articles:
            attempts += 1
            continue

        # Break while loop if no more articles are available
        if len(article_titles) == 0:
            warnings.warn(f'No more articles to choose from, storing {len(selected_articles)} articles.')
            break

        wiki_output = get_wikipedia_text(random_article['title'], user_agent=user_agent, num_paragraphs=num_paragraphs)
        wikipedia_text = wiki_output[0]
        wikipedia_len = wiki_output[1]
        print(f"\nAttempt {attempts + 1}/{max_attempts}:")

        if 200 <= wikipedia_len <= 500:
            selected_articles.add(random_article['title'])
            ai_text = generate_text(random_article['title'], wikipedia_len, key)
            data.append({
                "title": random_article['title'],
                "topic": random_article['topic'],
                "section": random_article['section'],
                "wikipedia_text": wikipedia_text,
                "ai_text": ai_text,
                "wiki_characters": len(wikipedia_text),
                "wiki_words": count_words(wikipedia_text),
                "ai_characters": len(ai_text),
                "ai_words": count_words(ai_text)
            })
            article_titles.remove(random_article)
            print(f"\t'{random_article['title']}' ({wikipedia_len} words) added")
        else:
            print(f"\t'{random_article['title']}' ({wikipedia_len} words) not added")
            article_titles.remove(random_article)
        attempts += 1
        print(f"{len(selected_articles)}/{num_articles} articles selected")
    with open("text_data_rnd.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def scrape_vital():
    """
    Scrapes all articles form the Wikipedia vital articles, sorts them and stores them in a csv.
    :return: None
    """
    data = []
    for letter in ascii_uppercase:
        url = f"https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/data/{letter}.json"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the main table containing the data
        outer_table = soup.find('tbody')
        # Loop only through direct <tr> children of the outer table
        for row in outer_table.find_all('tr', recursive=False):
            # Find the article title
            title_span = row.find('th')
            if title_span and title_span.find('span'):
                title = title_span.find('span').text.strip()
                print(title)
            else:
                continue  # Skip rows without a valid title
            # Initialize default values
            topic = None
            section = None
            level = None

            # Find the nested <table class="mw-json">
            json_table = row.find('table', class_='mw-json')
            if json_table:
                # Inner row contains topic and section
                for inner_row in json_table.find_all('tr'):
                    key_span = inner_row.find('th')
                    value_td = inner_row.find('td', class_='mw-json-value')
                    if key_span and value_td:
                        key = key_span.text.strip()
                        value = value_td.text.strip().strip('"')  # Remove extra quotes
                        if key == "topic":
                            topic = value
                        elif key == "section":
                            section = value
                        elif key == "level":
                            level = int(value)

            # Append the data
            data.append({
                'Title': title,
                'Topic': topic,
                'Section': section,
                'Level': level
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by='Title').reset_index(drop=True)
    df.to_csv('vital_articles.csv', index=False)


def sample_articles(num_articles, user_agent, num_paragraphs=3):
    article_titles = []
    articles = []
    with open('Data/vital_articles.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                article_titles.append(row[0].strip())

    selected_articles = set()
    attempts = 0
    max_attempts = 100000
    while len(selected_articles) < num_articles and attempts < max_attempts:
        random_article = random.choice(article_titles)
        # Make sure article not already in list
        if random_article in selected_articles:
            attempts += 1
            continue

        article_data = get_wikipedia_text(random_article, user_agent=user_agent, num_paragraphs=num_paragraphs)
        if 200 <= article_data[1] <= 500:
            selected_articles.add(random_article)
            print(random_article)
            articles.append(article_data)
        attempts += 1
    return articles



# for user: contact information of the format "<Application Name>/<Version> (<Description>, <Contact Information>)",
# example: "ReasearchProject/1.0 (University of Uppsala, contact: email.example@student.uu.se)"
user = ""
token = ""

store_data_from_random(5000, token, user)