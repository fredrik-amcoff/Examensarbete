import openai
import wikipediaapi
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def count_words(text):
    words = text.split()
    return len(words)


def generate_text(title, length, key):
    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write a text about {title} in around {length} words."}],
    )
    text_response = response.choices[0].message.content
    return text_response


def get_wikipedia_text(title, user_agent, num_paragraphs=3):
    wiki = wikipediaapi.Wikipedia(language="en", user_agent=user_agent)
    page = wiki.page(title)
    if not page.exists():
        print(f"{title} not found")

    paragraphs = page.text.split("\n")
    paragraphs = [p for p in paragraphs if p.strip()] # remove empty lists
    output_text = "\n\n".join(paragraphs[:num_paragraphs])
    words = output_text.split()
    return output_text, len(words)


def store_data(titles, key, user_agent, num_paragraphs=3):
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


def generate_pair_plots():
    file_path = "text_statistics.csv"
    df = pd.read_csv(file_path)
    numerical_columns = [col for col in df.columns if col != "ai"]
    sns.set_theme(style="whitegrid")

    # Create pairplot with category-based coloring
    g = sns.pairplot(df, vars=numerical_columns, hue="ai", palette={0: "blue", 1: "red"}, plot_kws={'alpha': 0.6})
    output_path = 'preliminary_results.png'
    g.savefig(output_path, dpi=300)
    plt.show()


# for user: contact information of the format "<Application Name>/<Version> (<Description>, <Contact Information>)",
# example: "ReasearchProject/1.0 (University of Uppsala, contact: email.example@student.uu.se)"
user = ""
token = ""  # OpenAI API key

wikipedia_topics = [
    "Quantum mechanics",
    "Artificial intelligence",
    "Theory of relativity",
    "CRISPR",
    "Black hole",
    "Nanotechnology",
    "Renewable energy",
    "Machine learning",
    "History of the Internet",
    "SpaceX",
    "Fall of the Western Roman Empire",
    "Cold War",
    "French Revolution",
    "Nelson Mandela",
    "Civil rights movement",
    "History of democracy",
    "Industrial Revolution",
    "Cuban Missile Crisis",
    "World War I",
    "Mesopotamia",
    "Great Wall of China",
    "Amazon rainforest",
    "Sahara Desert",
    "Mariana Trench",
    "Mount Everest",
    "Atlantis",
    "Aurora (astronomy)",
    "Great Barrier Reef",
    "Wonders of the World",
    "Climate change",
    "William Shakespeare",
    "Mona Lisa",
    "Science fiction",
    "Art and politics",
    "Greek mythology",
    "Vincent van Gogh",
    "Graphic novel",
    "Anime",
    "Surrealism",
    "Digital art",
    "Existentialism",
    "Stoicism",
    "Buddhism",
    "Ethics of artificial intelligence",
    "Religion in ancient Rome",
    "Free will",
    "Confucianism",
    "Vatican City",
    "Meaning of life",
    "Atheism",
    "Discovery of penicillin",
    "Vaccine",
    "Human microbiome",
    "Sleep",
    "Stress (biology)",
    "Nutrition and mental health",
    "Anesthesia",
    "Physical exercise",
    "Gut microbiota",
    "Black Death",
    "Olympic Games",
    "Chess",
    "Formula One",
    "Sports psychology",
    "Esports",
    "FIFA World Cup",
    "Martial arts",
    "Tour de France",
    "Sports rivalry",
    "History of basketball",
    "Rock and roll",
    "Hip hop music",
    "The Beatles",
    "Jazz",
    "Animation",
    "Golden Age of Hollywood",
    "Music streaming service",
    "K-pop",
    "Video game console",
    "Horror film",
    "Dot-com bubble",
    "Cryptocurrency",
    "Stock market",
    "Central bank",
    "Gig economy",
    "Economic impact of the COVID-19 pandemic",
    "Globalization",
    "Universal basic income",
    "Silicon Valley",
    "Space economy",
    "History of timekeeping devices",
    "Conspiracy theory",
    "Memory",
    "April Fools' Day",
    "Déjà vu",
    "Dark matter",
    "Playing card",
    "List of languages by number of speakers",
    "Humanoid robot",
    "Luck"
]

store_data(wikipedia_topics, token, user)