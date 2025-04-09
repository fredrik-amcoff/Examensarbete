import re
import json
from googletrans import Translator
from tqdm import tqdm

# Initialize the translator
translator = Translator()

# Function to clean the text before translation
def clean_text(text):
    text = re.sub(r"\[\d+\]|\(\d{4}\)", "", text)  # Remove references
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()'\"-]", "", text)  # Remove unwanted characters
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text.strip()

# Function to translate cleaned text
def translate_text(text, src_language='en', dest_language='sv'):
    cleaned = clean_text(text)
    return translator.translate(cleaned, src=src_language, dest=dest_language).text

# Read the JSON data from the file
with open("text_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Create a progress bar and translate each entry in the dataset
for entry in tqdm(data, desc="Translating", unit="entry"):
    entry["wikipedia_text_sv"] = translate_text(entry["wikipedia_text"])
    entry["ai_text_sv"] = translate_text(entry["ai_text"])

# Write the translated data to a new JSON file
with open("translated_text_data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
