import re
import json
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MarianMT model and tokenizer for English to Swedish translation
model_name = 'Helsinki-NLP/opus-mt-en-sv'
model = MarianMTModel.from_pretrained(model_name).to(device)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Function to clean the text before translation
def clean_text(text):
    text = re.sub(r"\[\d+\]|\(\d{4}\)", "", text)  # Remove references
    text = re.sub(r"[^a-zA-Z0-9\s.,!?()'\"-]", "", text)  # Remove unwanted characters
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text.strip()

# Function to translate cleaned text
def translate(text):
    cleaned_text = clean_text(text)
    tokens = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**tokens)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Function to process the JSON file and translate its content
def translate_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    translated_data = []
    
    for item in tqdm(data, desc="Translating", unit="entry"):
        translated_item = item.copy()
        translated_item['wikipedia_text_sv'] = translate(item['wikipedia_text'])
        translated_item['ai_text_sv'] = translate(item['ai_text'])
        translated_data.append(translated_item)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(translated_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"Translation completed. Results saved to {output_file}.")

# Run translation
input_file = 'text_data.json'
output_file = 'translated_text_data.json'
translate_json(input_file, output_file)
