import re
import json
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch
import nltk

# Download necessary NLTK resources for sentence tokenization
nltk.download('punkt')

# Check if CUDA is available and set device accordingly
device = torch.device("cuda")

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

#post-process common artifacts
def normalize(text):
    text = re.sub(r"(\.\s)+\.", ".", text)   # Collapse spaced dot sequences to single dot
    text = re.sub(r"\.{2,}", "", text)       # Remove sequences of 2+ dots
    text = re.sub(r"-{2,}", "", text)        # Remove sequences of 2+ hyphens
    return text

# Function to translate a single sentence
def translate_sentence(sentence):
    cleaned_sentence = clean_text(sentence)
    tokens = tokenizer(cleaned_sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**tokens)
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_sentence

# Function to batch translate sentences
def translate_sentences_batch(sentences, batch_size=8):
    translated = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**tokens)
        translated_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translated.extend(translated_batch)
    return translated

# Function to translate an entire paragraph
def translate_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    translated_sentences = translate_sentences_batch(sentences)
    joined = ' '.join(translated_sentences)
    return normalize(joined)

# Function to process JSON file and translate content
def translate_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    translated_data = []

    for item in tqdm(data, desc="Translating", unit="entry"):
        translated_item = item.copy()
        translated_item['wikipedia_text_sv'] = translate_paragraph(item['wikipedia_text'])
        translated_item['ai_text_sv'] = translate_paragraph(item['ai_text'])
        translated_data.append(translated_item)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(translated_data, outfile, ensure_ascii=False, indent=4)

    print(f"Translation completed. Results saved to {output_file}.")

# Run translation
input_file = 'translated_rnd.json'
output_file = 'translated_rnd_NO_DOT.json'
translate_json(input_file, output_file)
