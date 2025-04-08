import json
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MarianMT model and tokenizer for English to Swedish translation
model_name = 'Helsinki-NLP/opus-mt-en-sv'
model = MarianMTModel.from_pretrained(model_name).to(device)  # Move model to GPU if available
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Function to translate text from English to Swedish
def translate(text):
    # Tokenize the input text and move tensors to the device (GPU/CPU)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Perform translation
    translated_tokens = model.generate(**tokens)
    
    # Decode the translated tokens back into text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Function to process the JSON file and translate its content
def translate_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    translated_data = []
    
    # Iterate through each entry in the JSON and translate the text fields
    for item in tqdm(data, desc="Translating", unit="entry"):
        translated_item = item.copy()  # Copy the original item to keep other fields intact
        
        # Translate the 'wikipedia_text' and 'ai_text' fields
        translated_item['wikipedia_text_sv'] = translate(item['wikipedia_text'])
        translated_item['ai_text_sv'] = translate(item['ai_text'])
        
        # Add the translated item to the list
        translated_data.append(translated_item)
    
    # Save the translated data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(translated_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"Translation completed. Results saved to {output_file}.")

# Example usage
input_file = 'text_data.json'  # JSON file containing the text to translate
output_file = 'translated_text_data.json'  # File to save translated Swedish text
translate_json(input_file, output_file)
