import json
import math
import os

def split_json(input_file, num_parts=10):
    # Load the JSON data with UTF-8 encoding
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate the size of each part
    chunk_size = math.ceil(len(data) / num_parts)

    # Get the base name of the file (without extension)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Split the data into parts and write to separate files
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        part_data = data[start_idx:end_idx]
        
        # Create the output filename
        output_filename = f"{base_name}_{i+1}.json"
        
        # Write the part data to the file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(part_data, f, indent=4, ensure_ascii=False)

# Example usage:
split_json('split_10.json')
