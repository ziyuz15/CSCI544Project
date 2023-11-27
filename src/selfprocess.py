import pandas as pd
import os
import re

def preprocess_scientific_text(text):
    # Convert to lowercase (optional based on need)
    text = text.lower()

    # Replace or remove mathematical notations
    text = re.sub(r'@\w+', ' ', text)

    # Replace section and figure references
    text = re.sub(r'\[\w+\]', ' ', text)

    # Remove non-alphabetical characters (customize as needed)
    text = re.sub('[^a-zA-Z0-9,.\s]', ' ', text)

    # Remove extra spaces
    text = ' '.join(text.split())

    return text


print(os.getcwd())
# df = pd.read_parquet("E:\CSCI544\CSCI544\BertSum\src/0000.parquet")

# # Convert DataFrame to a CSV file
# df.to_csv('output_file.csv', index=False)

data = pd.read_csv("output_file.csv", encoding="utf-8", on_bad_lines='skip')

output_dir = 'arxiv_raw_stories'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
skip=[3655,5157,5930,7795,10362]
for i in range(len(data)):
    print(i)
    if i in skip:  # Status update every 100 articles
        print(f'Skip {i} articles.')
        continue
    if i == 10000:
        print(f'finsied {i} articles.')
        break
    # Main content of the paper
    content = data['article'][i].strip()
    
    # Abstract sentences, assuming the abstract is already sentence-split
    abstract_sentences = data['abstract'][i].strip()

    # apply preprocessing
    content=preprocess_scientific_text(content)
    abstract_sentences=preprocess_scientific_text(abstract_sentences)
    abstract_sentences = abstract_sentences.split('. ')
    content=content.split('. ')

    # Construct the .story file content
    story_content = "\n\n".join(content) + "\n\n" + "\n\n@highlight\n\n".join(abstract_sentences)

    # Define the output file path
    output_file_path = os.path.join(output_dir, f"paper_{i}.story")

    
    # Write the content to the .story file
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        out_file.write(story_content)

    

print('Finished processing all articles.')
