from bs4 import BeautifulSoup
import glob
import os

input_folder = "html"
output_file = "txt/numbered_chunks.txt"

all_chunks = []

# Loop through all HTML files
for html_file in glob.glob(os.path.join(input_folder, "*.html")):
    with open(html_file, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        # Find all paragraphs with an id
        for p in soup.find_all("p"):
            pid = p.get("id")
            text = p.get_text().strip()
            if pid and text:
                chunk = f"[{pid}] {text}"
                all_chunks.append(chunk)

# Save all chunks to a text file
with open(output_file, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(chunk + "\n")