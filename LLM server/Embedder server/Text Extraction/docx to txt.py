import pypandoc
import json
import re


def extract_sections_from_docx(input_file, output_json):
    try:
        # Convert DOCX to Markdown to preserve headings structure
        markdown_text = pypandoc.convert_file(input_file, 'markdown')

        # Parse sections and subsections using regex
        sections = {}
        current_section = None
        current_subsection = None

        for line in markdown_text.split("\n"):
            line = line.strip()

            # Detect main section (e.g., "# Section 1")
            if re.match(r"^#\s+(.+)", line):
                current_section = re.match(r"^#\s+(.+)", line).group(1)
                sections[current_section] = {}
                current_subsection = None  # Reset subsection

            # Detect subsection (e.g., "## Subsection 1.1")
            elif re.match(r"^##\s+(.+)", line):
                current_subsection = re.match(r"^##\s+(.+)", line).group(1)
                sections[current_section][current_subsection] = []

            # Capture paragraph text inside a section/subsection
            elif line:
                if current_section and current_subsection:
                    sections[current_section][current_subsection].append(line)
                elif current_section:
                    sections[current_section].setdefault("text", []).append(line)

        # Save to JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(sections, f, ensure_ascii=False, indent=4)

        print(f"✅ JSON extraction successful! File saved as: {output_json}")

    except Exception as e:
        print(f"❌ Error: {e}")


input_file = "./Testing Docx/22104-i30.docx"

# Json & Text Extraction
# Write to output file
output_json = "./Results/22104-i30.json"
output_txt = "./Results/22104-i30.txt"
extract_sections_from_docx(input_file, output_json)

# Markdown Extraction
output_markdown = "./Results/22104-i30.md"
markdown_text = pypandoc.convert_file(input_file, 'markdown')
with open(output_markdown, 'w', encoding='utf-8') as f:
    f.write(markdown_text)

