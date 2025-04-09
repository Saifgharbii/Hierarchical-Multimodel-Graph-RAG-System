import time

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.paragraph import Run
import os
import re
from typing import Dict, List, Any
import json

from docx.oxml.ns import nsmap as default_nsmap
from lxml import etree
import lxml

# Create a local copy of the namespace mapping and add the "v" namespace if not present.
custom_nsmap = default_nsmap.copy()
custom_nsmap["v"] = "urn:schemas-microsoft-com:vml"
custom_nsmap["wp"] = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
custom_nsmap["a"] = "http://schemas.openxmlformats.org/drawingml/2006/main"
def custom_xpath(element: CT_P, xpath_str: str) -> List[lxml.etree._Element]:
    # Use lxml.etree.XPath with our custom namespace mapping.
    return etree.XPath(xpath_str, namespaces=custom_nsmap)(element)


def iter_block_items(parent: DocxDocument) -> list:
    if isinstance(parent, DocxDocument):
        parent_elm = parent._element.body
    else:
        parent_elm = parent
    parts = []
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            parts.append(Paragraph(child, parent))
        elif isinstance(child, CT_Tbl):
            parts.append(Table(child, parent))
    return parts


def run_has_image(run: Run):
    has_inline = custom_xpath(run._element,'.//wp:inline')
    has_anchor = custom_xpath(run._element,'.//wp:anchor')
    has_pict = custom_xpath(run._element,'.//w:pict')
    has_shape = custom_xpath(run._element,'.//w:object//v:imagedata')
    if has_shape :
        print("the type of has_shape is ", type(has_shape[0]))
    return bool(has_inline or has_anchor or has_pict or has_shape)


def extract_image_from_run(run: Run, doc: Document) -> tuple[bytes, str]:
    """Extracts an image from a Run object and returns its binary data and extension."""
    r_id = None
    # Check for inline or anchored images
    blip_elements = custom_xpath(run._element,'.//a:blip')
    if blip_elements:
        r_id = blip_elements[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")

    # Check for VML-based images (WMF, EMF, BMP, etc.)
    imagedata_elements = custom_xpath(run._element,'.//v:imagedata')
    if imagedata_elements and not r_id:
        r_id = imagedata_elements[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")

    if r_id:
        try:
            image_part = doc.part.rels[r_id].target_part
            image_bytes = image_part.blob
            image_ext = os.path.splitext(image_part.partname)[-1].lower()  # Get extension (e.g., .png, .jpg)
            return image_bytes, image_ext
        except KeyError:
            print(f"Warning: Relationship ID {r_id} not found.")
            return None, None
    return None, None


def sanitize_filename(name: str) -> str:
    """Sanitize figure name to create a safe filename."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name.strip().replace(" ", "_")[:50]  # limit to 50 chars


def preprocess_figure(run: Run, figure_name: str, doc: Document) -> str:
    """Extracts the image from the run, saves it, and (optionally) sends it to Vision LM."""
    image_bytes, image_ext = extract_image_from_run(run, doc)

    if image_bytes:
        # Create 'testing_figures' directory in the program's root if it doesn't exist
        save_dir = os.path.join(os.getcwd(), "testing_figures")
        os.makedirs(save_dir, exist_ok=True)

        # Sanitize the figure name to create a valid filename
        filename = f"{sanitize_filename(figure_name)}{image_ext}"
        save_path = os.path.join(save_dir, filename)

        # Save image
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        print(f"[Image saved: {save_path}]")
        # Placeholder for Vision LM call
        print(f"[Extracted image insights from Vision LM: {image_ext} image]")
        return f"[Figure saved: {filename}]"

    print("[Image not found]")
    return "[Image not found]"

def extract_text(runs: list[Run]) -> str:
    figure_description_list = []
    for para in runs:
        figure_description_list.append(para.text)
    return ' '.join(figure_description_list).strip()


def process_paragraph(paragraph: Paragraph, next_paragraphs: list[Paragraph], doc: Document) -> tuple[str, int]:
    para_text = []
    num = 0
    for idx, run in enumerate(paragraph.runs):
        if run_has_image(run):
            num += 1
            if next_paragraphs:
                figure_name = "This figure has no name"
                for i in range(min(3, len(next_paragraphs)-1)):
                    if isinstance(next_paragraphs[i], Paragraph):
                        # checks if the next paragraph is Paragraph or Table
                        # in some cases we can make an appendix for figures
                        try:
                            if next_paragraphs[i].runs[0].text.lower().strip().startswith(("fig", "note")):
                                # checks if the next paragraph is the real caption of the figure or another random text
                                figure_name = extract_text(next_paragraphs[0].runs)
                                print("this description of the figure :", figure_name)
                                break
                        except:
                            pass

                    elif i == 0:
                        # The figure has an appendix to be extracted
                        # This appendix should be saved in the database and extracted to be related to the following
                        # figure
                        print("there we have an appendix")
                        pass
            else:
                figure_name = "This figure has no name"
            para_text.append(preprocess_figure(run, figure_name, doc))
        else:
            para_text.append(run.text)
    return ''.join(para_text).strip(), num


def extract_docx_structure(file_path: str) -> Dict[str, Any]:
    document_json = {
        "document_name": os.path.basename(file_path),
        "content": []
    }
    number_figures = 0
    doc = Document(file_path)
    current_section : dict[str:Any] = None
    current_subsection : dict[str:Any] = None
    current_subsubsection : dict[str:Any] = None
    last_processed_paragraph : str or None = None
    parts = iter_block_items(doc)
    for idx, element in enumerate(parts):
        if isinstance(element, Paragraph):
            paragraph: Paragraph = element
            if paragraph.style.name.startswith('Heading'):
                try:
                    heading_level = int(paragraph.style.name.split()[1])
                except:
                    continue

                if heading_level == 1:
                    current_section = {
                        "title": paragraph.text.strip(),
                        "description": "",
                        "summary": "",
                        "tables": [],
                        "subsections": []
                    }
                    document_json["content"].append(current_section)
                    current_subsection = None
                    current_subsubsection = None
                elif heading_level == 2:
                    current_subsection = {
                        "title": paragraph.text.strip(),
                        "description": "",
                        "summary": "",
                        "text_content": "",
                        "tables": [],
                        "subsubsections": []
                    }
                    if current_section:
                        current_section["subsections"].append(current_subsection)
                    current_subsubsection = None
                elif heading_level == 3:
                    current_subsubsection = {
                        "title": paragraph.text.strip(),
                        "text_content": "",
                        "tables": []
                    }
                    if current_subsection:
                        current_subsection["subsubsections"].append(current_subsubsection)
                last_processed_paragraph = None
            else:
                try:
                    next_paragraphs = parts[idx + 1:]
                except:
                    next_paragraphs = None
                processed_text, num = process_paragraph(paragraph, next_paragraphs, doc)
                number_figures += num
                if current_subsubsection:
                    current_subsubsection["text_content"] += processed_text + "\n"
                elif current_subsection:
                    current_subsection["text_content"] += processed_text + "\n"
                elif current_section:
                    current_section["description"] += processed_text + "\n"
                last_processed_paragraph = paragraph

        elif isinstance(element, Table):
            table_description = last_processed_paragraph.text.strip() if last_processed_paragraph else ""
            table_entry = {
                "description": table_description,
                "summary": "",
                "name (in the NoSql database)": ""
            }

            if current_subsubsection:
                current_subsubsection["tables"].append(table_entry)
            elif current_subsection:
                current_subsection["tables"].append(table_entry)
            elif current_section:
                current_section["tables"].append(table_entry)
            last_processed_paragraph = None
    print("number of figures is :", number_figures)
    return document_json


def extract_from_all_files(directory):
    list_dir = os.listdir(directory)
    print("here's the list of directory :", list_dir)
    for idx, filename in enumerate(list_dir):
        if filename.endswith(".docx"):
            t1 = time.time()
            results = extract_docx_structure(os.path.join(directory, filename))
            t2 = time.time()
            print(f"it takes to extract from this file : {filename}  :", t2 - t1, end="\n\n")
            print(f"the {idx} file has been extracted \n\n\n")


# if __name__ == "__main__":
#     directory = "../RAG_dataset"
#     extract_from_all_files(directory)

testing_docx_path = "../RAG_dataset/23003-i40.docx"
doc = Document(testing_docx_path)
t1 = time.time()
results = extract_docx_structure(testing_docx_path)
t2 = time.time()
print(f"it takes to extract from this file : ", t2 - t1)

with open('./Text Extraction/Results/22104-i30-testing.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

"""
for item in iter_block_items(doc):
    if isinstance(item, Paragraph):
        print("The style name is :",item.style.name)
        print("Paragraph:", item.text, end="\n\n")
    elif isinstance(item, Table):
        print("Table detected",end="\n\n")        
"""
