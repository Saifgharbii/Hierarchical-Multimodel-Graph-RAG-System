import time

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.paragraph import Run
from docx.oxml.ns import nsmap as default_nsmap

import os

from typing import Dict, List, Any, Tuple
import json
import requests
import io

from lxml import etree
import lxml

from ImageProcessing import safe_convert_to_png

VISION_PROMPT = """Task: Describe {figure_name} using technical details from its visual elements and the context: 
"{last_paragraph}"
Focus:
- Decode symbols/annotations first
- Explain spatial/data relationships
- Note anomalies if present

Examples:

1. Context: "Figure 5 shows cellular handoff mechanics."
Name: "Figure 5: Inter-cell Handover"
→ "Figure 5 diagrams a UE transitioning between two gNBs using X2 interfaces. Measurement reports (RSRP > -100dBm) 
trigger handoff decisions (hexagonal decision node). Timing constraints (T<sub>TTT</sub> = 2ms) appear as subscript 
annotations. Arrows differentiate successful handovers (solid) vs. dropped connections (red dotted)."

2. Context: "The flowchart outlines fault diagnosis steps."
Name: "Figure 6: Network Troubleshooting Workflow"
→ "Figure 6 begins with a diamond-shaped 'Alarm Triggered?' node branching to packet capture (wireshark icon) or log 
analysis (scroll symbol). Parallel processes use fork/join symbols (thick horizontal bars). Critical severity 
alerts (flame icons) bypass standard escalation paths. Green checkmarks denote resolution endpoints."
"""

VISION_MODEL_NAME = "deepseek-ai/deepseek-vl-7b-chat"
DEVICE = "cpu"  # on kaggel let it cuda

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
    has_inline = custom_xpath(run._element, './/wp:inline')
    has_anchor = custom_xpath(run._element, './/wp:anchor')
    has_pict = custom_xpath(run._element, './/w:pict')
    has_shape = custom_xpath(run._element, './/w:object//v:imagedata')
    return bool(has_inline or has_anchor or has_pict or has_shape)


def extract_image_from_run(run: Run, doc: Document) -> tuple[bytes, str]:
    """Extracts an image from a Run object and returns its binary data and extension."""
    r_id = None
    # Check for inline or anchored images
    blip_elements = custom_xpath(run._element, './/a:blip')
    if blip_elements:
        r_id = blip_elements[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")

    # Check for VML-based images (WMF, EMF, BMP, etc.)
    imagedata_elements = custom_xpath(run._element, './/v:imagedata')
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


# def preprocess_figure(run: Run, figure_name: str, doc: Document, last_paragraph: str = "") -> str:
#     """Extracts the image from the run, saves it, and (optionally) sends it to Vision LM."""
#     image_bytes, image_ext = extract_image_from_run(run, doc)
#
#     if image_bytes:
#         prompt = VISION_PROMPT.format(last_paragraph=last_paragraph, figure_name=figure_name)
#         t1 = time.time()
#         image_bytes, image_ext = safe_convert_to_png(image_bytes, image_ext)
#         try:
#             result = process_image_batch(image_bytes_list=[image_bytes],
#                                          prompts=[prompt],
#                                          model_name=VISION_MODEL_NAME,
#                                          device=DEVICE, batch_size=1)
#         except Exception as e:
#             print(f"exception occurred : {e}")
#             return ""
#         t2 = time.time()
#         print(f"for the figure {figure_name} it takes {t2 - t1} seconds\n and here's the result:\n {result}")
#         return result[0]
#
#     print("[Image not found]")
#     return ""


def extract_text(runs: list[Run]) -> str:
    figure_description_list = []
    for para in runs:
        figure_description_list.append(para.text)
    return ' '.join(figure_description_list).strip()


def process_paragraph(paragraph: Paragraph, next_paragraphs: list[Paragraph], doc: Document, last_paragraph_txt) \
        -> tuple[str, int, list[dict[str:any]]]:
    """
    Processes a document paragraph, extracting text and identifying embedded figures/images.

    For each run in the paragraph:
    - If the run contains an image, extracts the image data and attempts to find its caption
    - Otherwise, collects the regular text content

    Args:
        paragraph: The Paragraph object to be processed
        next_paragraphs: List of subsequent Paragraph objects used to search for figure captions
        doc: The parent Document object containing the paragraph (used for image extraction)

    Returns:
        tuple: A 3-element tuple containing:
            - str: The processed text content of the paragraph with figure placeholders (<<<figure_name>>>)
            - int: The number of figures found in the paragraph
            - list[dict]: Figure metadata dictionaries with keys:
                * "figure_id": Placeholder ID (<<<figure_name>>>)
                * "figure_name": Extracted figure name/caption
                * "last_paragraph": Text content preceding the figure
                * "image_bytes": Binary image data (converted to PNG format)
                * "image_ext": File extension (always 'png' after conversion)

    Notes:
        - Figures are identified by checking runs for embedded images
        - Captions are searched for in the next 3 paragraphs (looking for text starting with "fig" or "note")
        - All images are converted to PNG format for consistency
        - Figure placeholders are inserted into the text where images were found
    """

    para_text = []
    figures_data: list[dict[str:str | str:bytes]] = []
    num = 0
    for idx, run in enumerate(paragraph.runs):
        if run_has_image(run):
            num += 1
            figure_name = ""
            if next_paragraphs:
                for i in range(min(3, len(next_paragraphs) - 1)):
                    if isinstance(next_paragraphs[i], Paragraph):
                        # checks if the next paragraph is Paragraph or Table
                        # in some cases we can make an appendix for figures
                        try:
                            if next_paragraphs[i].runs[0].text.lower().strip().startswith(("fig", "note")):
                                # checks if the next paragraph is the real caption of the figure or another random text
                                figure_name = extract_text(next_paragraphs[0].runs).replace('\xa0', ' ')
                                break
                        except:
                            pass

                    elif i == 0:
                        # The figure has an appendix to be extracted
                        # This appendix should be saved in the database and extracted to be related to the following
                        # figure
                        print("there we have an appendix")
                        pass
            image_bytes, image_ext = extract_image_from_run(run=run, doc=doc)
            image_bytes, image_ext = safe_convert_to_png(image_bytes, image_ext)
            if image_bytes and figure_name != "":
                figure_data_dict: dict[str:str | str:bytes] = {
                    "figure_id": f"<<<{figure_name}>>>",
                    "figure_name": figure_name,
                    "last_paragraph": last_paragraph_txt + ''.join(para_text).strip(),
                    "image_bytes": image_bytes,
                    "image_ext": image_ext
                }
                figures_data.append(figure_data_dict)
                para_text.append(f"<<<{figure_name}>>>")
        else:
            para_text.append(run.text)
    return ''.join(para_text).strip().replace('\xa0', ' '), num, figures_data


def extract_docx_structure(file_path: str) -> tuple[dict[str: Any], list[Any]]:
    """
    Extracts and structures the content of a DOCX file into a hierarchical JSON format,
    while identifying figures and tables for preprocessing.

    Parameters:
        file_path (str): Path to the input DOCX file to be processed.

    Returns:
        tuple[dict, list]: A tuple containing:
            1. document_json (dict): Structured document content with keys:
                - "document_name": Basename of the input file
                - "content": List of sections (each containing):
                    - "title": Section heading
                    - "description"/"text_content": Paragraph content
                    - "tables": List of tables in section
                    - "figures_meta_data": List of figure metadata
                    - "subsections": Nested subsections (same structure, up to 3 levels)
            2. figures_for_preprocessing (list): List of figure metadata dicts with:
                - "figure_id": Placeholder ID (e.g., "<<<figure_name>>>")
                - "figure_name": Extracted figure name/caption
                - "last_paragraph": Text preceding the figure
                - "image_bytes": Binary image data (PNG)
                - "image_ext": Always 'png'
                - Context fields: "section_title", "subsection_title"
                - Placement fields : section_idx, subsection_idx, subsubsection_idx

    Behavior:
        1. Parses DOCX file using python-docx
        2. Creates hierarchical structure based on heading levels (1-3)
        3. Processes:
            - Paragraphs (including figure extraction via process_paragraph)
            - Tables (with running count and description from preceding paragraph)
        4. Outputs structured document and separate figure list for preprocessing

    Example return structure:
        (
            {
                "document_name": "example.docx",
                "content": [
                    {
                        "title": "Introduction",
                        "description": "...",
                        "tables": [...],
                        "figures_meta_data": [...],
                        "subsections": [...]
                    }
                ]
            },
            [
                {
                    "figure_id": "<<<fig1>>>",
                    "figure_name": "System Diagram",
                    "last_paragraph": "As shown in Figure 1...",
                    "image_bytes": b'...',
                    "image_ext": "png",
                    "section_title": "Introduction",
                    "subsection_title": "",
                    "subsubsection_title": "",
                    "section_idx" : 0
                    "subsection_idx" : -1
                    "subsubsection_idx" : -1
                    ...
                }
            ]
        )
    """

    document_json = {
        "document_name": os.path.basename(file_path),
        "content": []
    }
    number_figures = 0
    figures_for_preprocessing: list[dict[str:str | str:bytes]] = []
    table_number = 1
    doc = Document(file_path)
    current_section: dict[str:Any] = None
    current_subsection: dict[str:Any] = None
    current_subsubsection: dict[str:Any] = None
    section_idx, subsection_idx, subsubsection_idx = -1, -1, -1
    last_processed_paragraph: str or None = None
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
                        "title": paragraph.text.strip().replace('\xa0', ' '),
                        "description": "",
                        "summary": "",
                        "tables": [],
                        "figures_meta_data": [],
                        "subsections": []
                    }
                    document_json["content"].append(current_section)
                    section_idx += 1
                    current_subsection, subsection_idx = None, -1
                    current_subsubsection, subsubsection_idx = None, -1
                elif heading_level == 2:
                    current_subsection = {
                        "title": paragraph.text.strip().replace('\xa0', ' '),
                        "description": "",
                        "summary": "",
                        "text_content": "",
                        "tables": [],
                        "figures_meta_data": [],
                        "subsubsections": []
                    }
                    subsection_idx += 1
                    if current_section:
                        current_section["subsections"].append(current_subsection)
                    current_subsubsection, subsubsection_idx = None, -1
                elif heading_level == 3:
                    current_subsubsection = {
                        "title": paragraph.text.strip().replace('\xa0', ' '),
                        "text_content": "",
                        "figures_meta_data": [],
                        "tables": []
                    }
                    subsubsection_idx += 1
                    if current_subsection:
                        current_subsection["subsubsections"].append(current_subsubsection)
                last_processed_paragraph = None
            else:
                try:
                    next_paragraphs = parts[idx + 1:]
                except:
                    next_paragraphs = None
                # Determine the current context
                if current_subsubsection:
                    context = current_subsubsection
                    text_field = "text_content"
                elif current_subsection:
                    context = current_subsection
                    text_field = "text_content"
                elif current_section:
                    context = current_section
                    text_field = "description"
                else:
                    continue

                processed_text, num, figures_data = process_paragraph(paragraph, next_paragraphs, doc,
                                                                      last_paragraph_txt=context[text_field])
                number_figures += num
                context[text_field] += processed_text
                if figures_data:
                    figures_meta_data = [{
                        key: figure_data_dict[key]
                        for key in ["figure_id", "last_paragraph"]
                    } for figure_data_dict in figures_data]
                    context["figures_meta_data"].extend(figures_meta_data)
                    for figure_data in figures_data:
                        figure_data["section_title"] = current_section["title"]
                        figure_data["section_idx"] = section_idx
                        # print(f"Section {section_idx} of title {current_section['title']}")
                        figure_data["subsection_title"], figure_data["subsection_idx"] = \
                            ((current_subsection["title"], subsection_idx) if current_subsection else ("", -1))
                        # if current_subsection:
                            # print(f"\tSubSection {subsection_idx} of name {current_subsection['title']}")
                        figure_data["subsubsection_title"], figure_data["subsubsection_idx"] = \
                            (current_subsubsection["title"], subsubsection_idx) if current_subsubsection else (
                                "", -1)
                        # if current_subsubsection:
                            # print(f"\t\tSubSubSection {subsubsection_idx} of name {current_subsubsection['title']}")
                    figures_for_preprocessing.extend(figures_data)
                last_processed_paragraph = paragraph
        elif isinstance(element, Table):
            table_description = last_processed_paragraph.text.strip() if last_processed_paragraph else ""
            table_entry = {
                "description": table_description,
                "table number": table_number,
                "summary": "",
                "name": ""
            }
            table_number += 1

            if current_subsubsection:
                current_subsubsection["tables"].append(table_entry)
            elif current_subsection:
                current_subsection["tables"].append(table_entry)
            elif current_section:
                current_section["tables"].append(table_entry)
            last_processed_paragraph = None
    print("number of figures is :", number_figures)
    return document_json, figures_for_preprocessing


def process_image_document(
        document_json: Dict[str, Any],
        figures_for_preprocessing,
        vision_prompt: str = VISION_PROMPT,
        endpoint_url: str = "http://127.0.0.1:5003/process"
) -> Dict[str, Any]:
    """
    Iterate through figures, send each with a vision prompt to a local processing endpoint,
    and insert the returned description into the document JSON.

    Args:
        document_json: The original document structure.
        figures_for_preprocessing: List of figure metadata dicts.
        vision_prompt: A template string containing placeholders {last_paragraph} and {figure_name}.
        endpoint_url: URL of the local image-processing service.

    Returns:
        Updated document_json with descriptions filled in for each figure.
    """
    content = document_json.get("content", [])
    for fig in figures_for_preprocessing:
        # Build the prompt
        last_paragraph = fig.get("last_paragraph", "")
        figure_name = fig.get("figure_name", "")
        prompt = vision_prompt.format(
            last_paragraph=last_paragraph,
            figure_name=figure_name
        )

        # Prepare files payload using in-memory bytes
        file_tuple = (
            f"{figure_name}.{fig.get('image_ext', 'png')}",
            io.BytesIO(fig["image_bytes"]),
            f"image/{fig.get('image_ext', 'png')}"
        )
        files = {"image": file_tuple}
        data = {"message": prompt}

        # Send request
        response = requests.post(endpoint_url, files=files, data=data)
        try:
            response.raise_for_status()
        except  :
            print(f"Error {response.json()}")
            continue
        description = response.text.strip()
        print(f"the description of the figure {figure_name} is {description}")

        section_idx, subsection_idx, subsubsection_idx = (
            int(fig["section_idx"]), int(fig["subsection_idx"]), int(fig["subsubsection_idx"])
        )
        if subsection_idx == -1:
            section_part, text_field = content[section_idx], "description"
        elif subsubsection_idx == -1:
            section_part, text_field = content[section_idx]["subsections"][subsection_idx], "text_content"
        else:
            section_part, text_field = (
                content[section_idx]["subsections"][subsection_idx]["subsubsections"][subsubsection_idx],
                "text_content")
        section_part[text_field].replace(f"<<<{figure_name}>>>", description)

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
results, figures_for_preprocessing = extract_docx_structure(testing_docx_path)
results = process_image_document(results, figures_for_preprocessing)
t2 = time.time()
print(f"it takes to extract from this file : ", t2 - t1)

with open('./ProcessedDocuments/23003-i40-testing.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
