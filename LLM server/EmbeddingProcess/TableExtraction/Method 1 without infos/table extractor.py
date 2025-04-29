import time

import tabula



from docx import Document
import pandas as pd


def extract_tables_from_docx(docx_path,base_filename):
    doc = Document(docx_path)
    all_tables = []

    for table_idx, table in enumerate(doc.tables):
        data = []
        for row in table.rows:
            data.append([cell.text.strip() for cell in row.cells])

        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)

        # Handling merged cells by filling empty values (Optional: Adjust as needed)
        df.fillna(method="ffill", axis=0, inplace=True)
        df.fillna(method="ffill", axis=1, inplace=True)

        all_tables.append(df)
    for idx, table in enumerate(all_tables):
        table.to_csv(f".\\tables_testing\\{base_filename}_table_{idx + 1}.csv", index=False, encoding="utf-8")
        table.to_json(f".\\tables_testing\\{base_filename}_table_{idx + 1}.json", orient="records", indent=4)
    print(f"Extracted {len(all_tables)} tables and saved as CSV and JSON.")




output_json = "tables.json"

docx_file = "../Testing Docx/22104-i30.docx"
t1 = time.time()
extract_tables_from_docx(docx_file,"./tables_testing/22104-i30")
t2 = time.time()

print(f"it takes about {t2-t1} to extract all the tables" )

