import os
import sys
import json
import traceback
import time
from pathlib import Path
from setup import HierarchicalGraphRAG  # Make sure this import works

class JsonProcessor:
    def __init__(self, weaviate_url="http://localhost:8080", backup_fixed=True):
        """
        Initialize the JSON processor
        
        Args:
            weaviate_url (str): URL for the Weaviate instance
            backup_fixed (bool): Whether to save fixed JSON files
        """
        self.weaviate_url = weaviate_url
        self.backup_fixed = backup_fixed
        self.rag_system = None
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "fixed": 0
        }
        self.failed_files = []
        
    def initialize_rag(self):
        """Initialize the HierarchicalGraphRAG system"""
        try:
            print("\n🚀 Initializing HierarchicalGraphRAG...")
            self.rag_system = HierarchicalGraphRAG(
                weaviate_url=self.weaviate_url,
                optimize_disk=True,
                num_clusters=5  # Using a smaller number for test
            )
            print("✅ HierarchicalGraphRAG initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize HierarchicalGraphRAG: {e}")
            traceback.print_exc()
            return False
    
    def test_json_structure(self, json_data):
        """
        Validate the JSON structure before attempting ingestion
        
        Args:
            json_data: The JSON data to validate
            
        Returns:
            list: Issues found in the JSON structure
        """
        issues = []
        
        # Check basic structure
        if not isinstance(json_data, dict):
            issues.append("JSON data is not a dictionary")
            return issues
        
        # Check for required fields
        if "document_name" not in json_data:
            issues.append("Missing 'document_name' field")
        
        if "content" not in json_data:
            issues.append("Missing 'content' field")
            return issues
        
        if not isinstance(json_data["content"], list):
            issues.append("'content' field is not a list")
            return issues
        
        # Check sections structure
        for i, section in enumerate(json_data["content"]):
            # Check section fields
            if not isinstance(section, dict):
                issues.append(f"Section {i} is not a dictionary")
                continue
                
            # Title check with handling for both string and dict formats
            if "title" not in section:
                issues.append(f"Section {i} is missing 'title' field")
            elif isinstance(section["title"], dict):
                if "title" not in section["title"]:
                    issues.append(f"Section {i} has malformed title dictionary (missing 'title' key)")
                if "embedding" in section["title"] and not isinstance(section["title"]["embedding"], list):
                    issues.append(f"Section {i} has invalid title embedding format")
                    
            # Description check with handling for both string and dict formats
            if "description" in section:
                if isinstance(section["description"], dict):
                    if "description" not in section["description"]:
                        issues.append(f"Section {i} has malformed description dictionary (missing 'description' key)")
                    if "embedding" in section["description"] and not isinstance(section["description"]["embedding"], list):
                        issues.append(f"Section {i} has invalid description embedding format")
            
            # Check subsections if they exist
            if "subsections" in section:
                if not isinstance(section["subsections"], list):
                    issues.append(f"Section {i} 'subsections' is not a list")
                    continue
                    
                for j, subsec in enumerate(section["subsections"]):
                    if not isinstance(subsec, dict):
                        issues.append(f"Subsection {j} in section {i} is not a dictionary")
                        continue
                    
                    # Title check
                    if "title" not in subsec:
                        issues.append(f"Subsection {j} in section {i} is missing 'title' field")
                    elif isinstance(subsec["title"], dict):
                        if "title" not in subsec["title"]:
                            issues.append(f"Subsection {j} in section {i} has malformed title dictionary")
                        if "embedding" in subsec["title"] and not isinstance(subsec["title"]["embedding"], list):
                            issues.append(f"Subsection {j} in section {i} has invalid title embedding format")
                    
                    # Check chunks if they exist
                    if "chunks" in subsec:
                        if not isinstance(subsec["chunks"], list):
                            issues.append(f"Subsection {j} in section {i} 'chunks' is not a list")
                            continue
                            
                        for k, chunk in enumerate(subsec["chunks"]):
                            if not isinstance(chunk, dict):
                                issues.append(f"Chunk {k} in subsection {j}, section {i} is not a dictionary")
                                continue
                                
                            # Check for content field in chunks
                            content_found = False
                            for content_field in ["chunk", "content"]:
                                if content_field in chunk:
                                    content_found = True
                                    break
                                    
                            if not content_found:
                                issues.append(f"Chunk {k} in subsection {j}, section {i} is missing content field")
                                
                            # Check embedding format
                            if "embedding" in chunk and not isinstance(chunk["embedding"], list):
                                issues.append(f"Chunk {k} in subsection {j}, section {i} has invalid embedding format")
        
        return issues

    def fix_json_issues(self, json_data, filename=""):
        """
        Attempt to fix common issues in the JSON data
        
        Args:
            json_data: The JSON data to fix
            filename: Name of the file being processed (for logs)
            
        Returns:
            dict: The fixed JSON data
        """
        if not isinstance(json_data, dict):
            print(f"⚠️ [{filename}] Cannot fix: JSON data is not a dictionary")
            return json_data
        
        fixes_applied = 0
        
        # Ensure required fields exist
        if "document_name" not in json_data:
            doc_name = os.path.splitext(os.path.basename(filename))[0] if filename else "Unnamed Document"
            json_data["document_name"] = doc_name
            print(f"⚠️ [{filename}] Added document_name: '{doc_name}'")
            fixes_applied += 1
        
        if "content" not in json_data:
            json_data["content"] = []
            print(f"⚠️ [{filename}] Added empty content array")
            fixes_applied += 1
            return json_data  # No point continuing if no content
        
        if not isinstance(json_data["content"], list):
            content_value = json_data["content"]
            json_data["content"] = []
            print(f"⚠️ [{filename}] Converted content to an empty array (previous value was not a list)")
            fixes_applied += 1
            return json_data
        
        # Process each section
        for i, section in enumerate(json_data["content"]):
            if not isinstance(section, dict):
                json_data["content"][i] = {"title": f"Section {i+1}", "description": ""}
                print(f"⚠️ [{filename}] Replaced invalid Section {i} with placeholder")
                fixes_applied += 1
                continue
            
            # Fix title
            if "title" not in section:
                section["title"] = f"Section {i+1}"
                print(f"⚠️ [{filename}] Added default title to Section {i}")
                fixes_applied += 1
            elif isinstance(section["title"], dict):
                if "title" not in section["title"]:
                    section["title"]["title"] = f"Section {i+1}"
                    print(f"⚠️ [{filename}] Added default title text to Section {i}")
                    fixes_applied += 1
                if "embedding" in section["title"] and not isinstance(section["title"]["embedding"], list):
                    section["title"]["embedding"] = [0.0] * 10  # Default placeholder embedding
                    print(f"⚠️ [{filename}] Fixed invalid title embedding format in Section {i}")
                    fixes_applied += 1
            
            # Fix description
            if "description" in section and isinstance(section["description"], dict):
                if "description" not in section["description"]:
                    section["description"]["description"] = ""
                    print(f"⚠️ [{filename}] Added empty description text to Section {i}")
                    fixes_applied += 1
                if "embedding" in section["description"] and not isinstance(section["description"]["embedding"], list):
                    section["description"]["embedding"] = [0.0] * 10
                    print(f"⚠️ [{filename}] Fixed invalid description embedding format in Section {i}")
                    fixes_applied += 1
            
            # Process subsections if they exist
            if "subsections" in section:
                if not isinstance(section["subsections"], list):
                    section["subsections"] = []
                    print(f"⚠️ [{filename}] Replaced invalid subsections in Section {i} with empty array")
                    fixes_applied += 1
                    continue
                    
                for j, subsec in enumerate(section["subsections"]):
                    if not isinstance(subsec, dict):
                        section["subsections"][j] = {"title": f"Subsection {j+1}", "description": ""}
                        print(f"⚠️ [{filename}] Replaced invalid Subsection {j} in Section {i} with placeholder")
                        fixes_applied += 1
                        continue
                    
                    # Fix subsection title
                    if "title" not in subsec:
                        subsec["title"] = f"Subsection {j+1}"
                        print(f"⚠️ [{filename}] Added default title to Subsection {j} in Section {i}")
                        fixes_applied += 1
                    elif isinstance(subsec["title"], dict):
                        if "title" not in subsec["title"]:
                            subsec["title"]["title"] = f"Subsection {j+1}"
                            print(f"⚠️ [{filename}] Added default title text to Subsection {j} in Section {i}")
                            fixes_applied += 1
                        if "embedding" in subsec["title"] and not isinstance(subsec["title"]["embedding"], list):
                            subsec["title"]["embedding"] = [0.0] * 10
                            print(f"⚠️ [{filename}] Fixed invalid title embedding format in Subsection {j}, Section {i}")
                            fixes_applied += 1
                    
                    # Process chunks if they exist
                    if "chunks" in subsec:
                        if not isinstance(subsec["chunks"], list):
                            subsec["chunks"] = []
                            print(f"⚠️ [{filename}] Replaced invalid chunks in Subsection {j}, Section {i} with empty array")
                            fixes_applied += 1
                            continue
                            
                        for k, chunk in enumerate(subsec["chunks"]):
                            if not isinstance(chunk, dict):
                                subsec["chunks"][k] = {"content": "Empty chunk", "embedding": [0.0] * 10}
                                print(f"⚠️ [{filename}] Replaced invalid Chunk {k} in Subsection {j}, Section {i} with placeholder")
                                fixes_applied += 1
                                continue
                            
                            # Ensure chunk has content
                            content_found = False
                            for content_field in ["chunk", "content"]:
                                if content_field in chunk:
                                    content_found = True
                                    break
                                    
                            if not content_found:
                                chunk["content"] = "Empty chunk"
                                print(f"⚠️ [{filename}] Added default content to Chunk {k} in Subsection {j}, Section {i}")
                                fixes_applied += 1
                            
                            # Fix embedding format
                            if "embedding" in chunk and not isinstance(chunk["embedding"], list):
                                chunk["embedding"] = [0.0] * 10
                                print(f"⚠️ [{filename}] Fixed invalid embedding format in Chunk {k}, Subsection {j}, Section {i}")
                                fixes_applied += 1
        
        if fixes_applied > 0:
            self.stats["fixed"] += 1
            
        return json_data

    def process_json_file(self, filepath):
        """
        Process a single JSON file
        
        Args:
            filepath (str): Path to the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        filename = os.path.basename(filepath)
        try:
            print(f"\n🔍 Loading JSON file: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validate the JSON structure
            print(f"🔎 Validating JSON structure for {filename}...")
            issues = self.test_json_structure(json_data)
            
            if issues:
                print(f"⚠️ [{filename}] Found {len(issues)} issues with the JSON structure:")
                for i, issue in enumerate(issues):
                    print(f"  {i+1}. {issue}")
                    
                print(f"🔧 [{filename}] Attempting to fix issues...")
                fixed_json = self.fix_json_issues(json_data, filename)
                
                # Validate again after fixing
                fixed_issues = self.test_json_structure(fixed_json)
                if fixed_issues:
                    print(f"⚠️ [{filename}] Still have {len(fixed_issues)} issues after fixing:")
                    for i, issue in enumerate(fixed_issues):
                        print(f"  {i+1}. {issue}")
                    print(f"⚠️ [{filename}] Proceeding with partially fixed JSON")
                else:
                    print(f"✅ [{filename}] Successfully fixed all issues!")
                    
                json_data = fixed_json
                
                # Save the fixed JSON if requested
                if self.backup_fixed:
                    fixed_path = filepath.replace('.json', '_fixed.json')
                    with open(fixed_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
                    print(f"✅ [{filename}] Fixed JSON saved to: {os.path.basename(fixed_path)}")
            else:
                print(f"✅ [{filename}] JSON structure is valid")
            
            # Basic JSON info
            print(f"📄 [{filename}] Document name: {json_data.get('document_name', 'Unknown')}")
            print(f"📊 [{filename}] Content sections: {len(json_data.get('content', []))}")
            
            # Make sure RAG is initialized
            if self.rag_system is None:
                if not self.initialize_rag():
                    return False
            
            # Ingest the document
            print(f"📥 [{filename}] Ingesting document...")
            try:
                self.rag_system.ingest_documents([json_data])
                print(f"✅ [{filename}] Document successfully ingested!")
                self.stats["successful"] += 1
                return True
                
            except Exception as e:
                print(f"❌ [{filename}] Error during ingestion: {e}")
                self.failed_files.append((filename, str(e)))
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ [{filename}] JSON parsing error: {e}")
            print(f"   Error at line {e.lineno}, column {e.colno}")
            print(f"   Error location: {e.msg}")
            self.failed_files.append((filename, f"JSON parsing error: {e}"))
            return False
        except Exception as e:
            print(f"❌ [{filename}] Unexpected error: {e}")
            traceback.print_exc()
            self.failed_files.append((filename, str(e)))
            return False

    def process_directory(self, directory_path):
        """
        Process all JSON files in a directory
        
        Args:
            directory_path (str): Path to the directory containing JSON files
        """
        start_time = time.time()
        
        if not os.path.isdir(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return
        
        # Get all JSON files in the directory
        json_files = list(Path(directory_path).glob("*.json"))
        
        if not json_files:
            print(f"⚠️ No JSON files found in directory: {directory_path}")
            return
        
        print(f"📂 Found {len(json_files)} JSON files in {directory_path}")
        
        # Initialize the RAG system once
        if not self.initialize_rag():
            print("❌ Cannot proceed without initializing RAG system")
            return
        
        # Process each file
        for i, json_file in enumerate(json_files):
            print(f"\n[{i+1}/{len(json_files)}] Processing {os.path.basename(json_file)}")
            self.stats["processed"] += 1
            if not self.process_json_file(str(json_file)):
                self.stats["failed"] += 1
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"📊 Processing Summary (Total time: {elapsed_time:.2f} seconds)")
        print("="*50)
        print(f"Total files processed: {self.stats['processed']}")
        print(f"Successfully ingested: {self.stats['successful']}")
        print(f"Files with errors: {self.stats['failed']}")
        print(f"Files fixed: {self.stats['fixed']}")
        
        if self.failed_files:
            print("\n❌ Failed files:")
            for filename, error in self.failed_files:
                print(f"  - {filename}: {error}")
        
        print("="*50)

def main():
    # Parse command-line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Interactive mode
        print("📂 Please enter the path to your JSON file or directory:")
        path = input("> ").strip()
    
    # Initialize processor
    processor = JsonProcessor(backup_fixed=True)
    
    # Check if path is file or directory
    if os.path.isfile(path):
        if path.lower().endswith('.json'):
            processor.process_json_file(path)
        else:
            print(f"❌ Not a JSON file: {path}")
    elif os.path.isdir(path):
        processor.process_directory(path)
    else:
        print(f"❌ Path not found: {path}")

if __name__ == "__main__":
    main()