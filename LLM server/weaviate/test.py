import weaviate
import json
import sys


def check_weaviate(url="http://localhost:8080"):
    """
    Diagnostic tool to check Weaviate connection and database status
    """
    print(f"🔌 Connecting to Weaviate at {url}...")
    
    try:
        # Connect to the Weaviate instance
        client = weaviate.Client(url=url)
        
        # Check if Weaviate is running
        meta = client.get_meta()
        print(f"✅ Connected to Weaviate version: {meta['version']}")
        
        # Check schema
        schema = client.schema.get()
        
        # Get all class names
        class_names = [c["class"] for c in schema["classes"]] if "classes" in schema else []     
        print("\n=== Available Classes ===")
        print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
        
        # For each class, count the objects and get some sample data
        print("\n=== Object Counts ===")
        for class_name in class_names:
            try:
                # Count objects in the class
                count_result = client.query.aggregate(class_name).with_meta_count().do()
                count = count_result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
                print(f"{class_name}: {count} objects")
            except Exception as e:
                print(f"{class_name}: Error getting count - {e}")

        # Get sample data from each class
        print("\n=== Sample Data ===")
        for class_name in class_names:
            try:
                # Get schema for this class to find available properties
                class_schema = next((c for c in schema["classes"] if c["class"] == class_name), None)
                if not class_schema:
                    print(f"{class_name}: No schema found")
                    continue
                    
                # Get property names from schema
                property_names = [p["name"] for p in class_schema.get("properties", [])]
                
                # If no properties found, at least get the ID
                if not property_names:
                    property_names = []
                
                # Query with properties and additional ID field
                result = (client.query
                         .get(class_name, property_names)
                         .with_additional(["id"])
                         .with_limit(3)
                         .do())
                
                if "data" in result and "Get" in result["data"] and class_name in result["data"]["Get"]:
                    samples = result["data"]["Get"][class_name]
                    if samples:
                        print(f"{class_name}: {len(samples)} sample(s) retrieved")
                        
                        # Print a summary of each sample
                        for i, sample in enumerate(samples):
                            obj_id = sample.get("_additional", {}).get("id", "unknown-id")
                            # Summarize properties
                            props = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v) 
                                     for k, v in sample.items() if k != "_additional"}
                            print(f"  Sample {i+1} (ID: {obj_id}): {json.dumps(props, indent=2)}")
                    else:
                        print(f"{class_name}: Empty result set returned")
                else:
                    print(f"{class_name}: No data returned from query")
            except Exception as e:
                print(f"{class_name}: Error retrieving samples - {str(e)}")

        # Check for any pending operations
        try:
            status = client.backup.get_status()
            if status.get("status") == "PENDING":
                print("\n⚠️ Warning: There are pending backup operations")
        except:
            # This might not be available in all Weaviate versions
            pass
            
        print("\n✅ Weaviate diagnostics completed")
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow custom URL through command line
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    check_weaviate(url)