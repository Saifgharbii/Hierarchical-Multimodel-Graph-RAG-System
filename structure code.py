import os


def load_gitignore():
    """Load ignored patterns from .gitignore (if exists)."""
    gitignore_path = ".gitignore"
    ignored_patterns = set()

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignored_patterns.add(line.strip("/"))  # Normalize paths
    return ignored_patterns


def print_tree(start_path, prefix="", ignored_patterns=None, output_file=None):
    """Recursively print the directory tree, ignoring specified folders."""
    if ignored_patterns is None:
        ignored_patterns = set()

    entries = sorted(os.listdir(start_path))  # Sort alphabetically
    entries = [e for e in entries if e not in ignored_patterns and ".git" not in e]  # Apply ignore filter

    for index, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        line = prefix + connector + entry

        # Write to file
        output_file.write(line + "\n")

        if os.path.isdir(path):
            new_prefix = prefix + ("    " if index == len(entries) - 1 else "│   ")
            print_tree(path, new_prefix, ignored_patterns, output_file)


# Load .gitignore rules and add 'venv' explicitly
ignored_patterns = load_gitignore()
ignored_patterns.add("venv")  # Ensure all venv folders are ignored

# Generate the tree and save to structure.txt
with open("structure.txt", "w", encoding="utf-8") as file:
    print_tree(".", ignored_patterns=ignored_patterns, output_file=file)

print("✅ Folder structure saved to structure.txt")
