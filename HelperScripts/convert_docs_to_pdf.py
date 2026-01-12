import os
import sys
import subprocess

if len(sys.argv) != 2:
    print("Usage: python convert_docs_to_pdf.py /path/to/folder")
    sys.exit(1)

folder = sys.argv[1]

if not os.path.isdir(folder):
    print("Error: Folder does not exist.")
    sys.exit(1)

# Path to LibreOffice on macOS (adjust if installed elsewhere)
soffice_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

if not os.path.exists(soffice_path):
    print("Error: LibreOffice not found. Install it from libreoffice.org.")
    sys.exit(1)

# Find all .doc and .docx files
doc_files = [f for f in os.listdir(folder) if f.lower().endswith(('.doc', '.docx'))]

if not doc_files:
    print("No .doc or .docx files found in the folder.")
    sys.exit(0)

for file in doc_files:
    input_path = os.path.join(folder, file)
    print(f"Converting {file} to PDF...")
    try:
        subprocess.run([soffice_path, '--headless', '--convert-to', 'pdf', input_path], cwd=folder, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file}: {e}")

print("Conversion complete! PDFs are in the same folder.")