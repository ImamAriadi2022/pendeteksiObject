from nbconvert import ScriptExporter
import nbformat

# Membaca file .ipynb
with open("project.ipynb") as f:
    notebook_content = f.read()

# Parsing isi notebook
notebook_node = nbformat.reads(notebook_content, as_version=4)

# Mengonversi ke script Python
script_exporter = ScriptExporter()
script, resources = script_exporter.from_notebook_node(notebook_node)

# Menyimpan ke file .py
with open("project.py", "w") as f:
    f.write(script)