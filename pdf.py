import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def get_code_files(directory, excluded_files=None, excluded_dirs=None):
    """Fetch all project files, excluding cache + binary files."""

    if excluded_files is None:
        excluded_files = {
            ".DS_Store",
            "Thumbs.db",
            "Desktop.ini",
            "best_model.pth",  # binary model file
            ".gitignore",  # optional: include if needed
            "script.py",  # skip this file
            "pdf.py",  # skip this file
        }

    if excluded_dirs is None:
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
            "venv",
            "env",
            ".pytest_cache",
        }

    valid_extensions = {".py", ".txt", ".yaml", ".yml", ".md"}

    code_files = {}

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if file in excluded_files:
                continue

            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)

            if ext.lower() not in valid_extensions:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_files[file_path] = f.readlines()

            except Exception as e:
                code_files[file_path] = [f"[Error reading file: {str(e)}]"]

    return code_files


def wrap_text(line, max_chars=100):
    """Wrap long lines into chunks without cutting the PDF."""
    chunks = []
    line = line.rstrip("\n")
    while len(line) > max_chars:
        chunks.append(line[:max_chars])
        line = line[max_chars:]
    chunks.append(line)
    return chunks


def get_file_type_label(file_path):
    """Return a label based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    labels = {
        ".py": "[PY]",
        ".yaml": "[YAML]",
        ".yml": "[YAML]",
        ".txt": "[TXT]",
        ".md": "[MD]",
    }
    
    return labels.get(ext, "[FILE]")


def create_pdf(code_data, output_pdf="Plant_Traning.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    line_height = 10
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "🌱 Plant Health Monitoring - Code Export")
    y -= 2 * line_height

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📄 Project Files:")
    y -= 2 * line_height

    file_paths = sorted(list(code_data.keys()))

    # List of files
    c.setFont("Courier", 8)
    for path in file_paths:
        if y < margin:
            c.showPage()
            c.setFont("Courier", 8)
            y = height - margin

        display_path = os.path.relpath(path)
        file_type = get_file_type_label(path)

        c.drawString(margin, y, f"- {file_type} {display_path}")
        y -= line_height

    # New page for content
    c.showPage()
    y = height - margin

    # Write file contents
    for file_path in file_paths:
        lines = code_data[file_path]
        rel_path = os.path.relpath(file_path)

        if y < margin + 4 * line_height:
            c.showPage()
            y = height - margin

        # File header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"📄 File: {rel_path}")
        y -= line_height

        c.setFont("Courier", 8)
        c.drawString(margin, y, "=" * 100)
        y -= line_height

        # Content
        for line_num, line in enumerate(lines, start=1):

            wrapped_lines = wrap_text(line, max_chars=110)

            for wrapped in wrapped_lines:
                if y < margin:
                    c.showPage()
                    c.setFont("Courier", 8)
                    y = height - margin

                display_line = f"{line_num:3d}: {wrapped}"
                c.drawString(margin, y, display_line)
                y -= line_height

        # Spacer
        y -= line_height
        if y > margin:
            c.drawString(margin, y, "-" * 100)
            y -= 2 * line_height

    c.save()
    print(f"✅ PDF created: {output_pdf}")
    print(f"📄 Total files included: {len(code_data)}")


def main():
    # Running from within the project directory
    project_dir = os.getcwd()
    
    # Alternative: if running from parent directory
    # project_dir = os.path.join(os.getcwd(), "internship", "Plant-Health-monitoring-Internship")

    print("🔍 Scanning Plant Health Monitoring project folder...")
    
    if not os.path.exists(project_dir):
        print(f"❌ Directory not found: {project_dir}")
        print("💡 Make sure you're running this from the correct location")
        return

    code_files = get_code_files(project_dir)

    if not code_files:
        print("❌ No project files found")
        return

    print(f"📁 Files found: {len(code_files)}")
    for f in sorted(code_files.keys()):
        print("   📄", os.path.relpath(f))

    create_pdf(code_files)


if __name__ == "__main__":
    main()