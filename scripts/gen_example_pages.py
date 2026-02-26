"""Generate documentation pages for examples.

Scans examples/*.py, extracts the module docstring for each,
and generates a markdown page with the description and a snippet
include of the source.

    uv run python scripts/gen_example_pages.py
    uv run --group docs zensical build
"""

import ast
import shutil
from pathlib import Path

root = Path(__file__).parent.parent
examples = root / "examples"
out = root / "docs" / "examples"

# Clean and recreate
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

entries = []

for path in sorted(examples.glob("*.py")):
    # Extract module docstring
    tree = ast.parse(path.read_text())
    docstring = ast.get_docstring(tree) or ""

    # Use first line as title, rest as description
    lines = docstring.strip().splitlines()
    if lines:
        title = lines[0].rstrip(".")
        description = "\n".join(lines[1:]).strip()
    else:
        # Fallback: humanize the filename
        title = path.stem.replace("_", " ").title()
        description = ""

    slug = path.stem
    doc_path = out / f"{slug}.md"

    page_lines = [f"# {title}\n"]
    if description:
        page_lines.append(f"{description}\n")
    page_lines.append("## Source\n")
    page_lines.append(f'```python\n--8<-- "examples/{path.name}"\n```\n')

    doc_path.write_text("\n".join(page_lines))
    entries.append((title, f"examples/{slug}.md"))

# Generate index page linking to all examples
index_lines = ["# Examples\n"]
for title, link in entries:
    # link is "examples/slug.md", make it relative to examples/
    rel = link.removeprefix("examples/")
    index_lines.append(f"- [{title}]({rel})")

index = out / "index.md"
index.write_text("\n".join(index_lines) + "\n")

print(f"Generated {len(entries)} example pages in {out.relative_to(root)}")
for title, path in entries:
    print(f"  {path}: {title}")
