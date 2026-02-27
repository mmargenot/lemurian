"""Generate the code reference pages.

Standalone script — no mkdocs plugin dependencies. Run before building:

    uv run python scripts/gen_ref_pages.py
    uv run --group docs zensical build
"""

from pathlib import Path
import shutil

root = Path(__file__).parent.parent
src = root / "src"
out = root / "docs" / "reference"

# Clean and recreate
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

modules = []

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    parts = list(module_path.parts)

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = out / Path(*parts) / "index.md"
    else:
        doc_path = out / Path(*parts).with_suffix(".md")
        modules.append((parts[-1], f"lemurian/{parts[-1]}.md"))

    ident = ".".join(parts)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(f"::: {ident}\n")

# Generate index with links to each module page
lines = ["# API Reference\n"]
for name, link in modules:
    lines.append(f"- [`lemurian.{name}`]({link})")

index = out / "index.md"
index.write_text("\n".join(lines) + "\n")

print(f"Generated reference pages in {out.relative_to(root)}")
