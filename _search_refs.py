"""Temporary helper: find references to specific patterns across the repo."""
import os

PATTERNS = [
    "assets/styles",
    "assets/app",
    "st.components.v1.html",
    "save_index",
    "load_index",
    "generate_literature_survey",
    "literature_survey",
    "import app",
]

SKIP_DIRS = {".venv", ".git", "vector_db", "processed_data", "logs", "data",
             "uploads", "models", "__pycache__", ".pytest_cache"}

EXTENSIONS = {".py", ".html", ".toml", ".md", ".txt", ".yml", ".yaml", ".css", ".js"}

def main():
    hits = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith(tuple(EXTENSIONS)):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for pat in PATTERNS:
                            if pat in line:
                                hits.append((path, lineno, pat, line.strip()[:150]))
                                break
            except OSError:
                continue

    if not hits:
        print("No references found.")
        return
    for path, lineno, pat, text in hits:
        print(f"{path}:{lineno} [{pat}] {text}")

if __name__ == "__main__":
    main()

