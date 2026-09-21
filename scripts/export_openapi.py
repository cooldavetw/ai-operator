"""Run from the repository root: python -m scripts.export_openapi."""
import json
from pathlib import Path

from app.config import Settings
from app.main import create_app


def main():
    # Schema generation does not start the application, open a DB, or load a model.
    app = create_app(Settings(api_key="schema-generation-placeholder-key"))
    path = Path("docs/openapi.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

