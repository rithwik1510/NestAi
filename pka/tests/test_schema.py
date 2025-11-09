import json
from pathlib import Path

import jsonschema


def test_chat_response_schema_accepts_valid_payload() -> None:
    schema_path = Path("pka/app/services/synth/response_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sample = {
        "abstain": False,
        "answer": "Sample grounded answer.",
        "bullets": ["Key point one"],
        "conflicts": [
            {
                "claim": "Conflicting claim",
                "sources": [{"id": "doc:1", "loc": "L10-L12"}],
            }
        ],
        "sources": [{"id": "doc:1", "loc": "L10-L12"}],
    }

    jsonschema.validate(sample, schema)
