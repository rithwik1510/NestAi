import asyncio
from pathlib import Path

from pka.app.core.settings import settings
from pka.app.services.retrieval.context_builder import ContextSnippet
from pka.app.services.synth.llama_local import ChatService
from pka.app.services.synth.templates import PromptTemplate, PromptTemplateRegistry

registry = PromptTemplateRegistry()
registry.register(
    PromptTemplate(
        name="debug",
        version="1.0.0",
        content=(
            "You must answer the user's question using ONLY these context snippets:\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Return a JSON object matching this schema exactly:\n\n"
            "{schema_json}\n\n"
            "Rules:\n"
            "- Cite every claim with the provided citation identifiers.\n"
            "- If the context is insufficient, set \"abstain\": true and give actionable guidance.\n"
            "- Do not invent sources or information beyond the snippets."
        ),
    )
)

chat_service = ChatService(
    base_url=settings.ollama_base_url,
    model=settings.ollama_chat_model,
    temperature=settings.llm_temperature,
    seed=settings.llm_seed,
    timeout=settings.ollama_timeout_seconds,
    template_registry=registry,
    template_name="debug",
    schema_path=Path("pka/app/services/synth/response_schema.json"),
)

snippet = ContextSnippet(
    document_id=1,
    chunk_id=1,
    content="Deep work guardrails improved focus during morning sessions.",
    citation="deep_work.md:L10-L14",
    rationale="relevant lines",
)

async def main() -> None:
    answer = await chat_service.generate(
        question="What do the notes say about focus improvements?",
        snippets=[snippet],
        mode="synthesize",
    )
    print(answer.model_dump())
    await chat_service.close()

asyncio.run(main())
