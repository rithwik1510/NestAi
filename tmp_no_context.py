import asyncio
from pathlib import Path

from pka.app.core.settings import settings
from pka.app.services.synth.llama_local import ChatService
from pka.app.services.synth.templates import PromptTemplate, PromptTemplateRegistry

registry = PromptTemplateRegistry()
registry.register(
    PromptTemplate(
        name="debug",
        version="1.0.0",
        content=(
            "You must answer strictly in JSON. Context snippets follow.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Schema: {schema_json}\n"
            "Remember to provide actionable abstain guidance if needed."
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

async def main():
    answer = await chat_service.generate(
        question="Summarize the focus improvements.",
        snippets=[],
        mode="synthesize",
    )
    print(answer.model_dump())
    await chat_service.close()

asyncio.run(main())
