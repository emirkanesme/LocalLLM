from config import VERBOSE_LLM
from foundry_client import complete_chat, get_init_error, initialize, is_ready

NO_INFO_MESSAGE = "Üzgünüm, belgelerimde bu soruya dair bir bilgi bulunmuyor."


def _build_context(retrieved_chunks):
    return "\n\n".join(
        f"[Kaynak: {chunk['source']}]\n{chunk['content']}"
        for chunk in retrieved_chunks
    )


def _build_messages(context_text, query):
    system = (
        "You are a technical documentation assistant. "
        "Answer ONLY using the CONTEXT below. "
        "Be concise, accurate, and step-by-step when relevant. "
        "Always mention which source file you used. "
        "Reply in the same language as the user's question. "
        f"If the answer is not in the context, reply with exactly: {NO_INFO_MESSAGE}"
    )
    user = f"""CONTEXT:
{context_text}

QUESTION: {query}"""
    return system, user


def warmup_foundry(on_status=None):
    """Load Foundry model at startup. Returns True on success."""
    return initialize(on_status=on_status)


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return NO_INFO_MESSAGE

    context_text = _build_context(retrieved_chunks)
    system_prompt, user_message = _build_messages(context_text, query)

    if VERBOSE_LLM:
        print("\n[SİSTEM LOGU - FOUNDRY PROMPT]")
        print("SYSTEM:", system_prompt)
        print("USER:", user_message)
        print("-" * 50 + "\n")

    if not is_ready() and not initialize():
        error = get_init_error() or "bilinmeyen hata"
        return (
            f"[HATA] Foundry Local yüklenemedi: {error}\n\n"
            "Kurulum: pip install foundry-local-sdk-winml"
        )

    try:
        answer = complete_chat(system_prompt, user_message)
        if VERBOSE_LLM:
            print("[LLM] Foundry Local kullanıldı.\n")
        return answer or NO_INFO_MESSAGE
    except Exception as exc:
        return f"[HATA] Foundry Local yanıt üretemedi: {exc}"
