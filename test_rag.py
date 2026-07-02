import time

from cli_utils import configure_console
from database import get_document_count, init_db
from embedding_utils import find_relevant_chunks
from llm_integration import NO_INFO_MESSAGE, generate_answer, warmup_foundry
from foundry_client import shutdown

TEST_CASES = [
    {
        "type": "E-Sys kurulum (In-Domain)",
        "query": "E-Sys Launcher nasıl kurulur?",
        "must_contain_any": ["launcher", "esys", "pin", "install"],
        "must_not_contain": [],
    },
    {
        "type": "Esyslauncher anahtar kelime",
        "query": "esyslauncher",
        "must_contain_any": ["launcher", "esys", "premium"],
        "must_not_contain": [],
    },
    {
        "type": "Kapsam dışı (Out-of-Domain)",
        "query": "Kıbrıs'ta hava durumu genellikle nasıldır?",
        "must_contain_any": [NO_INFO_MESSAGE.lower()[:20]],
        "must_not_contain": [],
    },
    {
        "type": "Bağlantı adımları",
        "query": "How do I connect to the car with ENET?",
        "must_contain_any": ["enet", "vin", "connect"],
        "must_not_contain": [],
    },
]


def _check_expectations(answer, test):
    answer_lower = answer.lower()
    passed = True
    notes = []

    for term in test.get("must_contain_any", []):
        if term.lower() not in answer_lower:
            passed = False
            notes.append(f"eksik: '{term}'")

    for term in test.get("must_not_contain", []):
        if term.lower() in answer_lower:
            passed = False
            notes.append(f"istenmeyen: '{term}'")

    return passed, notes


def run_tests():
    print("--- RAG OTOMATİK TEST ---\n")
    init_db(quiet=True)

    count = get_document_count()
    if count == 0:
        print("[HATA] Veritabanı boş. Önce 'python ingest.py' çalıştırın.")
        return

    print(f"{count} parça yüklü.\n")

    print("Foundry Local yükleniyor...")
    if not warmup_foundry():
        print("[HATA] Foundry Local başlatılamadı.")
        return
    print("Foundry Local hazır.\n")

    passed_total = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"Test #{i} — {test['type']}")
        print(f"  Soru: {test['query']}")

        start = time.time()
        chunks = find_relevant_chunks(test["query"])
        answer = generate_answer(test["query"], chunks)
        elapsed = time.time() - start

        ok, notes = _check_expectations(answer, test)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed_total += 1

        print(f"  Durum: {status} ({elapsed:.3f}s)")
        if chunks:
            print(f"  En iyi kaynak: {chunks[0]['source']} ({chunks[0]['similarity']:.2f})")
        if notes:
            print(f"  Notlar: {', '.join(notes)}")
        print(f"  Yanıt: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print("-" * 50)

    print(f"\nSonuç: {passed_total}/{len(TEST_CASES)} geçti.")
    shutdown()


if __name__ == "__main__":
    configure_console()
    run_tests()
