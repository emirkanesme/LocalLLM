import os

from cli_utils import configure_console
from config import TOP_K
from database import get_document_count, init_db
from embedding_utils import find_relevant_chunks
from foundry_client import get_model_name, shutdown
from llm_integration import generate_answer, warmup_foundry


class Colors:
    USER = "\033[94m"
    ASSISTANT = "\033[92m"
    SYSTEM = "\033[93m"
    DIM = "\033[90m"
    END = "\033[0m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def _status_callback(phase, percent, message):
    """Print Foundry load progress to the terminal."""
    if message:
        print(f"{Colors.SYSTEM}{message}{Colors.END}")
    if phase == "download" and percent > 0:
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"{Colors.DIM}  [{bar}] {percent:.0f}%{Colors.END}", end="\r", flush=True)
        if percent >= 100:
            print()


def main():
    clear_screen()
    print(f"{Colors.SYSTEM}{'=' * 50}{Colors.END}")
    print(f"{Colors.SYSTEM}  Yerel RAG AI Asistanı (Foundry Local){Colors.END}")
    print(f"{Colors.SYSTEM}  Çıkmak: q / exit / quit / çıkış{Colors.END}")
    print(f"{Colors.SYSTEM}{'=' * 50}\n{Colors.END}")

    try:
        init_db(quiet=True)
    except Exception as e:
        print(f"{Colors.SYSTEM}[HATA] Veritabanı başlatılamadı: {e}{Colors.END}")
        return

    doc_count = get_document_count()
    if doc_count == 0:
        print(
            f"{Colors.SYSTEM}[UYARI] Veritabanında doküman yok. "
            f"Önce 'python ingest.py' çalıştırın.{Colors.END}\n"
        )
    else:
        print(f"{Colors.DIM}{doc_count} parça yüklü.{Colors.END}")

    print(f"{Colors.DIM}Model: {get_model_name()}{Colors.END}\n")

    if not warmup_foundry(on_status=_status_callback):
        print(
            f"{Colors.SYSTEM}[HATA] Foundry Local başlatılamadı. "
            f"pip install foundry-local-sdk-winml{Colors.END}"
        )
        return

    print()

    try:
        while True:
            try:
                query = input(f"{Colors.USER}Sen: {Colors.END}").strip()

                if query.lower() in ("q", "çıkış", "exit", "quit"):
                    print(f"\n{Colors.SYSTEM}Görüşmek üzere!{Colors.END}")
                    break

                if not query:
                    continue

                clear_screen()
                print(f"{Colors.USER}Sen:{Colors.END} {query}\n")

                if doc_count == 0:
                    print(
                        f"{Colors.ASSISTANT}Asistan:{Colors.END}\n"
                        "Henüz doküman yüklenmemiş. `python ingest.py` ile docs/ klasörünü indeksleyin.\n"
                    )
                    print(f"{Colors.SYSTEM}{'-' * 50}\n{Colors.END}")
                    continue

                print(f"{Colors.DIM}Aranıyor...{Colors.END}\n")

                chunks = find_relevant_chunks(query, top_k=TOP_K)

                if chunks:
                    print(f"{Colors.DIM}Bulunan kaynaklar:{Colors.END}")
                    for c in chunks:
                        print(
                            f"  {Colors.DIM}• {c['source']} "
                            f"(benzerlik: {c['similarity']:.2f}){Colors.END}"
                        )
                    print()

                print(f"{Colors.DIM}Foundry Local düşünüyor...{Colors.END}\n")
                answer = generate_answer(query, chunks)
                print(f"{Colors.ASSISTANT}Asistan:{Colors.END}\n{answer}\n")
                print(f"{Colors.SYSTEM}{'-' * 50}\n{Colors.END}")

            except KeyboardInterrupt:
                print(f"\n{Colors.SYSTEM}Çıkılıyor...{Colors.END}")
                break
            except Exception as e:
                print(f"\n{Colors.SYSTEM}[HATA] {e}{Colors.END}")
    finally:
        shutdown()


if __name__ == "__main__":
    configure_console()
    main()
