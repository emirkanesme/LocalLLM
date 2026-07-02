import glob
import os

from cli_utils import configure_console
from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from database import (
    clear_documents,
    get_document_count,
    init_db,
    insert_document,
    set_metadata,
)
from embedding_utils import build_idf, chunk_text, term_frequencies, tokenize


def process_files(clear_first=True):
    """Read all .txt files in docs/ and index them into SQLite."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"'{DOCS_DIR}' klasörü oluşturuldu. Lütfen içine .txt dosyaları ekleyin.")
        return

    file_paths = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not file_paths:
        print(f"Uyarı: '{DOCS_DIR}' klasöründe hiç .txt dosyası bulunamadı.")
        return

    init_db(quiet=True)

    if clear_first:
        clear_documents()
        print("Mevcut indeks temizlendi.")

    print(f"Toplam {len(file_paths)} dosya bulundu. İşleniyor...\n")

    all_term_freqs = []
    total_chunks = 0

    for file_path in file_paths:
        source_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            if not text_content.strip():
                print(f"⏭  {source_name} boş, atlandı.")
                continue

            chunks = chunk_text(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                tf = term_frequencies(tokenize(chunk))
                insert_document(chunk, tf, source_name)
                all_term_freqs.append(tf)
                total_chunks += 1

            print(f"✅ {source_name} — {len(chunks)} parça")

        except Exception as e:
            print(f"❌ {source_name} hatası: {e}")

    if all_term_freqs:
        idf_map = build_idf(all_term_freqs)
        set_metadata("idf", idf_map)
        set_metadata("chunk_size", CHUNK_SIZE)
        set_metadata("chunk_overlap", CHUNK_OVERLAP)
        set_metadata("doc_count", len(file_paths))

    print(f"\nTamamlandı: {total_chunks} parça, {get_document_count()} kayıt veritabanında.")


if __name__ == "__main__":
    configure_console()
    print("--- RAG Veri İçeri Alma (Ingestion) ---\n")
    process_files()
