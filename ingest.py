import os
import glob
import csv
import io
# Yeni kütüphaneleri dahil ediyoruz
from pypdf import PdfReader
import docx

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

def extract_text_from_file(file_path):
    """Dosya uzantısına göre uygun okuma yöntemini seçer, kodlama hatalarına karşı dayanıklıdır."""
    ext = file_path.lower().split('.')[-1]
    text_content = ""

    # Dışarıdan gelen metin tabanlı dosyalar için (TXT, CSV)
    if ext in ['txt', 'csv']:
        file_text = ""
        # Sırasıyla denenecek karakter kodlamaları: Standart, Türkçe Windows, Batı Avrupa
        encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1254', 'latin-1']
        
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    file_text = f.read()
                break  # Hatasız okunduysa döngüyü kır ve çık
            except UnicodeDecodeError:
                continue  # Hata verdiyse çökmek yerine bir sonraki kodlamayı dene
                
        # Eğer yukarıdaki kodlamaların HİÇBİRİ işe yaramadıysa, hatalı karakterleri yut (replace)
        if not file_text:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                file_text = f.read()

        # Okunan ham metni uzantıya göre işle
        if ext == 'txt':
            text_content = file_text
        elif ext == 'csv':
            # Ham metni hafızada (RAM) sanal bir dosya gibi açıp CSV ayrıştırıcısına veriyoruz
            reader = csv.reader(io.StringIO(file_text))
            for row in reader:
                text_content += " ".join(row) + "\n"

    # PDF Dosyaları için
    elif ext == 'pdf':
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"

    # Word Dosyaları için
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])

    return text_content

def process_files(clear_first=True):
    """Read all supported files in docs/ and index them into SQLite."""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"'{DOCS_DIR}' klasörü oluşturuldu. Lütfen içine dosyalarınızı ekleyin.")
        return

    # Desteklenen tüm uzantıları tarıyoruz
    supported_extensions = ["*.txt", "*.csv", "*.pdf", "*.docx"]
    file_paths = []
    for ext in supported_extensions:
        file_paths.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    if not file_paths:
        print(f"Uyarı: '{DOCS_DIR}' klasöründe desteklenen formatta dosya bulunamadı.")
        print("Desteklenen formatlar: .txt, .csv, .pdf, .docx")
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
            # Okuma işlemini yeni yardımcı fonksiyonumuza devrettik
            text_content = extract_text_from_file(file_path)

            if not text_content.strip():
                print(f"⏭  {source_name} boş veya okunamadı, atlandı.")
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