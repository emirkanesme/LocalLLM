import atexit
import sys
from flask import Flask, render_template, request, jsonify
import os
from ingest import process_files # Yazdığın ingest.py dosyasını import et
from config import DOCS_DIR
import io
# Mevcut modüllerin
from config import TOP_K
from database import get_document_count, init_db
from embedding_utils import find_relevant_chunks
from foundry_client import get_model_name, shutdown
from llm_integration import generate_answer, warmup_foundry

app = Flask(__name__)

def setup_system():
    """Uygulama ilk başladığında veritabanını ve modeli hazırla."""
    init_db(quiet=True)
    
    def status_callback(phase, percent, message):
        if message:
            print(f"[Foundry] {message} - %{percent:.0f}")
            
    print("Foundry Local modeli ısıtılıyor (Warm-up)...")
    if not warmup_foundry(on_status=status_callback):
        print("[HATA] Foundry Local başlatılamadı.")

# Sunucu başlamadan önce kurulumu yap
setup_system()

# Flask sunucusu kapandığında Foundry'yi güvenli bir şekilde kapat
atexit.register(shutdown)

@app.route("/")
def index():
    """Ana sayfayı ve HTML arayüzünü render eder."""
    doc_count = get_document_count()
    model_name = get_model_name()
    return render_template("index.html", doc_count=doc_count, model_name=model_name)

@app.route("/chat", methods=["POST"])
def chat():
    """Frontend'den gelen soruları alır ve RAG pipeline'ını çalıştırır."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Sorgu boş olamaz"}), 400

    doc_count = get_document_count()
    if doc_count == 0:
        return jsonify({
            "answer": "Henüz doküman yüklenmemiş. Lütfen önce `python ingest.py` ile verilerinizi indeksleyin.",
            "sources": []
        })

    # 1. Retrieval (Vektör + Leksikal Arama)
    chunks = find_relevant_chunks(query, top_k=TOP_K)

    # 2. Generation (LLM Yanıtı)
    answer = generate_answer(query, chunks)

    # Bulunan kaynakları ön yüze göndermek için formatlama
    sources = [
        {"name": c["source"], "score": float(c["similarity"])} 
        for c in chunks
    ]

    return jsonify({
        "answer": answer,
        "sources": sources
    })
@app.route("/ingest")
def ingest_page():
    """Ingestion arayüzünü açar"""
    return render_template("ingest.html")

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Arayüzden gelen dosyaları docs/ klasörüne kaydeder"""
    if 'files' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    files = request.files.getlist('files')
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        
    saved_count = 0
    for file in files:
        if file.filename:
            filepath = os.path.join(DOCS_DIR, file.filename)
            file.save(filepath)
            saved_count += 1
            
    return jsonify({"message": f"{saved_count} dosya başarıyla yüklendi."})

@app.route("/api/ingest", methods=["POST"])
def run_ingestion():
    """ingest.py içindeki process_files() fonksiyonunu çalıştırır ve terminal loglarını HTML'e yollar"""
    # Terminal çıktılarını (print) yakalamak için stdout'u geçici olarak yönlendiriyoruz
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        process_files(clear_first=True)
        log_output = captured_output.getvalue()
        return jsonify({"message": "Başarılı", "log": log_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        sys.stdout = old_stdout # Standart çıktıya geri dön
if __name__ == "__main__":
    # use_reloader=False önemlidir, aksi takdirde model iki kez belleğe yüklenmeye çalışır
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)