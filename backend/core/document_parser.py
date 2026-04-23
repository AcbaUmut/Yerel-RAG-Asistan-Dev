import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def parse_document(pdf_path: str):
    print(f"[{pdf_path}] dosyası ayrıştırıcıya (parser) alındı...")

    # 1. Aşama: Loader ve Markdown Dönüşümü (pymupdf4llm farkı)
    # Bu işlem PDF'i dümdüz okumaz; tabloları, başlıkları koruyarak Markdown'a çevirir.
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Metni LlamaIndex'in anlayacağı evrensel 'Document' formatına sarıyoruz
    # İleride buraya sayfa numarası, yazar gibi metadata (üst veri) bilgileri de ekleyeceğiz.
    doc = Document(text=md_text, metadata={"file_name": pdf_path})

    # 2. Aşama: Semantik Parçalama (Parser)
    parser = SentenceSplitter(
        chunk_size=450,  # Nomic v2 MoE'nin 512 sınırını asla aşmıyoruz. (Tokenizer farkları için güvenlik payı bırakıldı)
        chunk_overlap=150,  # Kesişimi genişlettik. Bir chunk'ın son 150 tokenı, diğerinin ilk 150 tokenı olacak. Anlam kopmayacak.
    )

    # Tek bir büyük Markdown dokümanını, sindirilebilir küçük düğümlere (nodes) ayırıyoruz
    nodes = parser.get_nodes_from_documents([doc])

    print(
        f"Başarılı! Doküman semantik yapısı korunarak {len(nodes)} adet düğüme ayrıştırıldı."
    )

    return nodes


# --- Test Alanı ---
if __name__ == "__main__":
    test_dosyasi = "test.pdf"

    try:
        uretilen_dugumler = parse_document(test_dosyasi)

        # İlk parçanın Markdown formatında nasıl göründüğünü inceleyelim
        print("\n--- İLK DÜĞÜMÜN (CHUNK) MARKDOWN İÇERİĞİ ---")
        for dugum in uretilen_dugumler:
            print(dugum.text, "\n")
        # print(uretilen_dugumler[0].text)
        print("---------------------------------------------")

    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
