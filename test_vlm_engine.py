import os

from backend.core.vlm_engine import VLMEngine


def test_ocr_extraction():
    # Model yollarını senin belirttiğin klasöre göre ayarlıyoruz
    model_yolu = "backend/models/ZwZ-4B-Q4_K_M.gguf"
    mmproj_yolu = "backend/models/mmproj-ZwZ-4B-F16.gguf"

    # Bir önceki aşamada çıkardığımız test resimlerinden birini seçiyoruz
    # Lütfen klasördeki var olan bir .png dosyasının adını buraya yaz (Fark makinesi olursa harika olur)
    test_resim_yolu = (
        "backend/data/temp_images/test/c.png"  # BURAYI KENDİ DOSYANA GÖRE GÜNCELLE
    )

    if not os.path.exists(test_resim_yolu):
        print(f"[HATA] Test edilecek resim bulunamadı: {test_resim_yolu}")
        return

    print("=== VLM OCR TESTİ BAŞLIYOR ===")
    try:
        # Motoru ayağa kaldır
        vlm = VLMEngine(model_path=model_yolu, mmproj_path=mmproj_yolu)

        # Resmi okut
        sonuc = vlm.extract_text(test_resim_yolu)

        print("\n--- VLM'DEN GELEN ÇIKTI ---")
        if sonuc == "":
            print(
                "[BOŞ ÇIKTI] VLM resimde bir metin veya tablo bulamadı (Sistem İstemi çalıştı)."
            )
        else:
            print(sonuc)
        print("---------------------------\n")

    except Exception as e:
        print(f"BİR HATA OLUŞTU: {e}")
    finally:
        # Ne olursa olsun VRAM'i boşalt
        if "vlm" in locals():
            vlm.unload()
            print("=== TEST BİTTİ, VRAM BOŞALTILDI ===")


if __name__ == "__main__":
    test_ocr_extraction()
