import os
import time

from core.vlm_engine import VLMEngine


def test_ocr_extraction():

    # Bir önceki aşamada çıkardığımız test resimlerinden birini seçiyoruz
    # Lütfen klasördeki var olan bir .png dosyasının adını buraya yaz (Fark makinesi olursa harika olur)
    test_resim_yolu = "backend/data/yuksek/test2/test2.pdf-0002-01.png"  # BURAYI KENDİ DOSYANA GÖRE GÜNCELLE

    if not os.path.exists(test_resim_yolu):
        print(f"[HATA] Test edilecek resim bulunamadı: {test_resim_yolu}")
        return

    start_time = time.time()
    print("=== VLM OCR TESTİ BAŞLIYOR ===")
    try:
        # Motoru ayağa kaldır
        vlm = VLMEngine()

        # Resmi okut
        sonuc = vlm.extract_text(test_resim_yolu)

        print("\n--- VLM'DEN GELEN ÇIKTI ---")
        if sonuc == "":
            print(
                "[BOŞ ÇIKTI] VLM resimde bir metin veya test2 bulamadı (Sistem İstemi çalıştı)."
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
            print("=== TEST BİTTİ, VRAM BOŞALTILDI ===\n\n")

    print(f"    CALISMA ZAMANI => {time.time() - start_time}\n\n")


if __name__ == "__main__":
    test_ocr_extraction()
