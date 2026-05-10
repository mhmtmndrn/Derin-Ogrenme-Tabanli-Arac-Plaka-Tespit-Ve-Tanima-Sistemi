# Kaggle Demo Kullanımı

Yerelde Python bulunmadığı için gerçek YOLO + EasyOCR demosu Kaggle üzerinde çalıştırılabilir.

## Gerekli Dosyalar

- `C:\Users\Mehmet Emin\Downloads\plaka_projesi_ciktilar.zip`
- `C:\Users\Mehmet Emin\Desktop\Python Programlamaya Giriş\Ödev-Codex\samples\ornek_arac.jpg`
- `notebooks/plaka_demo_kaggle.ipynb`

## Adımlar

1. Kaggle'da yeni bir notebook oluştur.
2. Sağ panelden `Internet on` yap.
3. `Accelerator` kısmını GPU seç.
4. `notebooks/plaka_demo_kaggle.ipynb` dosyasını içeri aktar.
5. Sağ panelde `Input > Upload` ile `plaka_projesi_ciktilar.zip` dosyasını yükle.
6. Aynı şekilde `ornek_arac.jpg` dosyasını yükle.
7. Notebook hücrelerini sırayla çalıştır.
8. Son hücre `plaka_demo_outputs.zip` dosyasını oluşturur.
9. Bu zip dosyasını indir.

## Beklenen Çıktılar

- `demo_result.jpg`: plaka kutusu çizilmiş çıktı
- `ornek_arac_plate_crop.jpg`: kırpılmış plaka görseli
- `results.csv`: OCR sonucu, YOLO güven skoru ve durum bilgisi

