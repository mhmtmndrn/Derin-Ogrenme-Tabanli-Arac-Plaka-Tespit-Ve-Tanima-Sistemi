# Derin Ogrenme Tabanli Arac Plaka Tespit ve Okuma Sistemi

Bu proje, bir arac fotografi uzerinden plaka bolgesini YOLO tabanli nesne tespitiyle bulup, tespit edilen plaka goruntusunu EasyOCR ile okuyacak ilk prototipi icerir.

## Proje Yapisi

```text
.
├── notebooks/
│   └── plaka_model_egitimi_colab.ipynb
├── src/
│   ├── detector.py
│   ├── main.py
│   ├── ocr_reader.py
│   ├── preprocess.py
│   └── visualize.py
├── docs/
│   ├── akis_semasi.mmd
│   ├── rapor.md
│   └── teslim_kontrol_listesi.md
├── models/
│   └── README.md
├── samples/
│   └── README.md
└── outputs/
    ├── README.md
    └── results.csv
```

## Egitim Akisi

Model egitimi yerelde degil Google Colab veya Kaggle uzerinde yapilacak sekilde hazirlandi. Colab GPU limiti nedeniyle bu projedeki asil egitim Kaggle uzerinde tamamlandi.

1. `notebooks/plaka_model_egitimi_kaggle.ipynb` dosyasini Kaggle'a yukle.
2. Roboflow API anahtariyla `License Plate Recognition v11` veri setini indir.
3. `YOLO11n` modelini egit.
4. Egitimden sonra olusan `plaka_projesi_ciktilar.zip` dosyasini indir.
5. `best.pt` dosyasini bu projede `models/best.pt` olarak kaydet.

Tamamlanan egitim sonucu:

- Epoch: 30
- Validasyon mAP50: 0.9685
- Test mAP50: 0.9693
- Model dosyasi: `models/best.pt`

## Yerel Prototipi Calistirma

Python kurulu bir ortamda:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py --image samples/ornek_arac.jpg --weights models/best.pt --output outputs/demo_result.jpg --csv outputs/results.csv
```

Not: Bu bilgisayardaki mevcut terminalde `python` ve `pip` komutlari gorunmedigi icin yerel test burada calistirilamadi. Colab not defteri egitim icin ana calisma ortami olacak.

## Cikti

Program basarili calistiginda sunlari uretir:

- Plaka kutusu cizilmis islenmis gorsel
- Kirpilmis plaka gorseli
- Okunan plaka metni
- Tespit/OCR guven skorlarini iceren CSV tablo

## Rapor

Raporun Word/PDF'ye aktarilmaya hazir hali `docs/rapor.md` dosyasindadir. Word belgesi uretmek icin:

```powershell
node tools/create_report_docx.js
```

Bu komut `docs/rapor.docx` dosyasini olusturur.
