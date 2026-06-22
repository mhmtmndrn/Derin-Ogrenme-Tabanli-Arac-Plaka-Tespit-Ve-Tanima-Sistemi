# Derin Ogrenme Tabanli Arac Plaka Tespit ve Tanima Sistemi

Bu proje, arac gorsellerindeki plakayi YOLO tabanli modelle tespit eder ve plaka metnini YOLO karakter okuyucu ile EasyOCR destekli hibrit akistan gecirerek okur.

## Proje Yapisi

```text
.
|-- arabalar/                 # Toplu islenecek arac gorselleri
|-- docs/                     # Rapor, akis semasi ve teslim notlari
|-- models/
|   |-- plate_detector.pt     # Plaka tespit modeli
|   `-- plate_reader.pt       # Karakter/plaka okuma modeli
|-- outputs/                  # Program ciktilari
|-- samples/                  # Örnek araç görselleri
|-- src/                      # Tespit, OCR, on isleme ve gorsellestirme kodlari
|-- tests/                    # Birim testleri
|-- run_all_images.py         # Tum gorselleri isleyen ana script
`-- requirements.txt
```
## Kullanılan Veri Setleri
Plaka Tespit Modeli İçin: https://www.kaggle.com/code/mercantl/turkish-lisence-plate-yolov8/input
Plaka Okuma Modeli İçin: https://github.com/ramajoballester/UC3M-LP

## Kurulum
Powershell uygulamasını açın.Aşağıdaki komutları girin.
```powershell
git clone https://github.com/mhmtmndrn/Derin-Ogrenme-Tabanli-Arac-Plaka-Tespit-Ve-Tanima-Sistemi.git
cd Derin-Ogrenme-Tabanli-Arac-Plaka-Tespit-Ve-Tanima-Sistemi
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Bu komutları girdikten sonra tarayıcı üzerinden " http://127.0.0.1:7860/ " adresine gidebilirsiniz.

## Toplu Çalıştırma

Tüm görsellerde denemek için kurulumu yaptıktan sonra aşağıdaki komutu girin.

```powershell
python run_all_images.py
```

Acik parametrelerle calistirmak icin:

```powershell
python run_all_images.py --images arabalar --detector-weights models/plate_detector.pt --reader-weights models/plate_reader.pt --output-dir outputs --reader-mode hybrid
```

`--reader-mode` degeri `hybrid`, `easyocr` veya `yolo` olabilir. Varsayilan `hybrid` modunda once YOLO karakter okuyucu denenir, dusuk guvenli veya temizlenemeyen okumalarda EasyOCR destegi kullanilir.

## Ciktilar

Program calistiginda `outputs/` altinda su dosyalar uretilir:

- `all_results.csv`: tum okuma sonuclari
- `index.html`: sonuc galerisi
- `contact_sheet.jpg`: toplu ozet gorsel
- `annotated/`: plaka kutulari islenmis gorseller
- `crops/`: kirpilmis plaka gorselleri

## Test

```powershell
python -m unittest discover -s tests
```

## Egitim Notlari

Egitim ve Kaggle/Colab sureci icin `notebooks/` ve `docs/` klasorlerine bakilabilir. Bu kopyada calisan yerel modeller `models/plate_detector.pt` ve `models/plate_reader.pt` dosyalaridir.
