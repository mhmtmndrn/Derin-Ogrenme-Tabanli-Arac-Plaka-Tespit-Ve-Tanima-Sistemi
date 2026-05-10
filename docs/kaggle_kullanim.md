# Kaggle ile Eğitim

Colab GPU limiti dolduğunda en pratik alternatif Kaggle Notebook kullanmaktır.

## Neden Kaggle?

- Ücretsiz GPU/TPU desteği vardır.
- Notebook mantığı Colab'a benzerdir.
- Eğitim çıktıları `/kaggle/working` altında ziplenip indirilebilir.
- Bu proje için ayrıca `notebooks/plaka_model_egitimi_kaggle.ipynb` dosyası hazırlandı.

## Kullanım Adımları

1. https://www.kaggle.com adresinden hesap aç.
2. Gerekirse telefon doğrulamasını tamamla.
3. `Code > New Notebook` oluştur.
4. Sağ panelden `Settings` aç.
5. `Accelerator` alanından GPU seç.
6. `Internet` ayarını aç.
7. `File > Import Notebook` veya notebook yükleme alanından şu dosyayı yükle:

```text
notebooks/plaka_model_egitimi_kaggle.ipynb
```

8. Hücreleri sırayla çalıştır.
9. Eğitim bitince son hücre `plaka_projesi_ciktilar.zip` oluşturur.
10. Sağ paneldeki `Output` kısmından zip dosyasını indir.

## Önemli Not

Kaggle'da eğitim çıktılarını kaybetmemek için eğitimden sonra mutlaka `Save Version` yapılmalı ve oluşan zip dosyası indirilmelidir.

