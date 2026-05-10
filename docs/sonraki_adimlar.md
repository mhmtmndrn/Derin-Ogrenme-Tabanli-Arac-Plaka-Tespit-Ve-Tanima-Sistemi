# Sonraki Adımlar

Bu dosya, raporun final PDF teslimine dönüşmesi için kalan zorunlu işleri listeler.

## Raporun Durumu

Rapor şu anda ödev belgesindeki başlıkları karşılayan hazır bir taslak durumundadır:

- Proje adı
- Proje konusu
- GitHub linki alanı
- Yöntem
- Algoritma akış şeması
- Uygulama tasarımı
- Başarı ölçütleri
- Teknik kısım
- Somut çıktı açıklaması
- APA kaynakça

Kaggle eğitimi tamamlandı, gerçek eğitim metrikleri rapora işlendi ve örnek araç fotoğrafı üzerinde demo çalıştırıldı. Final teslim için kalan ana iş, raporu Word üzerinden PDF'ye aktarmaktır.

## İndirilmesi veya Üretilmesi Gerekenler

1. `docs/rapor.docx` Word ile açılıp PDF olarak dışa aktarılacak.
2. GitHub public repo linki oluşturulursa rapordaki yer tutucu link gerçek linkle değiştirilecek.

## Demo Komutu

```powershell
python src/main.py --image samples/ornek_arac.jpg --weights models/best.pt --output outputs/demo_result.jpg --csv outputs/results.csv
```

## Yerel Bilgisayarda Eksik Görülenler

Bu ortamda şu komutlar bulunamadı:

- `python`
- `pip`
- `git`

Bu nedenle eğitim ve prototipin çalıştırılması için en pratik yol Google Colab'dir. Yerelde çalıştırmak istersek önce Python kurulumu gerekir.
