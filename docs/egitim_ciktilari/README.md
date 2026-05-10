# Eğitim Çıktıları

Bu klasörde Kaggle Notebook üzerinde eğitilen YOLO11n plaka tespit modelinin çıktıları bulunur.

## Özet Metrikler

- Eğitim ortamı: Kaggle Notebook
- GPU: 2 x Tesla T4
- Epoch: 30
- Eğitim süresi: yaklaşık 0.749 saat
- Validasyon mAP50: 0.9685
- Validasyon mAP50-95: 0.6872
- Test mAP50: 0.9693
- Test mAP50-95: 0.6893

## Dosyalar

- `results.csv`: epoch bazlı eğitim metrikleri
- `results.png`: eğitim grafiklerinin görsel özeti
- `confusion_matrix.png`: karışıklık matrisi
- `confusion_matrix_normalized.png`: normalize karışıklık matrisi
- `BoxPR_curve.png`: precision-recall eğrisi
- `val_batch0_pred.jpg`: örnek validasyon tahmini
- `args.yaml`: YOLO eğitim parametreleri

