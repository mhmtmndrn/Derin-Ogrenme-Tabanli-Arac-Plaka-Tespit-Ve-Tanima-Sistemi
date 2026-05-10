# GitHub Branch Stratejisi

Repo: `mhmtmndrn/Derin-Ogrenme-Tabanli-Arac-Plaka-Tespit-Ve-Tanima-Sistemi`

Doğrudan GitHub branch oluşturma denemesi GitHub entegrasyonu yetkisi nedeniyle `403 Resource not accessible by integration` hatası verdi. Bu nedenle dosyalar aşağıdaki branch paketlerine ayrıldı.

## 1. `codex/core-prototype`

Amaç: Çalışan Python prototipi ve komut satırı akışı.

İçerik:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/`
- `models/README.md`
- `samples/README.md`
- `outputs/README.md`
- `tools/create_report_docx.js`

Commit mesajı önerisi:

```text
Add plate detection and OCR prototype
```

## 2. `codex/training-workflow`

Amaç: Colab/Kaggle eğitim ve demo notebookları.

İçerik:

- `notebooks/`
- `docs/kaggle_kullanim.md`
- `docs/kaggle_demo_kullanim.md`

Commit mesajı önerisi:

```text
Add Kaggle and Colab training notebooks
```

## 3. `codex/report-results`

Amaç: Ödev raporu, akış şeması, eğitim çıktıları ve demo çıktıları.

İçerik:

- `docs/`
- `outputs/`
- `samples/ornek_arac.jpg`
- `models/best.pt`

Commit mesajı önerisi:

```text
Add report, training metrics, and demo outputs
```

## Not

`models/best.pt`, `.jpg`, `.png` ve `.docx` gibi binary dosyalar GitHub API aracılığıyla bu oturumda doğrudan yüklenemedi. Git veya GitHub Desktop ile branchlere eklenebilirler.

