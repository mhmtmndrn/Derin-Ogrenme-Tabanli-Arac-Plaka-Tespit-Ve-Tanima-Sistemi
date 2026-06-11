from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
import sys
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

from src.pipeline import SingleImagePlatePipeline, SingleImageResult


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATUS_LABELS = {
    "basarili": "Başarılı",
    "review_needed": "Kontrol gerekli",
    "ocr_okunamadi": "OCR okunamadı",
    "tespit_edilemedi": "Tespit edilemedi",
}


class UploadError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tek görsel plaka tespit arayüzü.")
    parser.add_argument("--host", default="127.0.0.1", help="Arayüzün dinleyeceği adres.")
    parser.add_argument("--port", type=int, default=7860, help="Arayüz portu.")
    parser.add_argument("--detector-weights", default="models/plate_detector.pt", help="Plaka tespit modeli.")
    parser.add_argument("--reader-weights", default="models/plate_reader.pt", help="Karakter/plaka okuma modeli.")
    parser.add_argument("--output-dir", default="outputs/web", help="Arayüz çıktılarının klasörü.")
    parser.add_argument("--conf", type=float, default=0.25, help="Plaka tespit güven eşiği.")
    parser.add_argument("--reader-conf", type=float, default=0.25, help="Karakter okuma güven eşiği.")
    parser.add_argument("--ocr-conf", type=float, default=0.50, help="Başarılı okuma için minimum güven.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tespit modeli girdi boyutu.")
    parser.add_argument("--reader-imgsz", type=int, default=256, help="Okuma modeli girdi boyutu.")
    parser.add_argument(
        "--reader-mode",
        choices=["hybrid", "easyocr", "yolo"],
        default="hybrid",
        help="Okuma modu.",
    )
    parser.add_argument("--easyocr-gpu", action="store_true", help="EasyOCR için GPU kullan.")
    parser.add_argument("--disable-easyocr", action="store_true", help="Hibrit modda EasyOCR fallback'i kapat.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SingleImagePlatePipeline(
        detector_weights=args.detector_weights,
        reader_weights=args.reader_weights,
        output_dir=output_dir,
        confidence=args.conf,
        reader_confidence=args.reader_conf,
        ocr_confidence=args.ocr_conf,
        image_size=args.imgsz,
        reader_image_size=args.reader_imgsz,
        reader_mode=args.reader_mode,
        easyocr_gpu=args.easyocr_gpu,
        disable_easyocr=args.disable_easyocr,
    )

    handler = make_handler(pipeline, output_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host_label = "127.0.0.1" if args.host in {"0.0.0.0", ""} else args.host
    print(f"Arayüz hazır: http://{host_label}:{args.port}")
    print("Durdurmak için Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArayüz durduruldu.")
    finally:
        server.server_close()
    return 0


def make_handler(pipeline: SingleImagePlatePipeline, output_dir: Path) -> type[BaseHTTPRequestHandler]:
    output_root = output_dir.resolve()

    class PlateAppHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_page(output_root=output_root))
                return
            if parsed.path.startswith("/files/"):
                self._send_file(parsed.path.removeprefix("/files/"), output_root)
                return
            self._send_html(render_page(error="Sayfa bulunamadı.", output_root=output_root), HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/process":
                self._send_html(render_page(error="Sayfa bulunamadı.", output_root=output_root), HTTPStatus.NOT_FOUND)
                return

            try:
                filename, payload = read_upload(self)
                upload_path = save_upload(filename, payload, output_root)
                result = pipeline.process(upload_path)
                self._send_html(render_page(result=result, output_root=output_root))
            except Exception as exc:
                message = str(exc) if isinstance(exc, UploadError) else f"İşlem tamamlanamadı: {exc}"
                self._send_html(render_page(error=message, output_root=output_root), HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, value: str, root: Path) -> None:
            try:
                relative = unquote(value).lstrip("/\\")
                target = (root / relative).resolve()
                target.relative_to(root)
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return

            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return PlateAppHandler


def read_upload(handler: BaseHTTPRequestHandler) -> tuple[str, bytes]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise UploadError("Lütfen bir görsel dosyası yükleyin.")

    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        raise UploadError("Yükleme boyutu okunamadı.")

    try:
        length = int(content_length)
    except ValueError as exc:
        raise UploadError("Yükleme boyutu geçersiz.") from exc

    if length <= 0:
        raise UploadError("Boş yükleme alındı.")
    if length > MAX_UPLOAD_BYTES:
        raise UploadError("Dosya çok büyük. En fazla 20 MB yükleyin.")

    body = handler.rfile.read(length)
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    if not message.is_multipart():
        raise UploadError("Yükleme formatı okunamadı.")

    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name != "image":
            continue
        filename = safe_filename(part.get_filename() or "upload.jpg")
        payload = part.get_payload(decode=True)
        if not payload:
            raise UploadError("Seçilen dosya boş görünüyor.")
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise UploadError(f"Desteklenmeyen dosya türü. Kabul edilenler: {allowed}")
        return filename, payload

    raise UploadError("Formda image alanı bulunamadı.")


def save_upload(filename: str, payload: bytes, output_root: Path) -> Path:
    upload_dir = output_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    upload_path = upload_dir / f"{run_id}_{filename}"
    upload_path.write_bytes(payload)
    return upload_path


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    cleaned_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not cleaned_stem:
        cleaned_stem = "upload"
    return f"{cleaned_stem}{suffix}"


def render_page(
    result: SingleImageResult | None = None,
    error: str = "",
    output_root: Path | None = None,
) -> str:
    result_html = render_result(result, output_root) if result is not None and output_root is not None else ""
    error_html = f'<div class="notice error">{escape(error)}</div>' if error else ""
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plaka Tespit Arayüzü</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d8dee6;
      --text: #18212f;
      --muted: #657386;
      --brand: #0f766e;
      --brand-dark: #115e59;
      --warning: #b45309;
      --danger: #b91c1c;
      --ok-bg: #ecfdf5;
      --warn-bg: #fff7ed;
      --error-bg: #fef2f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
    }}
    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .upload-panel, .result-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .upload-form {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    input[type="file"] {{
      width: 100%;
      min-height: 42px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      font-size: 15px;
    }}
    button {{
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--brand);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--brand-dark); }}
    .notice {{
      margin-top: 14px;
      border-radius: 6px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      font-weight: 700;
    }}
    .notice.error {{
      color: var(--danger);
      background: var(--error-bg);
      border-color: #fecaca;
    }}
    .result-panel {{
      margin-top: 18px;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(260px, .65fr);
      gap: 14px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    figure img {{
      display: block;
      width: 100%;
      max-height: 620px;
      object-fit: contain;
      background: #eef2f6;
    }}
    figcaption {{
      padding: 10px 12px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      font-size: 14px;
    }}
    .plate-text {{
      padding: 0 12px 12px;
      color: var(--text);
      font-size: 18px;
      font-weight: 700;
    }}
    .empty-crop {{
      display: grid;
      place-items: center;
      min-height: 180px;
      color: var(--muted);
      background: #f8fafc;
      padding: 20px;
      text-align: center;
    }}
    @media (max-width: 820px) {{
      header, .upload-form {{
        display: block;
      }}
      button {{
        width: 100%;
        margin-top: 10px;
      }}
      .image-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Tek Görsel Plaka Tespit Arayüzü</h1>
        <p class="subtitle">Bir araç görseli yükleyin; kırpılmış plaka, işaretli görsel ve plaka metni üretilsin.</p>
      </div>
    </header>
    <section class="upload-panel">
      <form class="upload-form" action="/process" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Görseli İşle</button>
      </form>
      {error_html}
    </section>
    {result_html}
  </main>
</body>
</html>
"""


def render_result(result: SingleImageResult, output_root: Path) -> str:
    plate = result.plate
    plate_message = f"Plaka okundu: {result.text}" if result.text else STATUS_LABELS.get(result.status, result.message)
    annotated_url = file_url(result.annotated_path, output_root)
    crop_html = render_crop(plate.crop_path, output_root, plate_message) if plate else render_empty_crop(plate_message)

    return f"""
    <section class="result-panel">
      <div class="image-grid">
        <figure>
          <img src="{escape(annotated_url)}" alt="Görsel üzerinde plaka sonucu">
          <figcaption>Görsel üzerinde çıktı</figcaption>
        </figure>
        {crop_html}
      </div>
    </section>
    """


def render_crop(path: Path, output_root: Path, plate_message: str) -> str:
    crop_url = file_url(path, output_root)
    return f"""
        <figure>
          <img src="{escape(crop_url)}" alt="Kırpılmış plaka">
          <figcaption>Kırpılmış plaka</figcaption>
          <div class="plate-text">{escape(plate_message)}</div>
        </figure>
    """


def render_empty_crop(plate_message: str) -> str:
    return f"""
        <figure>
          <div class="empty-crop">Plaka tespit edilemediği için kırpım oluşturulmadı.</div>
          <figcaption>Kırpılmış plaka</figcaption>
          <div class="plate-text">{escape(plate_message)}</div>
        </figure>
    """


def file_url(path: Path, output_root: Path) -> str:
    relative = path.resolve().relative_to(output_root).as_posix()
    return "/files/" + quote(relative)


if __name__ == "__main__":
    raise SystemExit(main())
