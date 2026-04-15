import cv2
from ultralytics import YOLO
import easyocr
import pytesseract

# Görüntüyü yükle
image = cv2.imread("arac.jpg")

# Gri tona çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Basit threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR ile metin oku
text = pytesseract.image_to_string(thresh)

print("Okunan Plaka:", text)

# Görüntüyü göster
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
