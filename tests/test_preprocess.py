import unittest

from src.preprocess import normalize_plate_text


class PlateNormalizationTests(unittest.TestCase):
    def test_accepts_spaces_and_lowercase(self) -> None:
        self.assertEqual(normalize_plate_text("34 ea 2525"), "34EA2525")

    def test_accepts_dash_separated_plate(self) -> None:
        self.assertEqual(normalize_plate_text("34-GS-1905"), "34GS1905")

    def test_corrects_common_digit_letter_confusion(self) -> None:
        self.assertEqual(normalize_plate_text("I6 BEK 42O"), "16BEK420")

    def test_rejects_invalid_province(self) -> None:
        self.assertEqual(normalize_plate_text("99ABC123"), "")


if __name__ == "__main__":
    unittest.main()
