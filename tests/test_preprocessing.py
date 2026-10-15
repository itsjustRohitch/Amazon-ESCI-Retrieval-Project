"""
tests/test_preprocessing.py
---------------------------
Automated test suite for our scratch-built text preprocessing pipeline.
"""

import unittest
from src.preprocessing import clean_text, tokenize, remove_stopwords, stem_tokens, preprocess_text


class TestPreprocessingPipeline(unittest.TestCase):

    def test_clean_text_html_stripping(self):
        raw = "<div><b>Nike Air Zoom</b> Pegasus 38<br/></div>"
        cleaned = clean_text(raw)
        self.assertNotIn("<div>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("<br/>", cleaned)
        self.assertIn("nike air zoom pegasus 38", cleaned)

    def test_clean_text_entity_decoding(self):
        raw = "Men &amp; Women &quot;Sport&quot; Shoes"
        cleaned = clean_text(raw)
        self.assertNotIn("&amp;", cleaned)
        self.assertNotIn("&quot;", cleaned)
        self.assertEqual(cleaned, "men women sport shoes")

    def test_tokenize_preserves_alphanumeric_specs(self):
        cleaned = "apple ipad pro 12.9 inch 256gb wifi"
        tokens = tokenize(cleaned)
        self.assertIn("12.9", tokens)
        self.assertIn("256gb", tokens)
        self.assertIn("ipad", tokens)

    def test_remove_stopwords(self):
        tokens = ["the", "best", "headphones", "for", "running", "in", "the", "rain"]
        filtered = remove_stopwords(tokens)
        self.assertNotIn("the", filtered)
        self.assertNotIn("for", filtered)
        self.assertNotIn("in", filtered)
        self.assertIn("best", filtered)
        self.assertIn("headphones", filtered)
        self.assertIn("running", filtered)

    def test_stem_tokens(self):
        tokens = ["running", "shoes", "tested", "wirelessly"]
        stemmed = stem_tokens(tokens)
        self.assertEqual(stemmed, ["run", "shoe", "test", "wirelessli"])

    def test_full_pipeline_end_to_end(self):
        raw_query = "<p>Wireless <i>Bluetooth</i> Earbuds for Running!!!</p>"
        result = preprocess_text(raw_query, apply_stemming=True, return_tokens=True)
        self.assertListEqual(result, ["wireless", "bluetooth", "earbud", "run"])

    def test_empty_input(self):
        self.assertEqual(preprocess_text(None), [])
        self.assertEqual(preprocess_text(""), [])
        self.assertEqual(preprocess_text("   "), [])


if __name__ == "__main__":
    unittest.main()
