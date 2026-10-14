"""Packaging oracles inspect the deliverable construction, not helper functions."""
import pathlib,re,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Review(unittest.TestCase):
    def test_A3_notice_inventory(self):
        text=(ROOT/'THIRD_PARTY_NOTICES.md').read_text()
        for required in ('Niels Lohmann','Copyright (c) 2003, The Regents of the University of California','## lmcpp','libquadmath'):
            with self.subTest(required=required):self.assertTrue(required in text,required)
    def test_A4_toolchain(self):
        text=(ROOT/'docker/Dockerfile').read_text()
        self.assertIn('snapshot.debian.org',text)
        self.assertRegex(text,r'cmake=[0-9]')
        self.assertRegex(text,r'nlohmann-json3-dev=[0-9]')
        self.assertIn('dpkg-query',text)
    def test_A5_layers(self):
        text=(ROOT/'docker/Dockerfile').read_text()
        self.assertRegex(text,r'FROM \$\{BASE\} AS runtime')
        self.assertRegex(text,r'COPY --from=pruned .*site-packages')
    def test_OF_caps_format(self):
        text=(ROOT/'README.md').read_text()
        self.assertIn('447',text);self.assertIn('shortest',text)
        self.assertNotIn('Finite\ndoubles use 17 significant digits',text)
    def test_F11_recipe(self):self.assertIn('"simulated"',(ROOT/'examples/solve.lms').read_text())
    def test_F7_success_last(self):
        text=(ROOT/'tests/s0.py').read_text().rstrip()
        self.assertTrue(text.endswith("contracts passed')"))
if __name__=='__main__':unittest.main()
