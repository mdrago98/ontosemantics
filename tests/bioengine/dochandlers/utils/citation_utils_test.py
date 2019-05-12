from unittest import TestCase

from parameterized import parameterized, param

from utils.citation_utils import strip_citations


class TestCitationUtils(TestCase):

    @parameterized.expand([
        param(name='Test[1]', expected='Test'),
        param(name='Test[1], Test', expected='Test , Test'),
        param(name='Test(1), Test', expected='Test , Test'),
        param(name='Test, Test groups in 1985 ( REFS 5,6)', expected='Test, Test groups in 1985 '),
        param(name='GLUT4 ( REF. 203)', expected='GLUT4'),
        param(name='Receptor process (Noy et al., 2009)', expected='Receptor process'),
    ])
    def test_strip_citations(self, name: str, expected: str):
        result = strip_citations(name)
        assert result.strip() == expected.strip()
