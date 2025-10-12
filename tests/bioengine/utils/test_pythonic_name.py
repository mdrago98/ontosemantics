
from unittest import TestCase
from parameterized import parameterized, param

from bioengine.utils.pythonic_name import convert_to_snake_case_from_camelcase, convert_to_snake_from_kebab_case, \
    get_pythonic_name


class TestGenModel(TestCase):

    def test_covert_to_snake_from_camel(self):
        result = convert_to_snake_case_from_camelcase('TestClass')
        assert result == 'test_class'

    def test_convert_to_snake_from_camel_with_underscore(self):
        result = convert_to_snake_case_from_camelcase('TestCase_2')
        assert result == 'test_case_2'

    def test_convert_to_snake_from_kebab_case(self):
        result = convert_to_snake_from_kebab_case('test-property')
        assert result == 'test_property'

    def test_convert_to_snake_from_kebab_case_with_camel_case_mix(self):
        result = convert_to_snake_from_kebab_case('TestCase-2')
        assert result == 'TestCase_2'

    @parameterized.expand([
        param(name='Test', expected='test'),
        param(name='Test-Case', expected='test_case'),
        param(name='testCase', expected='test_case'),
        param(name='test_Case', expected='test_case'),
        param(name='TestCase-Property', expected='test_case_property'),
        param(name='annotation-has_narrow_synonym', expected='annotation_has_narrow_synonym'),
        param(name='annotation-expand expression to', expected='annotation_expand_expression_to')
    ])
    def test_get_pythonic_name(self, name: str, expected: str):
        result = get_pythonic_name(name)
        assert result == expected
