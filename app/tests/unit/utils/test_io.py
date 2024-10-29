import main.utils.io as module
import pytest
from tests.unit.test_utils import matrix_equal


def test_read_json_file(mocker):
    json_file_data = '{"example": "json"}'
    expected = {'example': 'json'}
    json_file = 'test.json'

    mock_open = mocker.patch('builtins.open',
                             mocker.mock_open(read_data=json_file_data))

    actual = module.read_json_file(json_file)

    assert actual == expected
    mock_open.assert_called_once_with(json_file, 'r')


@pytest.mark.parametrize(
    'expected, mode',
    [
        ('Fake data in a text file.', 'r'),
        (b'Fake binary data', 'rb')
    ])
def test_read_file(mocker, expected, mode):
    test_file = 'test.txt'

    mock_open = mocker.patch('builtins.open',
                             mocker.mock_open(read_data=expected))

    actual = module.read_file(test_file, mode)

    assert actual == expected
    mock_open.assert_called_once_with(test_file, mode)
    mock_open().read.assert_called_once()


def test_read_matrix_ark(mocker, fixed_signal):
    expected = fixed_signal

    mock_file = mocker.Mock()
    mock_read = mocker.patch.object(module.kaldi_io, 'read_mat_ark',
                                    return_value=[('key', fixed_signal)])

    actual = module.read_matrix_ark(mock_file)

    assert matrix_equal(actual, expected)
    mock_read.assert_called_once_with(mock_file)


def test_read_ark_file(mocker, fixed_signal):
    example_ark_file = 'example.ark'
    expected = fixed_signal

    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch.object(module, 'read_matrix_ark',
                        return_value=fixed_signal)

    actual = module.read_ark_file(example_ark_file)

    assert matrix_equal(actual, expected)
    mock_open.assert_called_once_with(example_ark_file, 'rb')
    module.read_matrix_ark.assert_called_once_with(mock_open())
