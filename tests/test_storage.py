from unittest.mock import MagicMock, patch
from src import storage

def test_download_bytes_reads_object_body():
    fake_body = MagicMock()
    fake_body.read.return_value = b'{"hello": 1}'
    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {"Body": fake_body}
    with patch.object(storage, "_s3", return_value=fake_s3):
        data = storage.download_bytes("some/key.json")
    assert data == b'{"hello": 1}'
    fake_s3.get_object.assert_called_once()
