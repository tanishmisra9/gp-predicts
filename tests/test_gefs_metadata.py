"""Metadata receipt boundaries; no network."""
from experiments.e220_gefs_metadata import before_cutoff


def test_receipt():
    row = dict(status=200, content_length="123", cutoff="2020-08-13T00:00:00+00:00",
               last_modified="Wed, 12 Aug 2020 18:00:00 GMT")
    assert before_cutoff(row)
    assert before_cutoff(row | {"last_modified": "Thu, 13 Aug 2020 00:00:00 GMT"})
    for change in ({"last_modified": "Thu, 13 Aug 2020 00:00:01 GMT"},
                   {"last_modified": "bad"}, {"last_modified": None},
                   {"last_modified": "Wed, 12 Aug 2020 18:00:00"},
                   {"content_length": "0"}, {"content_length": None},
                   {"status": 404}, {"status": 302}, {"status": 403}):
        assert not before_cutoff(row | change), change
    assert not before_cutoff({})


if __name__ == "__main__":
    test_receipt()
    print("GEFS metadata boundary tests ok")
