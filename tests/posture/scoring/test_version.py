from riskvane.posture.scoring import version


def test_parses_dpkg_style_version():
    assert version.parse("5.1-6ubuntu1") == (5, 1)


def test_parses_rpm_style_version():
    assert version.parse("7.76.1-23.el9") == (7, 76, 1)


def test_parse_returns_empty_tuple_for_unparseable_version():
    assert version.parse("unknown") == ()


def test_is_older_than_true_case():
    assert version.is_older_than("1.1.1m", "1.1.1n") is True


def test_is_older_than_false_when_equal_or_newer():
    assert version.is_older_than("1.1.1n", "1.1.1n") is False
    assert version.is_older_than("1.1.2", "1.1.1n") is False


def test_is_older_than_false_when_unparseable():
    assert version.is_older_than("unknown", "1.1.1n") is False
