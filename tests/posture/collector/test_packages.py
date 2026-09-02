from riskvane.posture.collector import packages


def test_uses_dpkg_when_present():
    def fake_run(cmd):
        assert cmd[0] == "dpkg-query"
        return "bash 5.1-6ubuntu1\ncurl 7.81.0-1ubuntu1.15\n"

    def fake_which(name):
        return "/usr/bin/dpkg-query" if name == "dpkg-query" else None

    result = packages.collect(run=fake_run, which=fake_which)

    assert result[0].name == "bash"
    assert result[0].version == "5.1-6ubuntu1"
    assert len(result) == 2


def test_falls_back_to_rpm():
    def fake_run(cmd):
        assert cmd[0] == "rpm"
        return "curl 7.76.1-23.el9\n"

    def fake_which(name):
        return "/usr/bin/rpm" if name == "rpm" else None

    result = packages.collect(run=fake_run, which=fake_which)

    assert result == [packages.Package(name="curl", version="7.76.1-23.el9")]


def test_returns_empty_when_no_package_manager_found():
    result = packages.collect(run=lambda cmd: "", which=lambda name: None)

    assert result == []
