from riskvane.posture.collector import os_info


def test_parses_os_release(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'PRETTY_NAME="Ubuntu 22.04.3 LTS"\nNAME="Ubuntu"\nVERSION="22.04.3 LTS"\nVERSION_ID="22.04"\n'
    )

    info = os_info.collect(os_release_path=str(os_release))

    assert info.name == "Ubuntu 22.04.3 LTS"
    assert info.version == "22.04.3 LTS"
    assert info.kernel
    assert info.architecture


def test_missing_os_release_falls_back_to_platform(tmp_path):
    missing = tmp_path / "does-not-exist"

    info = os_info.collect(os_release_path=str(missing))

    assert info.name
    assert info.version
