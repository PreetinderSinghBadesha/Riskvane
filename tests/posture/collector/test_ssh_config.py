from riskvane.posture.collector import ssh_config


def test_parses_tracked_directives(tmp_path):
    config = tmp_path / "sshd_config"
    config.write_text("PermitRootLogin yes\nPasswordAuthentication no\nProtocol 2\n")

    result = ssh_config.collect(sshd_config_path=str(config))

    assert result.permit_root_login == "yes"
    assert result.password_authentication == "no"
    assert result.protocol == "2"


def test_follows_include_directive(tmp_path):
    included = tmp_path / "hardening.conf"
    included.write_text("PermitRootLogin no\n")
    config = tmp_path / "sshd_config"
    config.write_text(f"Include {included.name}\nPasswordAuthentication yes\n")

    result = ssh_config.collect(sshd_config_path=str(config))

    assert result.permit_root_login == "no"
    assert result.password_authentication == "yes"


def test_missing_file_returns_empty_config(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = ssh_config.collect(sshd_config_path=str(missing))

    assert result.permit_root_login is None
    assert result.password_authentication is None
    assert result.protocol is None
