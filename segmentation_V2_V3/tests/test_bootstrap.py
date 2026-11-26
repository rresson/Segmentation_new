"""用本地小 wheel 验证首次安装、失败重试和离线重启。"""
import os
from pathlib import Path
import shutil
import subprocess
import venv
from zipfile import ZipFile


def test_bootstrap_installs_before_cli_import_and_reuses_local_packages(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "test project"
    project.mkdir()
    for name in ("main.py", "pipeline.py", "bootstrap_dependencies.py"):
        shutil.copyfile(source / name, project / name)
    (project / "requirements-dev.txt").write_text("-r requirements.txt\n")
    (project / "requirements.txt").write_text("bootstrap-probe>=1,<2\n")
    package = project / "segmentation"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "cli.py").write_text(
        "from bootstrap_probe import value\n"
        "def main(argv, **kwargs): print(value)\n"
    )

    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "PYTHON"))}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_FIND_LINKS=wheels.as_uri(),
               PYTHONIOENCODING="utf-8")

    def run(entry):
        return subprocess.run([str(python), str(project / entry)], cwd=tmp_path, env=env,
                              capture_output=True, encoding="utf-8", timeout=120)

    failed = run("main.py")
    assert failed.returncode != 0 and "bootstrap-probe" in failed.stderr
    assert "ModuleNotFoundError" not in failed.stderr

    wheel = wheels / "bootstrap_probe-1.0-py3-none-any.whl"
    info = "bootstrap_probe-1.0.dist-info/"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("bootstrap_probe.py", "value = 'CLI ready'\n")
        archive.writestr(info + "METADATA", "Metadata-Version: 2.1\nName: bootstrap-probe\nVersion: 1.0\n")
        archive.writestr(info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        names = [*archive.namelist(), info + "RECORD"]
        archive.writestr(info + "RECORD", "".join(f"{name},,\n" for name in names))

    installed = run("main.py")
    assert installed.returncode == 0, installed.stderr
    assert "CLI ready" in installed.stdout
    wheel.unlink()
    reused = run("pipeline.py")
    assert reused.returncode == 0, reused.stderr
    assert reused.stdout.strip() == "CLI ready"

    (project / "requirements.txt").write_text("bootstrap-probe>=2\n")
    incompatible = run("pipeline.py")
    assert incompatible.returncode != 0 and "bootstrap-probe>=2" in incompatible.stderr
