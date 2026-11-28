"""显式启动脚本时，按当前工程的清单准备当前 Python 环境。"""
from contextlib import closing
import hashlib
from importlib import invalidate_caches
from importlib.util import find_spec
import json
import os
from pathlib import Path
import site
import sqlite3
import subprocess
import sys
import tempfile


def ensure_dependencies():
    requirements = Path(__file__).resolve().with_name("requirements-dev.txt")
    # 两个工程可共用一个环境。并发启动期间，仅允许一个进程检查和安装。
    environment = os.path.normcase(os.path.realpath(sys.prefix))
    key = hashlib.sha256(os.fsencode(environment + str(Path.home()))).hexdigest()
    lock = Path(tempfile.gettempdir()) / f"segmentation-pip-{key}.sqlite3"
    with closing(sqlite3.connect(lock, timeout=3600)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if find_spec("pip") is None:
                print("正在为当前 Python 环境安装 pip…", flush=True)
                subprocess.run([sys.executable, "-m", "ensurepip"], check=True)

            pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-input", "install"]
            manifest = ["-r", str(requirements)]
            # 让 pip 校验版本、环境标记及传递依赖。环境就绪时可离线启动。
            with tempfile.TemporaryDirectory() as temp:
                report = Path(temp) / "report.json"
                check = subprocess.run(
                    [*pip, "--dry-run", "--no-index", "--quiet", "--report", str(report), *manifest],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                needs_install = check.returncode != 0 or bool(json.loads(report.read_bytes())["install"])

            if needs_install:
                print("正在安装工程所需的 Python 依赖…", flush=True)
                subprocess.run([*pip, *manifest], check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"依赖安装失败（退出码 {error.returncode}）。请检查上方 pip 输出、软件源及当前环境的写入权限，"
                f"然后使用 {sys.executable} 重新启动。"
            ) from error

    # 首次安装到用户目录时，该目录可能在本解释器启动后才被创建。
    user_site = site.getusersitepackages()
    if site.ENABLE_USER_SITE and os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    invalidate_caches()
