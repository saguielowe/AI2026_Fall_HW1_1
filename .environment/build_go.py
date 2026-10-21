"""Compile the provided Go extension using the local Visual Studio toolchain."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools._distutils.compilers.C.msvc import _get_vc_env

root = Path(__file__).resolve().parents[1]
vc_env = _get_vc_env("x86_amd64")
env = os.environ.copy()
for key in ("include", "lib", "libpath"):
    if key in vc_env:
        env[key.upper()] = vc_env[key]
tool_dirs = [
    Path(vc_env["vctoolsinstalldir"]) / "bin" / "Hostx64" / "x64",
    Path(vc_env["windowssdkverbinpath"]) / "x64",
]
env["PATH"] = os.pathsep.join(map(str, tool_dirs)) + os.pathsep + os.environ["PATH"]
env["DISTUTILS_USE_SDK"] = "1"
for tool in ("cl.exe", "link.exe", "rc.exe"):
    location = shutil.which(tool, path=env["PATH"])
    if not location:
        raise RuntimeError(f"Missing build tool: {tool}")
    print(f"{tool}: {location}", flush=True)
subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=root / "code" / "mcts" / "env" / "go",
    env=env,
    check=True,
)
