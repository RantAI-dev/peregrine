"""A small execution sandbox for HumanEval+ (subprocess, wall-clock and resource limits).

HumanEval+ scores generated Python by running the dataset's own tests against it.  Tests
are trusted (they come from the pinned dataset); generated code is not.  Each candidate
runs in its own process with:

* a clean, isolated interpreter (``-I``: no user site, no ``PYTHONPATH``, no env),
* a fresh working directory,
* ``RLIMIT_CPU`` / ``RLIMIT_AS`` / ``RLIMIT_FSIZE`` / ``RLIMIT_NOFILE`` / ``RLIMIT_CORE``,
* a wall-clock timeout that kills the process group,
* an audit hook that denies socket and subprocess events (the host does not allow
  unprivileged network namespaces, so this is the available no-network guard).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

PREAMBLE = r"""
import sys

_BLOCKED = (
    "socket.__new__",
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.sethostname",
    "urllib.Request",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.fork",
    "ctypes.dlopen",
    "ctypes.dlsym",
)


def _guard(event, args):
    if event in _BLOCKED:
        raise PermissionError("sandbox: %s is not allowed" % event)


sys.addaudithook(_guard)
"""

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The first fenced code block, else the text from its first Python construct."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    match = re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+)", text)
    if match:
        return text[match.start():]
    return text


def _limits(timeout: float):
    import resource

    def apply():
        cpu = int(timeout) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    return apply


def run_python(code: str, *, python: str, timeout: float = 10.0, workdir: str | None = None) -> dict:
    """Run ``code`` as a fresh program and report how it ended."""
    tmp = tempfile.mkdtemp(prefix="pg-acc-sandbox-", dir=workdir)
    path = os.path.join(tmp, "main.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE)
        f.write(code)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
    }
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [python, "-I", path],
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limits(timeout),
        )
        return {
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "duration_s": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "returncode": None,
            "stdout": (exc.stdout or b"")[-2000:].decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")[-2000:],
            "stderr": "timeout",
            "duration_s": time.monotonic() - started,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_humaneval(candidate_code: str, test_code: str, entry_point: str, *, python: str, timeout: float = 10.0) -> dict:
    """Run the HumanEval+ test function against the candidate's ``entry_point``."""
    program = (
        candidate_code
        + "\n\n"
        + test_code
        + "\n\n"
        + f"check({entry_point})\n"
        + "print('SANDBOX-PASS')\n"
    )
    result = run_python(program, python=python, timeout=timeout)
    result["passed"] = result["passed"] and "SANDBOX-PASS" in result.get("stdout", "")
    return result
