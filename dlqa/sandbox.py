"""Run LLM-authored analysis code in a guarded subprocess.

"Good enough" isolation for a time-boxed contest (no Docker guarantee):
  - AST denylist for dangerous modules / os-sys calls / eval-exec
  - fresh subprocess, scrubbed env (no API keys), temp cwd
  - CPU-time rlimit + wall-clock timeout; address-space cap on Linux (Colab)
Reads are allowed (pandas needs them); the LLM is told to use absolute paths.
"""
import ast
import os
import subprocess
import sys
import tempfile

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None

BANNED_MODULES = {
    "subprocess", "socket", "shutil", "requests", "urllib", "http",
    "ctypes", "multiprocessing", "asyncio", "signal", "pty", "telnetlib",
}
# dangerous funcs, only when called as os.* / sys.* (so pandas df.rename/df.replace are fine)
OS_SYS_BANNED = {
    "system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename",
    "renames", "chmod", "chown", "kill", "fork", "spawnl", "spawnv", "spawnve",
    "execv", "execve", "execvp", "execvpe", "truncate", "link", "symlink",
    "mkfifo", "setuid", "setgid", "putenv", "_exit", "abort",
}
BANNED_NAMES = {"eval", "exec", "compile", "__import__"}


def check(code: str) -> None:
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    raise ValueError(f"banned import: {a.name}")
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] in BANNED_MODULES:
                raise ValueError(f"banned import: {n.module}")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in BANNED_NAMES:
                raise ValueError(f"banned call: {n.func.id}")
        elif isinstance(n, ast.Attribute):
            if (isinstance(n.value, ast.Name)
                    and n.value.id in {"os", "sys"}
                    and n.attr in OS_SYS_BANNED):
                raise ValueError(f"banned: {n.value.id}.{n.attr}")


def run(code: str, timeout: int = 25) -> dict:
    """Returns {ok, stdout, stderr}. ValueError only for a banned construct; a syntax
    error in the generated code is returned as a normal failure so the repair loop fixes it."""
    try:
        check(code)
    except SyntaxError as e:
        return {"ok": False, "stdout": "", "stderr": f"SyntaxError: {e}"}
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "snippet.py")
        with open(script, "w") as f:
            f.write(code)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": td, "TMPDIR": td,
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        }

        def _limits():
            if resource is None:
                return
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 2))
            except Exception:
                pass
            # RLIMIT_AS spuriously kills numpy/pandas on macOS (huge virtual maps); Linux only.
            if sys.platform.startswith("linux"):
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3,) * 2)
                except Exception:
                    pass

        try:
            p = subprocess.run(
                [sys.executable, script], cwd=td, env=env,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if resource is not None else None,
            )
            return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "stdout": e.stdout or "", "stderr": "TIMEOUT"}
