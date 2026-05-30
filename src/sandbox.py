"""Sandboxed execution of untrusted, model-generated Python (RLVR reward).

SECURITY NOTE
-------------
This runs code the *model* wrote. We shrink the blast radius with:
  - a separate subprocess per candidate (a crash/segfault can't take the
    trainer down),
  - OS resource limits (CPU time, address space, file size) via ``setrlimit``,
  - a "reliability guard" preamble that neuters the most destructive stdlib
    calls (os.system, os.remove, shutil.rmtree, subprocess.Popen, ...),
  - a wall-clock timeout.

This is **not** a real security boundary (no network/namespace isolation). Run
it only on a disposable training box. For stronger isolation, wrap the call in a
container / nsjail / firejail.

Two test protocols are supported, matching the TACO/APPS format:
  - **stdin/stdout** (default): feed ``inputs[i]`` to the program's stdin and
    compare its stdout to ``outputs[i]``.
  - **call-based** (``fn_name`` present): import the candidate, call
    ``fn_name(*args)`` (or ``Solution().fn_name(*args)``) and compare the return
    value to the expected output.
"""
import json
import os
import resource  # POSIX-only; the remote A100 box (and macOS dev box) both have it
import subprocess
import sys
import tempfile
import textwrap

# Prepended to every candidate program. Not an f-string: the braces below are
# literal Python. Mirrors the spirit of HumanEval's reliability_guard, trimmed.
RELIABILITY_GUARD = textwrap.dedent(
    """
    import os as _os, sys as _sys, builtins as _bi
    try:
        import faulthandler as _fh; _fh.disable()
    except Exception:
        pass
    _BLOCK = (
        "system", "remove", "removedirs", "rmdir", "unlink", "rename", "replace",
        "kill", "killpg", "putenv", "unsetenv", "chmod", "chown", "fork",
        "forkpty", "setuid", "truncate",
    )
    for _n in _BLOCK:
        if hasattr(_os, _n):
            try:
                setattr(_os, _n, None)
            except Exception:
                pass
    try:
        import shutil as _sh
        _sh.rmtree = _sh.move = _sh.chown = None
    except Exception:
        pass
    try:
        import subprocess as _sp
        _sp.Popen = _sp.run = _sp.call = None
    except Exception:
        pass
    _bi.exit = _bi.quit = None
    _sys.setrecursionlimit(100000)
    """
)


def _limits(mem_mb, cpu_s):
    """Return a POSIX ``preexec_fn`` capping memory / CPU time / file size."""

    def _apply():
        for res, soft in (
            (resource.RLIMIT_AS, mem_mb * 1024 * 1024),
            (resource.RLIMIT_CPU, cpu_s),
            (resource.RLIMIT_FSIZE, 16 * 1024 * 1024),
        ):
            try:
                resource.setrlimit(res, (soft, soft))
            except Exception:
                pass

    return _apply


def run_program(code, stdin_text="", timeout=6.0, mem_mb=1024):
    """Run ``code`` as a script, feeding ``stdin_text`` on stdin.

    Returns ``(ok, stdout, err)``; ``ok`` is True only on a clean (exit 0) run.
    """
    source = RELIABILITY_GUARD + "\n" + (code or "")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sol.py")
        with open(path, "w") as fh:
            fh.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=env,
                preexec_fn=_limits(mem_mb, int(timeout) + 1),
            )
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
        except Exception as exc:  # noqa: BLE001
            return False, "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, proc.stdout, (proc.stderr or f"exit {proc.returncode}")
    return True, proc.stdout, ""


def _norm_lines(s):
    text = (s or "").replace("\r\n", "\n").strip("\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def outputs_match(actual, expected):
    """Lenient stdout comparison: line-wise (trailing ws ignored), else token-wise."""
    if _norm_lines(actual) == _norm_lines(expected):
        return True
    return (actual or "").split() == (expected or "").split()


def _to_stdin(x):
    if isinstance(x, str):
        return x if x.endswith("\n") else x + "\n"
    if isinstance(x, (list, tuple)):
        return "\n".join(str(e) for e in x) + "\n"
    return str(x) + "\n"


def _to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(str(e) for e in x)
    return str(x)


def _check_call_based(code, fn_name, args, expected, timeout, mem_mb):
    """Call ``fn_name(*args)`` in a subprocess; compare its return to ``expected``."""
    if not isinstance(args, (list, tuple)):
        args = [args]
    harness = (code or "") + textwrap.dedent(
        f"""
        import json as _json, sys as _sys
        _payload = _json.loads(_sys.stdin.read())
        try:
            _fn = {fn_name}
        except NameError:
            try:
                _fn = Solution().{fn_name}
            except Exception:
                print("__NO_FN__"); _sys.exit(0)
        _res = _fn(*_payload)
        print("__RESULT__" + _json.dumps(_res, default=str))
        """
    )
    ok, out, _ = run_program(harness, json.dumps(list(args)), timeout, mem_mb)
    if not ok:
        return False
    line = ""
    for ln in (out or "").splitlines():
        if ln.startswith("__RESULT__"):
            line = ln[len("__RESULT__"):]
    if not line:
        return False
    try:
        got = json.loads(line)
    except Exception:  # noqa: BLE001
        return False
    exp = expected
    if isinstance(expected, list) and len(expected) == 1:
        exp = expected[0]
    try:
        if json.dumps(got, sort_keys=True) == json.dumps(exp, sort_keys=True):
            return True
    except Exception:  # noqa: BLE001
        pass
    return str(got) == str(exp)


def check_io(code, tests, timeout=6.0, mem_mb=1024, max_cases=20):
    """Run ``code`` against ``tests``; return the fraction of cases passed in [0, 1].

    ``tests`` = ``{"inputs": [...], "outputs": [...], "fn_name": <optional>}``.
    Fails closed: any error / timeout / mismatch counts as a failed case.
    """
    if not code or not isinstance(tests, dict):
        return 0.0
    inputs = tests.get("inputs") or []
    outputs = tests.get("outputs") or []
    fn_name = tests.get("fn_name")
    n = min(len(inputs), len(outputs))
    if n == 0:
        return 0.0
    if max_cases and max_cases > 0:
        n = min(n, max_cases)
    passed = 0
    for i in range(n):
        try:
            if fn_name:
                ok = _check_call_based(
                    code, fn_name, inputs[i], outputs[i], timeout, mem_mb
                )
            else:
                ran, out, _ = run_program(code, _to_stdin(inputs[i]), timeout, mem_mb)
                ok = ran and outputs_match(out, _to_text(outputs[i]))
        except Exception:  # noqa: BLE001
            ok = False
        passed += 1 if ok else 0
    return passed / n
