"""Local sign-in for the SurgProof server: one account on this machine, so a reviewer's name comes from a session.

The first visitor cannot simply claim the account: creating it needs a one-time code that only the person
who started the server can see, printed to the terminal and kept in a mode-600 file beside the account.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

SESSION_SECONDS = 12 * 3600
MIN_PASSWORD = 8
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60
COOKIE_NAME = "surgproof_session"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


class AuthError(Exception):
    """A refused auth request. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, maxmem=64 * 1024 * 1024, **_SCRYPT)


def _write_private(path: Path, text: str) -> None:
    """Mode 600 from the first byte: the file is never readable by others, even briefly."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


class LocalAuth:
    def __init__(self, folder: Path):
        self.account_path = Path(folder) / ".account.json"
        self.code_path = Path(folder) / ".setup_code"
        self._sessions: dict[str, tuple[str, float]] = {}
        self._failures: list[float] = []
        self._lock = threading.Lock()
        self._setup_lock = threading.Lock()

    def account(self) -> Optional[dict]:
        try:
            return json.loads(self.account_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def setup_code(self) -> Optional[str]:
        """The one-time code, created on first need. None once an account exists."""
        if self.account():
            return None
        try:
            return self.code_path.read_text(encoding="utf-8").strip()
        except OSError:
            code = secrets.token_urlsafe(9)
            _write_private(self.code_path, code)
            return code

    def _throttle(self) -> None:
        now = time.time()
        with self._lock:
            self._failures = [t for t in self._failures if now - t < LOCKOUT_SECONDS]
            if len(self._failures) >= MAX_FAILURES:
                raise AuthError(429, f"too many failed attempts, try again in {LOCKOUT_SECONDS} seconds")

    def _failed(self) -> None:
        with self._lock:
            self._failures.append(time.time())

    def _new_session(self, name: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._sessions = {k: v for k, v in self._sessions.items() if v[1] > now}
            self._sessions[hashlib.sha256(token.encode()).hexdigest()] = (name, now + SESSION_SECONDS)
        return token

    def setup(self, name: str, password: str, code: str) -> str:
        self._throttle()
        with self._setup_lock:  # two setups at once must not both pass the check below
            return self._create_account(name.strip(), password, code)

    def _create_account(self, name: str, password: str, code: str) -> str:
        if self.account_path.exists():
            raise AuthError(409, "an account already exists on this machine")
        expected = self.setup_code() or ""
        if not hmac.compare_digest(code.strip().encode(), expected.encode()):
            self._failed()
            raise AuthError(403, "that is not the setup code printed when the server started")
        if not 1 <= len(name) <= 80:
            raise AuthError(422, "give a name of 1 to 80 characters")
        if len(password) < MIN_PASSWORD:
            raise AuthError(422, f"use a password of at least {MIN_PASSWORD} characters")
        salt = secrets.token_bytes(16)
        _write_private(self.account_path, json.dumps({
            "name": name, "salt": salt.hex(), "hash": _hash_password(password, salt).hex(), "created": time.time()}))
        self.code_path.unlink(missing_ok=True)
        return self._new_session(name)

    def login(self, password: str) -> str:
        self._throttle()
        account = self.account()
        if account is None:
            raise AuthError(409, "no account yet: create one first")
        attempt = _hash_password(password, bytes.fromhex(account["salt"]))
        if not hmac.compare_digest(attempt, bytes.fromhex(account["hash"])):
            self._failed()
            raise AuthError(401, "wrong password")
        return self._new_session(account["name"])

    def user_for(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        key = hashlib.sha256(token.encode()).hexdigest()
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return None
            if entry[1] <= time.time():
                del self._sessions[key]
                return None
            return entry[0]

    def logout(self, token: Optional[str]) -> None:
        if token:
            with self._lock:
                self._sessions.pop(hashlib.sha256(token.encode()).hexdigest(), None)
