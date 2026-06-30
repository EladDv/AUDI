#!/usr/bin/env python3
"""Device-bound encryption helpers for AUDI payloads.

This is intended to stop a copied SD card or Docker image from running on a
different Pi. It is not a defense against root on the original running device.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

VERSION = 1
KDF_ITERATIONS = 250_000


class SecurePayloadError(RuntimeError):
    """Raised when a secure payload cannot be encrypted or decrypted."""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return None


def _cpuinfo_serial(root: Path) -> str | None:
    cpuinfo = _read_text(root / "proc/cpuinfo")
    if not cpuinfo:
        return None
    for line in cpuinfo.splitlines():
        if line.lower().startswith("serial"):
            _, value = line.split(":", 1)
            value = value.strip()
            if value and value != "0000000000000000":
                return value
    return None


def _network_macs(root: Path) -> list[str]:
    net_dir = root / "sys/class/net"
    ignored_prefixes = ("br-", "docker", "lo", "tailscale", "veth", "virbr")
    macs: list[str] = []
    try:
        interfaces = sorted(net_dir.iterdir())
    except OSError:
        return macs

    for iface in interfaces:
        name = iface.name
        if name.startswith(ignored_prefixes):
            continue
        mac = _read_text(iface / "address")
        if not mac:
            continue
        mac = mac.lower()
        if mac == "00:00:00:00:00:00":
            continue
        macs.append(f"{name}:{mac}")
    return macs


def device_fingerprint(root: Path = Path("/")) -> dict[str, Any]:
    """Return the local hardware identity used for device-bound encryption."""
    root = root.resolve()
    serial = (
        _read_text(root / "sys/firmware/devicetree/base/serial-number")
        or _cpuinfo_serial(root)
        or ""
    )
    model = (
        _read_text(root / "sys/firmware/devicetree/base/model")
        or _read_text(root / "proc/device-tree/model")
        or ""
    )
    macs = _network_macs(root)
    return {
        "version": VERSION,
        "serial": serial,
        "model": model,
        "macs": macs,
    }


def canonical_fingerprint(fingerprint: dict[str, Any]) -> bytes:
    return json.dumps(
        fingerprint,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_fingerprint(fingerprint)).hexdigest()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"), validate=True)


def _secret_bytes() -> bytes:
    # Optional local pepper. Leaving this unset still binds the payload to the
    # Pi fingerprint, but a TPM/USB-key/passphrase supplied here is stronger.
    return os.environ.get("AUDI_DEVICE_LOCK_SECRET", "").encode("utf-8")


def derive_key(
    fingerprint: dict[str, Any],
    salt: bytes,
    iterations: int = KDF_ITERATIONS,
) -> bytes:
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    material = canonical_fingerprint(fingerprint) + b"\0" + _secret_bytes()
    return PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    ).derive(material)


def encrypt_file(
    input_path: Path,
    out_dir: Path,
    fingerprint: dict[str, Any],
    *,
    source_name: str | None = None,
    kind: str = "audi-device-bound-file",
) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    input_path = input_path.resolve()
    source_name = source_name or input_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = derive_key(fingerprint, salt)
    plaintext = input_path.read_bytes()
    aad = canonical_fingerprint(fingerprint)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)

    enc_path = out_dir / f"{source_name}.enc"
    manifest_path = out_dir / f"{source_name}.enc.json"
    manifest = {
        "version": VERSION,
        "kind": kind,
        "algorithm": "AES-256-GCM",
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": KDF_ITERATIONS,
        "salt": _b64(salt),
        "nonce": _b64(nonce),
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "source_name": source_name,
        "ciphertext": enc_path.name,
    }
    enc_path.write_bytes(ciphertext)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return enc_path, manifest_path


def decrypt_file(
    manifest_path: Path,
    ciphertext_path: Path | None,
    output_path: Path,
    fingerprint: dict[str, Any],
) -> Path:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise SecurePayloadError(f"Unsupported secure payload version: {manifest}")
    actual_hash = fingerprint_hash(fingerprint)
    expected_hash = manifest.get("fingerprint_hash")
    if expected_hash != actual_hash:
        raise SecurePayloadError(
            "Device fingerprint mismatch; this payload is not bound to this Pi"
        )

    if ciphertext_path is None:
        ciphertext_path = manifest_path.parent / manifest["ciphertext"]
    salt = _unb64(manifest["salt"])
    nonce = _unb64(manifest["nonce"])
    key = derive_key(fingerprint, salt, int(manifest["iterations"]))
    aad = canonical_fingerprint(fingerprint)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext_path.read_bytes(), aad)
    except InvalidTag as exc:
        raise SecurePayloadError("Model decryption failed") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)
    output_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return output_path


def encrypt_model(
    model_path: Path,
    out_dir: Path,
    fingerprint: dict[str, Any],
) -> tuple[Path, Path]:
    return encrypt_file(
        model_path,
        out_dir,
        fingerprint,
        kind="audi-device-bound-model",
    )


def decrypt_model(
    manifest_path: Path,
    ciphertext_path: Path | None,
    output_path: Path,
    fingerprint: dict[str, Any],
) -> Path:
    return decrypt_file(manifest_path, ciphertext_path, output_path, fingerprint)


def _fingerprint_arg(value: str | None) -> dict[str, Any]:
    if value:
        return json.loads(value)
    return device_fingerprint()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUDI secure payload helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fp = sub.add_parser("fingerprint", help="Print this device fingerprint")
    fp.add_argument("--root", default="/", type=Path)

    enc = sub.add_parser("encrypt-model", help="Encrypt a model for one Pi")
    enc.add_argument("--model", required=True, type=Path)
    enc.add_argument("--out-dir", required=True, type=Path)
    enc.add_argument("--fingerprint-json")

    dec = sub.add_parser("decrypt-model", help="Decrypt a model on its bound Pi")
    dec.add_argument("--manifest", required=True, type=Path)
    dec.add_argument("--ciphertext", type=Path)
    dec.add_argument("--output", required=True, type=Path)
    dec.add_argument("--fingerprint-json")

    enc_file = sub.add_parser("encrypt-file", help="Encrypt any file for one Pi")
    enc_file.add_argument("--input", required=True, type=Path)
    enc_file.add_argument("--out-dir", required=True, type=Path)
    enc_file.add_argument("--source-name")
    enc_file.add_argument("--kind", default="audi-device-bound-file")
    enc_file.add_argument("--fingerprint-json")

    dec_file = sub.add_parser("decrypt-file", help="Decrypt any bound file")
    dec_file.add_argument("--manifest", required=True, type=Path)
    dec_file.add_argument("--ciphertext", type=Path)
    dec_file.add_argument("--output", required=True, type=Path)
    dec_file.add_argument("--fingerprint-json")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "fingerprint":
            print(json.dumps(device_fingerprint(args.root), sort_keys=True))
            return 0
        if args.cmd == "encrypt-model":
            enc_path, manifest_path = encrypt_model(
                args.model,
                args.out_dir,
                _fingerprint_arg(args.fingerprint_json),
            )
            print(f"encrypted_model={enc_path}")
            print(f"manifest={manifest_path}")
            return 0
        if args.cmd == "decrypt-model":
            output = decrypt_model(
                args.manifest,
                args.ciphertext,
                args.output,
                _fingerprint_arg(args.fingerprint_json),
            )
            print(f"decrypted_model={output}")
            return 0
        if args.cmd == "encrypt-file":
            enc_path, manifest_path = encrypt_file(
                args.input,
                args.out_dir,
                _fingerprint_arg(args.fingerprint_json),
                source_name=args.source_name,
                kind=args.kind,
            )
            print(f"encrypted_file={enc_path}")
            print(f"manifest={manifest_path}")
            return 0
        if args.cmd == "decrypt-file":
            output = decrypt_file(
                args.manifest,
                args.ciphertext,
                args.output,
                _fingerprint_arg(args.fingerprint_json),
            )
            print(f"decrypted_file={output}")
            return 0
    except SecurePayloadError as exc:
        print(f"secure payload error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
