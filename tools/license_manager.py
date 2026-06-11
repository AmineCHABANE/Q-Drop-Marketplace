#!/usr/bin/env python3
"""
license_manager.py — Q-Drop license issuance & verification CLI

Licenses are post-quantum: each license key is a WOTS hash-based signature
(src/q_sign.py) over the buyer's details, verifiable OFFLINE against the
Merkle root published in this repository (LICENSE_ROOT.txt). No license
server, no database, no phone-home — and no quantum computer can forge one.

Workflow after a payment arrives (Stripe email / PayPal notification):

  One-time setup (repository owner):
      python3 tools/license_manager.py init
        → creates tools/license_keys/private_state.json  (NEVER commit)
        → writes LICENSE_ROOT.txt at the repo root        (commit this)

  For each sale:
      python3 tools/license_manager.py issue --email buyer@example.com \\
          --order ST-12345
        → prints a license key (base64). Email it to the buyer.

  Anyone, anywhere, offline:
      python3 tools/license_manager.py verify --key <KEY>
        → checks the signature against LICENSE_ROOT.txt and prints
          the licensed email, order reference and issue date.

Key capacity: height=12 → 4096 licenses per keypair. When exhausted, run
`init --force` for a new keypair and publish the new root alongside the old
(old licenses keep verifying against the old root, listed in LICENSE_ROOT.txt).
"""

import argparse
import base64
import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

import q_sign  # noqa: E402

KEYS_DIR = os.path.join(_HERE, "license_keys")
PRIVATE_STATE = os.path.join(KEYS_DIR, "private_state.json")
ROOT_FILE = os.path.join(_REPO, "LICENSE_ROOT.txt")

HEIGHT = 12          # 2^12 = 4096 one-time license slots
PRODUCT = "q-drop-commercial-license-v1"


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _load_signer() -> q_sign.MerkleSigner:
    if not os.path.exists(PRIVATE_STATE):
        sys.exit("No private key found. Run: python3 tools/license_manager.py init")
    with open(PRIVATE_STATE) as f:
        return q_sign.MerkleSigner.from_state(json.load(f))


def _save_signer(signer: q_sign.MerkleSigner) -> None:
    os.makedirs(KEYS_DIR, exist_ok=True)
    tmp = PRIVATE_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(signer.to_state(), f)
    os.replace(tmp, PRIVATE_STATE)   # atomic: a crash can't lose the index


def _published_roots() -> list:
    """All valid roots, newest first (older keypairs stay verifiable)."""
    if not os.path.exists(ROOT_FILE):
        return []
    roots = []
    with open(ROOT_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                roots.append(bytes.fromhex(line.split()[0]))
    return roots


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    if os.path.exists(PRIVATE_STATE) and not args.force:
        sys.exit(f"{PRIVATE_STATE} already exists. Use --force to add a new keypair.")

    print(f"Generating Merkle keypair (height={HEIGHT}, {1 << HEIGHT} licenses)...")
    signer = q_sign.MerkleSigner.generate(height=HEIGHT)
    _save_signer(signer)

    root_hex = signer.public_root.hex()
    stamp = datetime.date.today().isoformat()
    line = f"{root_hex}  # keypair created {stamp}, capacity {1 << HEIGHT}\n"
    with open(ROOT_FILE, "a") as f:
        f.write(line)

    print(f"Private state : {PRIVATE_STATE}  (keep secret, NEVER commit)")
    print(f"Public root   : {root_hex}")
    print(f"Appended to   : {ROOT_FILE}  (commit this file)")


def cmd_issue(args) -> None:
    signer = _load_signer()
    if signer.signatures_remaining == 0:
        sys.exit("Keypair exhausted. Run `init --force` to add a new keypair.")

    payload = {
        "product": PRODUCT,
        "email": args.email.strip().lower(),
        "order": args.order,
        "issued": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    message = json.dumps(payload, sort_keys=True).encode()
    sig = signer.sign(message)
    _save_signer(signer)               # persist the consumed index FIRST

    key = base64.b64encode(
        len(message).to_bytes(4, "big") + message + sig.to_bytes()
    ).decode()

    print("License issued. Send this key to the buyer:\n")
    print(key)
    print(f"\nLicensed to : {payload['email']}")
    print(f"Order ref   : {payload['order']}")
    print(f"Remaining   : {signer.signatures_remaining} licenses on this keypair")


def cmd_verify(args) -> None:
    try:
        blob = base64.b64decode(args.key)
        msg_len = int.from_bytes(blob[:4], "big")
        message = blob[4:4 + msg_len]
        sig = q_sign.Signature.from_bytes(blob[4 + msg_len:])
        payload = json.loads(message)
    except Exception:
        sys.exit("INVALID — key is malformed")

    roots = _published_roots()
    if not roots:
        sys.exit(f"No published roots found in {ROOT_FILE}")

    if not any(q_sign.verify(message, sig, root) for root in roots):
        sys.exit("INVALID — signature does not match any published root")

    print("VALID — post-quantum signature verified offline")
    print(f"  Product : {payload.get('product')}")
    print(f"  Email   : {payload.get('email')}")
    print(f"  Order   : {payload.get('order')}")
    print(f"  Issued  : {payload.get('issued')}")


def cmd_status(args) -> None:
    if os.path.exists(PRIVATE_STATE):
        signer = _load_signer()
        print(f"Keypair       : height={signer.height}, "
              f"{signer.signatures_remaining}/{signer.n_leaves} licenses remaining")
        print(f"Current root  : {signer.public_root.hex()}")
    else:
        print("No private keypair on this machine.")
    roots = _published_roots()
    print(f"Published roots in {os.path.basename(ROOT_FILE)}: {len(roots)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="generate a signing keypair")
    sp.add_argument("--force", action="store_true",
                    help="add a new keypair even if one exists")
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("issue", help="issue a license after a payment")
    sp.add_argument("--email", required=True, help="buyer email (from receipt)")
    sp.add_argument("--order", required=True, help="Stripe/PayPal order reference")
    sp.set_defaults(fn=cmd_issue)

    sp = sub.add_parser("verify", help="verify a license key offline")
    sp.add_argument("--key", required=True, help="the base64 license key")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("status", help="show keypair capacity and roots")
    sp.set_defaults(fn=cmd_status)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
