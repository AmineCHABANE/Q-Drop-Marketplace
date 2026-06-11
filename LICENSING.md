# How Q-Drop licenses are issued and verified

Q-Drop uses its own post-quantum signature module (`src/q_sign.py`) to issue
license keys. The system is **offline, serverless, and quantum-safe**: a
license is a hash-based WOTS signature over the buyer's details, verifiable
by anyone against the Merkle root published in [`LICENSE_ROOT.txt`](LICENSE_ROOT.txt).

This is dogfooding: the product sells post-quantum reference implementations,
and the licenses themselves are secured by one of them.

## The flow, end to end

```
 Buyer                      Stripe / PayPal                Owner
   │  pays €9.99 ──────────────►│                            │
   │                            │── receipt email ──────────►│
   │                            │                            │ python3 tools/license_manager.py \
   │                            │                            │     issue --email buyer@x.com --order ST-123
   │◄────────── license key (base64, ~3.5 KB) by email ──────│
   │
   │  verifies any time, offline, no server:
   │  python3 tools/license_manager.py verify --key <KEY>
   ▼
 VALID — post-quantum signature verified offline
   Email  : buyer@x.com
   Order  : ST-123
   Issued : 2026-06-11T22:12:32Z
```

## Why this design

| Property | How |
|----------|-----|
| **No license server** | Verification only needs `LICENSE_ROOT.txt` (in the repo) and Python 3. Nothing to host, nothing that can go down. |
| **No database** | The key itself contains the licensed email, order reference, and issue date — signed. |
| **Unforgeable, even by quantum computers** | WOTS + Merkle signatures reduce to SHA-256 preimage resistance. Shor's algorithm (see `src/q_shor.py`) breaks RSA/ECDSA license schemes; it does nothing here. |
| **Privacy-respecting** | No phone-home, no activation, no telemetry. The buyer's receipt + key are the whole system. |
| **Auditable** | The exact signing and verification code is public in `src/q_sign.py` — 250 commented lines. |

## Owner operations

```bash
# One-time setup — generates the signing keypair (4096 license capacity)
python3 tools/license_manager.py init
#   → tools/license_keys/private_state.json   (gitignored — NEVER commit)
#   → LICENSE_ROOT.txt                        (commit and publish)

# For each sale (email + order ref come from the Stripe/PayPal receipt)
python3 tools/license_manager.py issue --email buyer@example.com --order ST-12345
#   → prints the license key; email it to the buyer

# Check remaining capacity
python3 tools/license_manager.py status
```

When the 4096 one-time slots run out, `init --force` adds a new keypair and
appends the new root to `LICENSE_ROOT.txt`. Old licenses keep verifying
against the old root — every root in the file is checked.

**Critical invariant:** each WOTS leaf signs exactly once. The CLI persists
the leaf counter atomically *before* printing a key, so a crash can never
cause leaf reuse (which would compromise the keypair).

## Buyer verification

```bash
git clone https://github.com/AmineCHABANE/Q-Drop-Marketplace
cd Q-Drop-Marketplace
python3 tools/license_manager.py verify --key "<the key from your email>"
```

No dependencies, no network, no account. If it prints `VALID`, the license
was issued by the holder of the private key behind the published root.

## What the license grants

See [LICENSE](LICENSE) for the legal terms: free for study/research/evaluation,
€9.99 one-time for commercial use, covering all current and future modules.
The receipt from Stripe/PayPal remains valid proof of purchase on its own —
the signed key is an additional, cryptographic proof that survives email
provider changes, payment-platform shutdowns, and quantum computers.
