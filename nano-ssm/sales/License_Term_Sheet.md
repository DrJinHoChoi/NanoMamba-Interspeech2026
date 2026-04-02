# NC-SSM Technology License Term Sheet

---

> **DISCLAIMER**: This document is a non-binding term sheet provided for informational and discussion purposes only. It does not constitute a legally binding agreement, offer, or commitment. Final licensing terms are subject to execution of a definitive license agreement between the parties. All pricing, terms, and conditions described herein are subject to change.

---

**Licensor:** Dr. Jin Ho Choi / NC-SSM Project
**Technology:** NC-SSM (Noise-Conditioned State Space Model)
**Date:** March 2026
**Status:** US and KR patent applications pending

---

## 1. Technology Overview

NC-SSM (Noise-Conditioned State Space Model) is an ultra-compact edge AI architecture purpose-built for keyword spotting and audio event detection on resource-constrained devices. By conditioning state space model dynamics on estimated noise characteristics, NC-SSM achieves noise-robust speech recognition performance that rivals models orders of magnitude larger.

### Key Specifications

| Metric | Range |
|---|---|
| Model Parameters | 7,443 -- 20,000 |
| Inference Latency | 7.1 -- 10.4 ms |
| Accuracy (Google Speech Commands v2) | 95.3 -- 96.4% |
| Noise Robustness | Built-in noise conditioning; maintains accuracy in adverse SNR conditions |
| Target Hardware | Cortex-M class MCUs, DSPs, low-power FPGAs, always-on AI accelerators |

### Deliverables

Licensed deliverables vary by tier (see Section 4) and may include pre-trained model weights, Python and C SDKs, training pipelines, RTL/Verilog source for hardware integration, technical documentation, and dedicated engineering support.

---

## 2. License Tiers

### Tier 1: Evaluation License -- Free

| | |
|---|---|
| **Fee** | No charge |
| **Duration** | 90 days (non-renewable without upgrade) |
| **Scope** | Internal evaluation and benchmarking only |
| **Deliverables** | Pre-trained model weights, basic documentation |
| **Production Use** | Not permitted |
| **Support** | Community forum |
| **Transferability** | Non-transferable |

---

### Tier 2: Developer License -- $500/month

| | |
|---|---|
| **Fee** | $500/month ($5,400/year with annual billing) |
| **Scope** | Development, prototyping, and limited production deployment |
| **Deliverables** | Full SDK (Python + C), training pipeline, documentation |
| **Custom Training** | Custom model training with licensee data |
| **Production Units** | Up to 10,000 units per year |
| **Support** | Email support (48-hour response) |
| **Transferability** | Non-transferable |

---

### Tier 3: Enterprise License -- $5,000/month

| | |
|---|---|
| **Fee** | $5,000/month ($54,000/year with annual billing) |
| **Scope** | Full commercial deployment, unlimited scale |
| **Deliverables** | Everything in Developer, plus source code access |
| **Custom Training** | Custom wake word training and model optimization |
| **Production Units** | Unlimited |
| **Support** | Priority support with SLA (4-hour response, dedicated contact) |
| **Source Code** | Full model source code and training infrastructure |
| **Transferability** | Transferable to wholly-owned subsidiaries |
| **Additional** | NDA required prior to execution |

---

### Tier 4: IP / Chip License -- Per-Unit Royalty

| | |
|---|---|
| **Fee** | $0.01 -- $0.05 per chip (volume-dependent, see schedule below) |
| **Minimum Annual Royalty** | $50,000 |
| **Scope** | Embedding NC-SSM IP into semiconductor products |
| **Deliverables** | RTL/Verilog source, integration documentation, verification testbenches |
| **Co-Development** | Joint engineering support for silicon integration |
| **Territory** | Exclusive territory options available upon negotiation |
| **Minimum Term** | 3 years |

**Per-Unit Royalty Schedule:**

| Annual Volume | Per-Chip Royalty |
|---|---|
| Up to 1M units | $0.05 |
| 1M -- 5M units | $0.03 |
| 5M -- 20M units | $0.02 |
| 20M+ units | $0.01 |

---

## 3. Custom Development Services

In addition to standard licensing, the following custom development engagements are available:

| Service | Price Range | Typical Timeline |
|---|---|---|
| Custom Wake Word Training | $10,000 -- $50,000 | 4 -- 8 weeks |
| Custom Model Architecture | $30,000 -- $100,000 | 8 -- 16 weeks |
| Edge AI Module Design | $50,000 -- $150,000 | 12 -- 24 weeks |
| Ongoing Maintenance & Updates | 15% of project value per year | Continuous |

All custom development projects include a statement of work (SOW), milestone-based delivery, and acceptance testing. Pricing depends on complexity, target hardware, vocabulary size, and performance requirements.

---

## 4. Deliverables by Tier

| Deliverable | Evaluation | Developer | Enterprise | IP/Chip |
|---|:---:|:---:|:---:|:---:|
| Pre-trained model weights | Yes | Yes | Yes | Yes |
| Technical documentation | Basic | Full | Full | Full |
| Python SDK | -- | Yes | Yes | Yes |
| C SDK (embedded) | -- | Yes | Yes | Yes |
| Training pipeline | -- | Yes | Yes | Yes |
| Custom model training | -- | Yes | Yes | Yes |
| Custom wake word training | -- | -- | Yes | Yes |
| Source code access | -- | -- | Yes | Yes |
| RTL/Verilog source | -- | -- | -- | Yes |
| Verification testbenches | -- | -- | -- | Yes |
| Community forum support | Yes | Yes | Yes | Yes |
| Email support | -- | Yes | Yes | Yes |
| Priority support + SLA | -- | -- | Yes | Yes |
| Dedicated engineering contact | -- | -- | Yes | Yes |
| Co-development support | -- | -- | -- | Yes |
| Production deployment rights | -- | 10K units/yr | Unlimited | Unlimited |

---

## 5. Payment Terms

| Term | Details |
|---|---|
| **Billing Cycle** | Monthly or annual (annual receives 10% discount) |
| **Payment Method** | Wire transfer or ACH |
| **Net Terms** | Net 30 for Enterprise and IP/Chip licensees; prepaid for Developer |
| **Royalty Reporting** | Quarterly unit shipment reports due within 30 days of quarter end |
| **Royalty Payment** | Quarterly, due within 45 days of quarter end |
| **Late Payment** | 1.5% per month on overdue balances |
| **Audit Rights** | Licensor reserves the right to audit royalty reports annually |

---

## 6. Intellectual Property & Confidentiality

- **Ownership.** All intellectual property rights in NC-SSM technology, including patents, trade secrets, and copyrights, remain the sole property of the Licensor. No transfer of ownership is implied or granted under any license tier.

- **Restrictions.** Licensee shall not reverse-engineer, decompile, or disassemble the licensed technology beyond what is expressly permitted by the applicable license tier.

- **Confidentiality.** A mutual non-disclosure agreement (NDA) is required for Tier 3 (Enterprise) and Tier 4 (IP/Chip) licenses prior to delivery of source code or RTL. All non-public technical information, pricing, and business terms are considered confidential.

- **Patent Indemnification.** Licensor provides patent indemnification for licensee's use of the technology within the scope of the granted license, subject to the terms of the definitive license agreement.

- **Patent Status.** US and KR patent applications are currently pending. License terms apply regardless of patent grant status.

---

## 7. Term & Termination

| License Tier | Initial Term | Renewal | Minimum Commitment |
|---|---|---|---|
| Evaluation | 90 days | N/A (upgrade required) | None |
| Developer | 1 year | Auto-renewal (annual) | None |
| Enterprise | 1 year | Auto-renewal (annual) | None |
| IP/Chip | 3 years | Negotiated renewal | 3 years |

- **Termination for Convenience.** Either party may terminate with 90 days written notice (Developer/Enterprise) or as specified in the definitive agreement (IP/Chip).

- **Termination for Breach.** Upon material breach, the non-breaching party shall provide written notice and a 30-day cure period. If the breach remains uncured, the agreement terminates automatically.

- **Effect of Termination.** Upon termination, licensee must cease all use of licensed technology, return or destroy all confidential materials, and certify compliance in writing.

- **Survival.** Provisions relating to intellectual property ownership, confidentiality, indemnification, limitation of liability, and audit rights shall survive termination.

---

## 8. Contact

**Dr. Jin Ho Choi**
NC-SSM Project -- Licensor

| | |
|---|---|
| **Email** | [licensing@nc-ssm.ai] |
| **Schedule a Call** | [Book a meeting -- Calendly link] |
| **Technical Inquiries** | [tech@nc-ssm.ai] |

---

*This term sheet is intended to outline proposed commercial terms and does not create any binding obligation on either party. Final terms will be set forth in a mutually executed definitive license agreement.*

*Last updated: March 2026*
