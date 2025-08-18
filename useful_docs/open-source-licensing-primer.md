# Open‑Source Licensing Primer

*What licenses are, the most common options on GitHub, and when & how to use them.*  
**Last updated:** Aug 18, 2025

---

**Summary:** An open‑source license is a permission set attached to your code that tells others what they may do (use, modify, redistribute), under what conditions (attribution, share‑alike), and with what limitations (e.g., no warranty). “Open source” follows the Open Source Initiative (OSI) definition—no field‑of‑use restrictions such as “non‑commercial” or “no AI use”.

> **Not open source:** Licenses with restrictions like “non‑commercial,” “no military use,” or “no AI use” are *source‑available* but not OSI‑approved open source.

---

## Table of contents

- [1) What is an open‑source license?](#1-what-is-an-open-source-license)
- [2) License families (the big picture)](#2-license-families-the-big-picture)
- [3) Common GitHub licenses at a glance](#3-common-github-licenses-at-a-glance)
- [4) How to choose (quick decision guide)](#4-how-to-choose-quick-decision-guide)
- [5) How to add a license on GitHub](#5-how-to-add-a-license-on-github)
- [6) Compatibility & gotchas](#6-compatibility--gotchas)
- [7) Quick “recipes”](#7-quick-recipes-for-typical-scenarios)
- [8) “When & how” in one minute](#8-when--how-in-one-minute)
- [Appendix: useful snippets](#appendix-useful-snippets)

---

## 1) What is an open‑source license?

An open‑source license grants permission to use, copy, modify, and redistribute your software. Each license spells out conditions (like attribution or share‑alike) and disclaimers (like “no warranty”). A project is considered “open source” when its license complies with the OSI’s criteria (free redistribution, access to source, no restrictions on fields of endeavor, etc.).

---

## 2) License families (the big picture)

- **Permissive** — Very few conditions; allow proprietary reuse.  
  *Examples:* **MIT**, **Apache‑2.0**, **BSD‑2‑Clause**, **BSD‑3‑Clause**.

- **Weak copyleft** — Share‑alike applies to specific files/libraries, not the entire combined work.  
  *Examples:* **MPL‑2.0** (file‑level), **LGPL‑3.0** (library‑level).

- **Strong copyleft** — Share‑alike applies to the whole combined work; AGPL includes network use.  
  *Examples:* **GPL‑3.0**, **AGPL‑3.0**.

- **Public‑domain equivalents** — No conditions; maximize reuse (commonly for data/snippets).  
  *Examples:* **Unlicense**, **CC0‑1.0** (better for data/assets than code).

---

## 3) Common GitHub licenses at a glance

| License (SPDX) | Type | What it requires | Good when you want… | Notes |
|---|---|---|---|---|
| **MIT** | Permissive | Preserve copyright & license notice. | Max adoption with minimal friction. | No explicit patent grant (often treated as implied). |
| **Apache‑2.0** | Permissive | Preserve notices; include `NOTICE` if used; state changes. | Permissive use with explicit patent grant & patent retaliation. | Corporate‑friendly due to clear patent terms. |
| **BSD‑2‑Clause** | Permissive | Preserve copyright & license notice. | MIT‑like with tiny differences. | Very close to MIT in practice. |
| **BSD‑3‑Clause** | Permissive | Same as BSD‑2; adds “no endorsement” clause. | MIT‑like with explicit non‑endorsement. | Common in academia/industry. |
| **MPL‑2.0** | Weak copyleft (file‑level) | Share source of *modified files*. | Keep changes to specific files open while allowing larger proprietary combinations. | Practical for plugins/extensions. |
| **LGPL‑3.0** | Weak copyleft (library) | Share changes to the library; dynamic linking allowed. | Libraries you want used broadly, but improved versions remain open. | Linking details matter; read carefully. |
| **GPL‑3.0** | Strong copyleft | Share source of the whole combined work when distributed. | Ensure derivatives remain open. | Compatible with Apache‑2.0. |
| **AGPL‑3.0** | Strong copyleft (network) | If users interact with your modified software over a network, offer the source. | Close the “SaaS”/network use loophole. | Common for server‑side apps and web services. |
| **Unlicense** | Public‑domain‑equivalent | No conditions. | Put code fully in the public domain. | CC0 is often preferred for non‑code assets. |
| **CC0‑1.0** | Public‑domain‑equivalent | No conditions. | Datasets, examples, small snippets. | Not recommended for software code; great for data. |

---

## 4) How to choose (quick decision guide)

1. **Lowest barrier to use?** → *MIT* or *BSD*. If patents are a concern (e.g., AI, crypto, networking), choose *Apache‑2.0* for its explicit patent grant.
2. **Want improvements to stay open (apps/tools)?** → *GPL‑3.0*. If the software will run as a web service and you want modified server code shared, pick *AGPL‑3.0*.
3. **Publishing a reusable library but don’t want to “GPL” everything that links to it?** → *LGPL‑3.0* (users can link proprietary code; changes to the library itself must be open).
4. **Want share‑alike only for files you change, not the entire repo?** → *MPL‑2.0* (file‑level copyleft).
5. **Data, pretrained weights, or examples** you want maximally reusable → *CC0‑1.0* (or *CC BY 4.0* if attribution is required; note CC licenses are for content, not code).

---

## 5) How to add a license on GitHub

### Via the web UI

1. Open your repository → **Add file ▸ Create new file** → name it `LICENSE` (or `LICENSE.md`).  
2. Click **Choose a license template**, pick one, review, then **Commit changes**.

### Best practices

- Put the full license text in `LICENSE`. *Apache‑2.0* may also use a `NOTICE` file if you have notices to carry forward.
- Add an SPDX header at the top of source files (see snippets below).
- Document the license in your `README` (“Licensed under …”).
- For contributions, consider a **DCO** (Developer Certificate of Origin, sign‑offs) or a **CLA** if your organization requires it.

---

## 6) Compatibility & gotchas

- **GPL compatibility:** Apache‑2.0 code can be included in GPLv3 projects (but not GPLv2‑only).
- **Linking to GPL:** If you link your code with GPL components and distribute the result, you likely must license the whole combined work under the GPL (details are nuanced).
- **Attribution & notices:** MIT/BSD require preserving copyright & license text; Apache‑2.0 also has NOTICE/attribution requirements.
- **Patents:** Apache‑2.0 grants explicit patent rights from contributors; MIT/BSD do not explicitly do so. In patent‑sensitive areas, Apache‑2.0 is often safer.

---

## 7) Quick “recipes” for typical scenarios

- **Research code / prototypes you want widely reused:** *MIT* (or *Apache‑2.0* if you want explicit patent terms).
- **Library intended for broad adoption but you want improvements to the library open:** *LGPL‑3.0*.
- **End‑user app/tool where you want derivatives to stay open:** *GPL‑3.0*.
- **Server software where you don’t want private SaaS forks:** *AGPL‑3.0*.
- **Datasets / examples / small snippets:** *CC0‑1.0* (or CC BY 4.0 if you require attribution).

---

## 8) “When & how” in one minute

1. **Decide your goal** (adoption vs. reciprocity vs. patents vs. network/SaaS use).
2. **Pick a license** from the table that matches that goal.
3. **Add `LICENSE`** (and `NOTICE` if Apache‑2.0) in your repo + optional SPDX headers in files.
4. **Document it** in your README.
5. **For contributions**, choose DCO or CLA as needed.

---

## Appendix: useful snippets

### 1) SPDX headers (copy into the first line of your source files)

Pick the license you chose; match the comment style to the language.

```
// SPDX-License-Identifier: Apache-2.0
```

```python
# SPDX-License-Identifier: MIT
```

```html
<!-- SPDX-License-Identifier: GPL-3.0-only -->
```

### 2) Apache‑2.0 NOTICE file (optional)

```
This product includes software developed by ACME Labs (https://example.com/).
Portions © 2025 ACME Labs and contributors. See the LICENSE file for details.
```

### 3) README blurb

```
## License
This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
```

---

*This guide is a general reference and not legal advice. For complex or high‑stakes situations, consult a lawyer.*
