#!/usr/bin/env python3
"""Inventory audit: recompute the repo's headline numbers and check that the
documents still agree with them.

Doc counts rot silently — this session found a summary table that said
"not started" for a family whose own section two hundred lines later said
"complete", and a handoff whose kernel/runner counts predated two batches.
This script makes those failures loud and is wired into `make test`.

Checks:
  1. every python/run_*.py is wired into the Makefile test target, and every
     Makefile runner exists;
  2. every sim/*.cpp mirror is referenced by some runner;
  3. the headline numbers in docs/RIG_HANDOFF_BRINGUP.md §0 match reality
     (kernel total, per-family split, runner and mirror counts);
  4. no kernel inside a quarantined *_rig.cl claims // ALG-VERIFIED for
     itself (references to other files' verified kernels are fine).

Run: python3 lint/audit_inventory.py
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kernel_lines(path):
    """Kernels are counted by declaration lines starting with `kernel`/
    `__kernel` — the same rule PERF_NOTES used, so counts stay comparable."""
    with open(path, encoding='utf-8', errors='replace') as f:
        return [l for l in f if re.match(r'^(__)?kernel\b', l)]


def main():
    os.chdir(ROOT)
    problems = []

    cl_files = sorted(glob.glob('src/opencl/*/kernels/*.cl'))
    per_family = {}
    total = 0
    for f in cl_files:
        n = len(kernel_lines(f))
        per_family[f.replace('\\', '/').split('/')[2]] = per_family.get(f.replace('\\', '/').split('/')[2], 0) + n
        total += n

    runners = sorted(glob.glob('python/run_*.py'))
    mirrors = sorted(glob.glob('sim/*.cpp'))

    # 1. wiring
    mk = open('Makefile', encoding='utf-8', errors='replace').read()
    wired = set(re.findall(r'^\tpython3 (python/\S+)', mk, re.M))
    for r in runners:
        if r.replace('\\', '/') not in wired:
            problems.append(f"runner not wired into make test: {r}")
    for r in sorted(wired):
        if not os.path.exists(r.replace('/', '\\')):
            problems.append(f"Makefile runs a missing runner: {r}")

    # 2. mirrors referenced
    referenced = set()
    for r in runners:
        t = open(r, encoding='utf-8', errors='replace').read()
        for m in re.findall(r'"([\w]+\.cpp)"', t):
            referenced.add('sim/' + m)
    for s in mirrors:
        if s.replace('\\', '/') not in referenced:
            problems.append(f"mirror never referenced by a runner: {s}")

    # 3. handoff headline numbers
    hb = 'docs/RIG_HANDOFF_BRINGUP.md'
    if os.path.exists(hb):
        t = open(hb, encoding='utf-8', errors='replace').read()
        want = [
            (rf'\*\*{total} OpenCL kernels\*\*', f'kernel total {total}'),
            (rf'\*\*{len(runners)} verification runners\*\*',
             f'runner count {len(runners)}'),
            (rf'\*\*{len(mirrors)} CPU mirrors\*\*',
             f'mirror count {len(mirrors)}'),
        ]
        for pat, what in want:
            if not re.search(pat, t):
                problems.append(f"{hb}: stale or missing {what}")
        for fam, n in sorted(per_family.items()):
            if not re.search(rf'{fam} {n}\b', t):
                problems.append(f"{hb}: stale per-family count for {fam} "
                                f"(actual {n})")
    else:
        problems.append(f"missing {hb}")

    # 4. quarantine markers
    for f in sorted(glob.glob('src/opencl/*/kernels/*_rig.cl')):
        lines = open(f, encoding='utf-8', errors='replace').readlines()
        for i, l in enumerate(lines):
            if re.match(r'^(__)?kernel\b', l):
                # look back a few lines for a self-claim
                back = ''.join(lines[max(0, i - 6):i])
                if re.search(r'//\s*ALG-VERIFIED', back):
                    problems.append(
                        f"{f}:{i+1}: quarantined kernel claims ALG-VERIFIED")

    print(f"inventory: {total} kernels "
          f"({', '.join(f'{k} {v}' for k, v in sorted(per_family.items()))}), "
          f"{len(runners)} runners, {len(mirrors)} mirrors")
    if problems:
        print(f"audit: {len(problems)} PROBLEM(S)")
        for p in problems:
            print("  !", p)
        return 1
    print("audit: OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
