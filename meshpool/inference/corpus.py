"""An original, copyright-free training corpus for the demo model.

Hand-written MeshPool-themed prose (plus a few templated variations to add volume
and teach sentence structure). All text here is original to this project, so the
trained checkpoint carries no third-party content.
"""
from __future__ import annotations

BASE = """
MeshPool turns the idle machines you already own into one pooled computer.
A home has a laptop, an old desktop, and a spare mini pc, and most of the time they sit unused.
MeshPool gathers their spare cores, memory, and storage into a single friendly pool.
You install a small agent on each machine, and in a few minutes they appear live in the dashboard.
The first machine becomes the control plane, and the others join with a short pairing code.
There is no port forwarding, no manual network setup, and no confusing configuration.
When a machine goes to sleep, the pool notices and keeps your work running on the others.

With MeshPool you can host a website across several machines with automatic failover.
You point at your app, choose how many copies to run, and the pool spreads them across nodes.
A load balancer sends each request to a healthy copy, so one machine failing drops nothing.
If a node drops, the pool heals itself and starts a fresh copy on a machine that is still awake.

You can run a big batch job that uses the spare cpu of every machine at once.
The job is a list of small independent units, and the pool sends each unit to a node with room.
If a node dies in the middle, its unfinished units are retried somewhere else, so nothing is lost.
The work spreads by fair share, and the faster machines simply take on more of the units.

The flagship feature is running one large model that is too big for any single machine.
MeshPool splits the model into contiguous blocks of layers and places one block on each node.
Only the small hidden state travels between machines, so a home network is fast enough.
The first stage holds the embedding, the middle stages hold the deep layers, the last stage holds the head.
When you chat with the model, each token flows through the pipeline of machines and back to you.
The dashboard reports the real speed in tokens per second, with no invented numbers.
If a machine leaves, MeshPool re-partitions the model across the survivors and keeps serving.
Because every node rebuilds the exact same weights, the pooled answer matches a single machine exactly.

Every machine keeps a donation policy that you control.
You can cap the cpu and memory you share, and choose a schedule such as idle only or outside work hours.
The limits are enforced by the operating system, never by trusting the workload to behave.
A prominent kill switch reclaims your resources instantly, with no approval and no waiting.
A contribution ledger records how much each machine donated and consumed, building simple trust.
You can see that you contributed many compute hours this week, and the pool rewards steady helpers.

MeshPool is private by default, for machines that one person or one team already owns.
It is not an anonymous public market, so it trades heavy proof for a light and fast trust model.
Workloads run sandboxed and isolated, and cannot scan or touch the rest of your home network.
Secrets are brokered briefly and are never written in plain text on a worker.
Everything is designed to feel like magic, to be friendly, and to never break under a failure.
"""

TEMPLATES = [
    "MeshPool pools {a} and {b} from the machines you already own.",
    "When a node {x}, the pool {y} and your work keeps running.",
    "You can {do} across several machines with automatic failover.",
    "The pool sends each {unit} to a machine with room to spare.",
    "Only the {small} travels between machines, so the network stays fast.",
]
FILL = {
    "a": ["spare cores", "idle memory", "unused storage", "spare cpu", "free capacity"],
    "b": ["idle memory", "spare cores", "spare gpu time", "unused disk", "quiet machines"],
    "x": ["sleeps", "drops", "goes offline", "is reclaimed", "loses power"],
    "y": ["heals itself", "reroutes traffic", "retries the work", "re-partitions the model",
          "starts a fresh copy"],
    "do": ["host a website", "run a batch job", "serve a large model", "deploy an app",
           "share compute"],
    "unit": ["work unit", "task", "request", "job piece", "token"],
    "small": ["hidden state", "activation tensor", "small tensor", "intermediate result"],
}


def build_corpus(repeat: int = 6, seed: int = 7) -> str:
    import random
    rng = random.Random(seed)
    parts = [BASE.strip()]
    for _ in range(repeat):
        # a shuffled pass of the base sentences keeps word statistics while adding variety
        sents = [s.strip() for s in BASE.strip().split("\n") if s.strip()]
        rng.shuffle(sents)
        parts.append(" ".join(sents))
        for _ in range(12):
            t = rng.choice(TEMPLATES)
            filled = t.format(**{k: rng.choice(v) for k, v in FILL.items()})
            parts.append(filled)
    return "\n".join(parts) + "\n"
