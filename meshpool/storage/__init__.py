"""Pooled distributed object store (spec §P3).

Turns spare disk across many nodes into one durable, content-addressed store:
objects are keyed by their SHA-256 digest, replicated across several nodes' donated
disk, served back from any surviving copy (bytes are integrity-checked), and
re-replicated automatically when a holding node drops.
"""
