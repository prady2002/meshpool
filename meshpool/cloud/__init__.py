"""MeshPool Cloud package — the hosted coordinator's global-connectivity layer.

The coordinator *is* the existing control plane, plus a reverse **tunnel** so that
agents anywhere (behind NAT/firewalls, no inbound ports) are reachable. Agents dial
out over ordinary HTTP; the coordinator pushes work down that agent-initiated
stream. See :mod:`meshpool.cloud.tunnel`.
"""
