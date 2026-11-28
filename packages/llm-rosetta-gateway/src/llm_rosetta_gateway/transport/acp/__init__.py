"""ACP transport: a local agent CLI over stdio JSON-RPC.

* :mod:`.transport` — :class:`AcpTransport`, the lane's transport.
* :mod:`.chunks` — the body and event codec.
* :mod:`.ledger` — the session ledger.
* :mod:`.mcp` — the per-session loopback MCP bridge for a request's tools.
* :mod:`.meta_client` — the ACP client that advertises no host capability.
* :mod:`.profile` — the gateway-owned agent profile.
* :mod:`.quota` — what an agent says about its own allowance.
"""

from .transport import AcpTransport, AcpTurnError

__all__ = ["AcpTransport", "AcpTurnError"]
