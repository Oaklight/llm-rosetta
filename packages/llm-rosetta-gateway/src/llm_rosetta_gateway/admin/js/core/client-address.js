/**
 * client-address.js — the address the request log column shows.
 *
 * `client_ip` is what the gateway resolved under `server.trusted_proxies`; with
 * no trusted proxy configured that is the TCP peer, which behind a proxy is the
 * proxy itself.  When nothing trusted resolved a different address and the wire
 * carried an `X-Forwarded-For` chain, the first entry of the chain is the best
 * hint of who called, so it is shown, marked unverified: anyone can send that
 * header.  DOM-free, so node runs it directly (#1830).
 */

/** The first non-empty entry of an `X-Forwarded-For` value, or ''. */
export function firstForwarded(chain) {
  if (!chain) return '';
  const first = String(chain).split(',')[0];
  return first.trim();
}

/**
 * What a log row shows as its client.
 *
 * @param {{client_ip?: string, peer_ip?: string, forwarded_for?: string}} entry
 * @returns {{ip: string, unverified: boolean, peer: string, chain: string}}
 *   `ip` is '' when the row holds no address at all.
 */
export function clientAddress(entry) {
  const resolved = entry.client_ip || '';
  const peer = entry.peer_ip || '';
  const chain = entry.forwarded_for || '';
  const hint = firstForwarded(chain);
  // A row from before peer_ip existed has no peer to compare with, so its
  // resolved address stands: it cannot be told apart from a trusted one.
  const unresolved = peer !== '' && resolved === peer;
  if (unresolved && hint) {
    return { ip: hint, unverified: true, peer, chain };
  }
  return { ip: resolved, unverified: false, peer, chain };
}
