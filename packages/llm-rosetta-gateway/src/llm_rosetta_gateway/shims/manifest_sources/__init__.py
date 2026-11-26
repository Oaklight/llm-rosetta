"""Install extensions from a source's published ``index.json`` (#1704).

An extension repository publishes ``index.json`` and its wheels as release
assets (the format is :mod:`llm_rosetta.extensions`).  The gateway's
``shims.sources`` names the indexes to read and ``shims.install`` the
extensions to take from them.  Per selected extension this package:

1. picks the newest release this interpreter and core can run
   (:func:`.releases.select_release`) and logs a WARNING for each newer one it
   passed over;
2. keeps the installed version when it is still compatible and not older than
   that choice: a gateway upgrades only when a newer installable version
   appears, never because the index was re-read;
3. otherwise downloads the wheel from its ``url``, checks its sha256 against
   the entry and installs it into the extensions directory with core's
   standard-library installer (:func:`llm_rosetta.api.install_wheel`).  The
   installed extension is then an ordinary extension: the registry rebuild's
   ``load_providers()`` loads it, and its load result is the vendor's status.

When the index cannot be fetched the last fetched copy is used, and when a
download fails the last good install is kept (marked ``stale``).  A digest
mismatch never installs.

Freshness guard: an index whose ``generated_at`` is older than the cached
one never replaces it.  The cached copy is kept and marked ``stale`` until a
fetch returns one at least as new.  A source may declare a second URL
(``fallback``); when the URL that answered serves an older copy, or lacks an
extension the selectors expect (an explicit ``<source>:<vendor>``, a moved
vendor), the other URL is fetched once and the index with the newer
``generated_at`` is used.

Precedence: core wins.  A shim name core already registers is not replaced,
and the loader refuses an extension whose declared names another owner holds
(the status says which).  An extension all of whose shims are held is skipped
before it is downloaded.

An installed extension loads whether or not ``shims.install`` still selects
it; the selection decides what is installed and kept current,
``shims.disable`` what is switched off.  An extension's
``requires.extensions`` that are not installed are reported in the vendor's
``note``, not installed on its behalf.

An installed extension may ship code that runs on load: an index is code
execution at the trust level of the repository that publishes it.
"""
