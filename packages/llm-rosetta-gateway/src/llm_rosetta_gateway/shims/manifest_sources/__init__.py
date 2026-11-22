"""Install shim vendors from a source's versioned ``manifest.json`` (#1660).

A shim source repository publishes ``dist/manifest.json`` with one record per
released vendor version (the format is
:mod:`llm_rosetta.contracts.shim_manifest`).  The gateway's ``shims.sources``
names the manifests to read and ``shims.install`` the vendors to take from
them.  Per selected vendor this module:

1. picks the newest version whose ``requires`` this core satisfies
   (:func:`~llm_rosetta.contracts.shim_manifest.select_release`) and logs a
   WARNING for each newer version it passed over;
2. keeps the installed version when it is still compatible and not older than
   that choice — a gateway upgrades only when a newer installable version
   appears, never because the manifest was re-read;
3. otherwise downloads ``<dist root>/<archive>``, checks its sha256 against
   the record and installs it into the shim cache atomically.

When the manifest cannot be fetched the last fetched copy is used, and when a
download fails the last good install is kept (marked ``stale``), the same
warm-cache behaviour any fetched source has.  A digest mismatch never installs.

Freshness guard: a manifest URL can answer 200 with an old copy (a CDN that
has not caught up with a publish).  When the URL that answered serves a
manifest whose ``generated_at`` is older than the cached one, or that lacks a
vendor the selectors expect (an explicit ``<source>:<vendor>``, a moved
vendor), the other URL is fetched once and
the manifest with the newer ``generated_at`` is used (the first on a tie).  A
cached manifest is never replaced by an older one: it is kept and marked
``stale`` until a fetch returns one at least as new.

Precedence (S3): core wins per shim — a shim name core already registers is
not loaded from the source, and the ignored names are logged with the source
and vendor; the vendor's other shims still install.  A vendor whose installable
record says ``promoted_to: core@X`` is not installed once this core is at
``X`` or later (a dev or pre-release of ``X`` is before it) and core really
registers the record's shims.

The release record is authoritative for what registers: an archive without a
``shims.yaml`` loads only the shim names its record lists.

Like any shim source, an installed vendor may ship a ``transforms.py`` that
runs on load: a manifest source is code execution at the trust level of the
repository that publishes it.
"""
