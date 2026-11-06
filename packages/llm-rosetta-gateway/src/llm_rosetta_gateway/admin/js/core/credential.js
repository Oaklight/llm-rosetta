// Pure helpers for the shim-credential section of the provider editor.
//
// Kept out of `tabs/providers.js` so they can be tested without a DOM: both
// are places where a string from outside the panel becomes something the
// browser acts on.

/**
 * A URL safe to put in an href, or null.
 *
 * The usage page is declared by a shim, and a shim is trusted code — but this
 * link lives in the admin origin, where the session cookie is, so a
 * `javascript:` value must not reach an `href` at all.  ``https`` only: these
 * are vendor pages, so plain ``http`` buys nothing and the admin page itself
 * blocks it as mixed content anyway.
 */
export function safeLinkUrl(raw) {
  if (typeof raw !== 'string' || raw === '') return null;
  let url;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (url.protocol !== 'https:') return null;
  return url.href;
}

/**
 * Parse a pasted credential record, or null.
 *
 * The exchange owns the record's shape, so this only insists on the JSON
 * object the endpoint expects; anything else is refused before a request
 * leaves the page.
 */
export function parseRecord(text) {
  if (typeof text !== 'string') return null;
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value;
}

/**
 * Where the provider's credential comes from, in the editor's vocabulary.
 *
 * The builder applies this precedence — a command written into the provider
 * entry outranks whatever the shim declares — so the panel must apply the same
 * one.  A shim ships either an ``auth.py`` exchange or a command, never both.
 */
export function credentialSource(shim, cfg) {
  if (cfg && cfg.token_command) return 'config_command';
  if (shim && shim.credential_kind === 'command') return 'shim_command';
  if (shim && shim.mints_credential) return 'auth_exchange';
  // A provider whose shim no longer loads still has a live credential source;
  // the payload reports which one, so the section does not disappear.
  if (!shim && cfg && cfg.credential_source) return cfg.credential_source;
  if (!shim && cfg && cfg.mints_credential) return 'auth_exchange';
  return 'static_key';
}

/**
 * The four source cards' ids, and the credential sources they stand for.
 *
 * The editor's vocabulary is deliberately shorter than the loader's: the
 * loader says ``config_command`` because it names who wrote the command, and
 * the cards say ``config`` because the operator is choosing *where* the
 * credential comes from.  The mapping is total in both directions so a card
 * never has to guess.
 */
const UI_TO_SOURCE = {
  static: 'static_key',
  config: 'config_command',
  shim: 'shim_command',
  exchange: 'auth_exchange',
};
const SOURCE_TO_UI = {
  static_key: 'static',
  config_command: 'config',
  shim_command: 'shim',
  auth_exchange: 'exchange',
};

/** The loader's source id for a card id. */
export function credentialSourceId(ui) {
  return UI_TO_SOURCE[ui] || 'static_key';
}

/** The card id for a loader source id. */
export function credentialUiSource(source) {
  return SOURCE_TO_UI[source] || 'static';
}

/**
 * Which source cards the loader can actually honour for this provider.
 *
 * A card that is offered must produce a saved entry the loader accepts:
 *
 * - ``config`` is always available — an argv in the entry outranks whatever
 *   the shim declares, and needs no shim at all.
 * - ``shim`` exists only when the shim ships a command.
 * - ``exchange`` exists only when the shim mints a non-command credential.
 *   A stored exchange is offered even after its shim stops loading, so the
 *   card that describes the live credential is never hidden.
 * - ``static`` exists only when the shim mints nothing.  A key beside a
 *   command or an exchange is the one pair the loader refuses.  A keyless
 *   shim still gets the card: the operator may bring a paid key.
 */
export function availableCredentialSources(shim, cfg) {
  const mints = shim
    ? !!shim.mints_credential
    : !!(cfg && cfg.mints_credential);
  const live = (cfg && cfg.credential_source) || null;
  // The entry's own source says as much as the shim does: a provider whose
  // live credential is ``config_command`` mints from a command, so the
  // sign-in card must not be offered for it (there is no exchange to install
  // a record into).  The shim's own kind wins when there is one.
  const shimKind = (shim && shim.credential_kind) || null;
  const kind = shimKind
    || (live === 'shim_command' || live === 'config_command' ? 'command' : null);
  const exchangeLive = live === 'auth_exchange' || (!!mints && kind !== 'command');
  return {
    static: !mints,
    config: true,
    // The shim card needs a shim: a command in the entry is the config card.
    shim: !!shim && shimKind === 'command',
    exchange: exchangeLive,
  };
}

/** The card id to select when the preferred one is not available. */
export function credentialFallbackSource(available, preferred) {
  if (available && available[preferred]) return preferred;
  // Keep the credential the operator's entry actually uses, then prefer a
  // minting source over a static one: offering only "config" for a provider
  // whose shim signs in would describe a save that clears the sign-in.
  if (available && available.exchange) return 'exchange';
  return 'config';
}

/**
 * The fields a *source switch* writes, apart from the command itself.
 *
 * Moving off a static key removes it.  The flag is what the backend needs
 * because an empty ``api_key`` in the body has always meant "keep the stored
 * one"; deleting the key is a different instruction.  A source that keeps the
 * key — an exchange that takes it as input, a command that receives it through
 * ``token_command_env`` — must not send the flag.
 */
export function credentialSourceFields({ to, needsKey }) {
  const fields = {};
  if (to !== 'static_key' && !needsKey) fields.clear_api_key = true;
  return { fields };
}

/**
 * Whether the credential state deserves the Auth tab's badge.
 *
 * ``static`` has no refresh state of its own, so it never badges.  A provider
 * that has not been saved yet has nothing to report either — the Auth pane
 * says "save first" once opened.
 */
export function credentialNeedsAttention(ui, data, legacyMarker) {
  if (legacyMarker) return true;
  if (ui === 'static') return false;
  if (!data) return false;
  return ['needs_setup', 'unreadable', 'failing', 'error'].includes(data.state);
}

/**
 * Whether the API-key box must stay visible beside the credential section.
 *
 * A command that reads the operator's key from ``token_command_env`` takes it
 * as *input* — the key is not a second credential — so hiding the box would
 * leave the account key uneditable.  An exchange that declares
 * ``consumes_key`` (AskSage mints a token from the account key) is the same
 * case, and its key would otherwise be cleared by the source switch.
 */
export function credentialNeedsKey(source, shim, cfg) {
  if (source === 'auth_exchange') {
    return !!(shim && shim.consumes_key);
  }
  if (source !== 'config_command' && source !== 'shim_command') return false;
  return !!((cfg && cfg.token_command_env) || (shim && shim.default_token_command_env));
}

/**
 * Parse the argv field, or null when it is not a non-empty string array.
 *
 * The command is stored as an array — argv, not a shell line — so a value that
 * would be split differently by the shell is refused here rather than stored
 * and rejected on reload.
 */
export function parseCommandArgv(text) {
  if (typeof text !== 'string') return null;
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!Array.isArray(value) || value.length === 0) return null;
  if (!value.every((v) => typeof v === 'string' && v !== '')) return null;
  return value;
}

/**
 * The credential fields a save carries, or the refusal that stops it.
 *
 * ``loaded`` says the section read this provider's live credential.  When it
 * did not — the GET failed, or the editor moved on — the command box shows a
 * fallback, and an empty box does not mean "clear it": sending ``[]`` would
 * delete a command the operator never touched.  An interval is a default in
 * every minting source, so it is safe to send whether or not the read worked.
 *
 * Choosing a source other than ``config_command`` *is* a statement about the
 * command: the entry must not carry one, so a loaded section clears it.  Its
 * ``token_command_env`` is cleared only when ``clearCommandEnv`` says the
 * entry has one — an empty value written for a field the operator never saw
 * would override the shim's own env var.
 *
 * Returns ``{fields}`` or ``{error}``; the caller owns the message.
 */
export function credentialSaveFields({
  source, loaded, command, env, interval, clearCommandEnv = false,
}) {
  const fields = {};
  if (source === 'config_command') {
    const raw = (command || '').trim();
    if (raw !== '') {
      // The rows on screen *are* the command: an explicit argv is sent even
      // when the live credential could not be read.  Only the empty box is
      // ambiguous, and that is what ``loaded`` guards.
      const argv = parseCommandArgv(raw);
      if (argv === null) return { error: 'token_command_argv' };
      fields.token_command = argv;
      fields.token_command_env = (env || '').trim();
    } else if (loaded) {
      // Clearing it hands the credential back to the shim or to the key box.
      fields.token_command = [];
      fields.token_command_env = '';
    }
  } else if (loaded) {
    // A shim command, an exchange or a static key: the entry must not also
    // carry a command.  Choosing one of these is how the operator clears an
    // override, so the clear is explicit and needs no empty box.  A shim's
    // own argv is never copied into the entry — the loader resolves it.
    fields.token_command = [];
    // ``token_command_env`` is only cleared when the entry actually has one.
    // The loader treats a *present* key as the override, even when it is
    // empty, so writing ``""`` for a field the operator never saw would pop a
    // shim's own env var (AskSage's account-key input) instead of leaving it.
    if (clearCommandEnv) fields.token_command_env = '';
  }
  if (source !== 'static_key') {
    const value = parseInt((interval || '').trim(), 10);
    if (!Number.isFinite(value) || value < 60) {
      return { error: 'token_refresh_interval' };
    }
    fields.token_refresh_interval = value;
  }
  return { fields };
}

/**
 * Whether the section read *editingName*'s live credential.
 *
 * The state lives in three places — which provider the section is for, which
 * provider the drawer is editing, and whether the last read failed — and a
 * save must consult all three before it treats an empty command box as
 * "clear it".  Kept pure so the call site cannot quietly drop one of them.
 */
export function credentialLoadedFor(forName, editingName, data) {
  return !!forName && forName === editingName && !!data && data.state !== 'error';
}

/**
 * The short form of a credential digest: ``cmd:0123456789ab…``.
 *
 * Tokens never reach the panel, but their digest does — enough to tell one
 * credential from another, and enough to see that rotation changed it.
 */
export function shortCredential(token) {
  if (typeof token !== 'string' || token === '') return null;
  const sep = token.indexOf(':');
  if (sep <= 0 || sep === token.length - 1) return null;
  return `${token.slice(0, sep)}:${token.slice(sep + 1, sep + 13)}…`;
}
