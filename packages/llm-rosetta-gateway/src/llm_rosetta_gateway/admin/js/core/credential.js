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
 * Whether the API-key box must stay visible beside the credential section.
 *
 * A command that reads the operator's key from ``token_command_env`` takes it
 * as *input* — the key is not a second credential — so hiding the box would
 * leave the account key uneditable.
 */
export function credentialNeedsKey(source, shim, cfg) {
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
 * Returns ``{fields}`` or ``{error}``; the caller owns the message.
 */
export function credentialSaveFields({ source, loaded, command, env, interval }) {
  const fields = {};
  if (source === 'config_command' && loaded) {
    const raw = (command || '').trim();
    if (raw === '') {
      // Clearing it hands the credential back to the shim or to the key box.
      fields.token_command = [];
      fields.token_command_env = '';
    } else {
      const argv = parseCommandArgv(raw);
      if (argv === null) return { error: 'token_command_argv' };
      fields.token_command = argv;
      fields.token_command_env = (env || '').trim();
    }
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
