import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// The provider drawer has three panes: what the provider is and how to reach it
// (Basic), what it serves and how each endpoint is addressed (Capabilities),
// and where its credential comes from (Auth).  These assertions pin the split:
// a section that drifts back into Basic makes the pane long again without any
// test failing on behaviour, because the form keeps working either way.

const here = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(
  here,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);

function code(path) {
  const src = readFileSync(join(ADMIN, path), 'utf8');
  let out = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    const next = src[i + 1];
    if (ch === '/' && next === '/') {
      const end = src.indexOf('\n', i);
      i = end === -1 ? src.length : end;
      continue;
    }
    if (ch === '/' && next === '*') {
      const end = src.indexOf('*/', i + 2);
      i = end === -1 ? src.length : end + 2;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      const quote = ch;
      out += ch;
      i += 1;
      while (i < src.length) {
        out += src[i];
        if (src[i] === '\\') { out += src[i + 1] ?? ''; i += 2; continue; }
        if (src[i] === quote) { i += 1; break; }
        i += 1;
      }
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');
const providers = code('js/tabs/providers.js');
const i18n = readFileSync(join(ADMIN, 'js/core/i18n.js'), 'utf8');

test('the drawer carries three panes, each named by its tab', () => {
  assert.match(
    html,
    /id="provTabCaps" role="tab" aria-selected="false" aria-controls="provPaneCaps" onclick="setProviderPane\('caps'\)"/,
  );
  assert.match(
    html,
    /<div class="prov-pane" id="provPaneCaps" role="tabpanel" aria-labelledby="provTabCaps" hidden>/,
  );
  // The count tells an operator how many are ticked without opening the pane,
  // which is the one thing a hidden form cannot do.
  assert.match(html, /id="provCapsCount" hidden/);
});

test('the endpoint settings live in the Capabilities pane, not in Basic', () => {
  const basic = html.slice(
    html.indexOf('id="provPaneBasic"'),
    html.indexOf('id="provPaneCaps"'),
  );
  const caps = html.slice(
    html.indexOf('id="provPaneCaps"'),
    html.indexOf('id="provPaneAuth"'),
  );

  assert.ok(caps.includes('id="provCapChecks"'), 'the ticks are in Caps');
  assert.ok(caps.includes('id="provLlmSection"'), 'the LLM endpoint is in Caps');
  assert.ok(caps.includes('id="provNonLlmSections"'), 'the typed sections are in Caps');
  assert.ok(!basic.includes('id="provCapChecks"'), 'the ticks left Basic');
  assert.ok(!basic.includes('id="provLlmSection"'), 'the LLM endpoint left Basic');
  assert.ok(!basic.includes('id="provNonLlmSections"'), 'the typed sections left Basic');
  // Basic keeps a read-only summary of the same fact, like its credential card.
  assert.ok(basic.includes('id="provCapsChips"'), 'Basic summarises the ticks');
});

test('the pane switch knows all three panes', () => {
  assert.match(
    providers,
    /const _PROVIDER_PANES = \['basic', 'caps', 'auth'\];/,
  );
  const body = oneFunction(providers, 'setProviderPane');
  for (const id of ['provPaneBasic', 'provPaneCaps', 'provPaneAuth']) {
    assert.ok(body.includes(id), `${id} is toggled`);
  }
  assert.ok(body.includes("['provTabCaps', _provPane === 'caps']"));
});

/** One function's own text: a later function's line must not satisfy an
 *  assertion about this one. */
function oneFunction(src, name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} was not found`);
  const end = src.indexOf('\n}\n', start);
  return src.slice(start, end === -1 ? src.length : end);
}

test('the summary and the ticks share one visual language', () => {
  // The summary renders the card's own badges, so a capability keeps the hue
  // it wears everywhere else.  A grey icon read as "disabled" on a provider
  // that serves.
  const body = oneFunction(providers, '_renderCapsSummary');
  assert.ok(body.includes('_faceSummaryRow'), 'one row per ticked face');
  assert.ok(
    oneFunction(providers, '_faceSummaryRow').includes('_capBadgesHtml([name]'),
    'badges come from the card',
  );
  assert.ok(body.includes("getElementById('provCapsCount')"), 'the count is updated');
  assert.ok(
    providers.includes("const inkClass = desc.badge_class ? ` ${desc.badge_class}` : '';"),
    'the tick icon wears the capability hue',
  );
});

test('a refusal on a hidden field brings its pane back', () => {
  const body = oneFunction(providers, '_rejectProviderField');
  assert.ok(body.includes("if (pane === 'provPaneCaps') setProviderPane('caps');"));
});

test('the tick row updates the summary as it changes', () => {
  const body = oneFunction(providers, 'toggleProvCapSection');
  assert.ok(body.includes('_renderCapsSummary();'));
});

test('every new string exists in both dictionaries', () => {
  const enBody = i18n.slice(i18n.indexOf('  en: {'), i18n.indexOf('  zh: {'));
  const zhStart = i18n.indexOf('  zh: {');
  const zhBody = i18n.slice(zhStart, i18n.indexOf('\n};', zhStart));
  for (const key of [
    'prov.tabCaps', 'prov.changeInCaps', 'prov.capsHint', 'prov.capsNone',
  ]) {
    assert.equal(enBody.split(`'${key}'`).length - 1, 1, `${key} in en`);
    assert.equal(zhBody.split(`'${key}'`).length - 1, 1, `${key} in zh`);
  }
});

test('the provider preset lives on Basic, and the Capabilities pane holds only faces', () => {
  const basicAt = html.indexOf('id="provPaneBasic"');
  const capsAt = html.indexOf('id="provPaneCaps"');
  const presetAt = html.indexOf('id="provPresetSection"');
  assert.ok(basicAt >= 0 && capsAt > basicAt, 'both panes exist, Basic first');
  assert.ok(presetAt > basicAt && presetAt < capsAt, 'the preset is inside Basic');
  assert.ok(html.indexOf('id="provModelsPath"') < capsAt, 'the model list path stays in Basic');
});

test('the LLM face has its own shim picker, and the saved type comes from it', () => {
  const llmAt = html.indexOf('id="provLlmSection"');
  const capsAt = html.indexOf('id="provPaneCaps"');
  const pickerAt = html.indexOf('id="provShim_llm"');
  assert.ok(pickerAt > llmAt && llmAt > capsAt, 'the LLM face picker is inside the LLM endpoint');
  assert.ok(
    oneFunction(providers, '_saveProvider').includes("getElementById('provShim_llm')"),
    'the type is read from the face, not from the preset',
  );
});

test('the preset is saved with the entry and reopened from it', () => {
  assert.ok(
    oneFunction(providers, '_saveProvider').includes("body.preset = getSelectedValue() || ''"),
    'save sends the preset',
  );
  assert.ok(providers.includes('_storedCfg.preset'), 'open reads the stored preset first');
});
