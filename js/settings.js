// The sidebar "Engine settings" panel: a live view onto ENGINE_CONFIG.
//
// The engine reads ENGINE_CONFIG at every decode step, so edits here apply
// from the computer's next move — no reload or new game needed. Values
// persist per-browser in localStorage and win over the source defaults on
// load; "Reset to defaults" clears the saved copy and restores the values
// engine.js shipped with.
//
// The nerf checkboxes drive BOTH knobs that mention nerf tokens: a checked
// token is sampleable (allowNerfTokens) AND never dropped by the min-p floor
// (minPExempt). Keeping them in lockstep is deliberate — a sampleable nerf
// token's mass is tiny by nature, so a floor that could drop it would undo
// the checkbox.

import { ENGINE_CONFIG } from './engine.js';
import { NERF_TOKENS } from './vocab.js';

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = 'gpct-engine-settings';

// The source defaults, captured before any saved settings are merged in.
const DEFAULTS = JSON.parse(JSON.stringify(ENGINE_CONFIG));

// Only these keys are editable here (and persisted) — thinkDelayMs stays
// whatever the source says.
const KEYS = [
  'temperature', 'tempMidFrom', 'tempMid', 'tempEndFrom', 'tempEnd',
  'topP', 'topK', 'minP', 'minPFromMove', 'minPExempt',
  'allowNerfTokens', 'forceQuality', 'forceMate', 'allowResign', 'allowDrawOffer',
];

const NERF_BOX = {
  '<inaccuracy>': 'set-nerf-inaccuracy',
  '<mistake>': 'set-nerf-mistake',
  '<blunder>': 'set-nerf-blunder',
};

function persist() {
  const out = {};
  for (const k of KEYS) out[k] = ENGINE_CONFIG[k];
  localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
}

function loadSaved() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { /* corrupt -> ignore */ }
  if (!saved || typeof saved !== 'object') return;
  for (const k of KEYS) if (k in saved) ENGINE_CONFIG[k] = saved[k];
}

// config -> inputs
function syncUI() {
  $('set-temperature').value = ENGINE_CONFIG.temperature;
  $('set-temperature-val').textContent = Number(ENGINE_CONFIG.temperature).toFixed(2);
  $('set-temp-mid-from').value = ENGINE_CONFIG.tempMidFrom ?? 0;
  $('set-temp-mid').value = ENGINE_CONFIG.tempMid ?? 0.7;
  $('set-temp-mid-val').textContent = Number(ENGINE_CONFIG.tempMid ?? 0.7).toFixed(2);
  $('set-temp-end-from').value = ENGINE_CONFIG.tempEndFrom ?? 0;
  $('set-temp-end').value = ENGINE_CONFIG.tempEnd ?? 0.4;
  $('set-temp-end-val').textContent = Number(ENGINE_CONFIG.tempEnd ?? 0.4).toFixed(2);
  $('set-topp').value = ENGINE_CONFIG.topP ?? 1;
  $('set-topp-val').textContent = ENGINE_CONFIG.topP > 0 && ENGINE_CONFIG.topP < 1
    ? Number(ENGINE_CONFIG.topP).toFixed(2) : 'off';
  $('set-topk').value = ENGINE_CONFIG.topK;
  $('set-minp').value = ENGINE_CONFIG.minP;
  $('set-minp-val').textContent = ENGINE_CONFIG.minP > 0 ? Number(ENGINE_CONFIG.minP).toFixed(2) : 'off';
  $('set-minp-from').value = ENGINE_CONFIG.minPFromMove ?? 1;
  const allowed = ENGINE_CONFIG.allowNerfTokens === true ? NERF_TOKENS
    : Array.isArray(ENGINE_CONFIG.allowNerfTokens) ? ENGINE_CONFIG.allowNerfTokens : [];
  for (const [token, id] of Object.entries(NERF_BOX)) $(id).checked = allowed.includes(token);
  $('set-force-quality').value = ENGINE_CONFIG.forceQuality ?? '';
  $('set-force-mate').checked = !!ENGINE_CONFIG.forceMate;
  $('set-allow-resign').checked = !!ENGINE_CONFIG.allowResign;
  $('set-allow-draw').checked = !!ENGINE_CONFIG.allowDrawOffer;
}

// inputs -> config
function apply() {
  ENGINE_CONFIG.temperature = Math.max(0.05, Number($('set-temperature').value) || DEFAULTS.temperature);
  ENGINE_CONFIG.tempMidFrom = Math.max(0, Math.round(Number($('set-temp-mid-from').value) || 0));
  ENGINE_CONFIG.tempMid = Math.max(0.05, Number($('set-temp-mid').value) || DEFAULTS.tempMid);
  ENGINE_CONFIG.tempEndFrom = Math.max(0, Math.round(Number($('set-temp-end-from').value) || 0));
  ENGINE_CONFIG.tempEnd = Math.max(0.05, Number($('set-temp-end').value) || DEFAULTS.tempEnd);
  ENGINE_CONFIG.topP = Math.min(1, Math.max(0.05, Number($('set-topp').value) || 1));
  ENGINE_CONFIG.topK = Math.max(0, Math.round(Number($('set-topk').value) || 0));
  ENGINE_CONFIG.minP = Math.max(0, Number($('set-minp').value) || 0);
  ENGINE_CONFIG.minPFromMove = Math.max(1, Math.round(Number($('set-minp-from').value) || 1));
  const allowed = Object.entries(NERF_BOX).filter(([, id]) => $(id).checked).map(([token]) => token);
  ENGINE_CONFIG.allowNerfTokens = allowed;
  ENGINE_CONFIG.minPExempt = allowed;
  ENGINE_CONFIG.forceQuality = $('set-force-quality').value || null;
  ENGINE_CONFIG.forceMate = $('set-force-mate').checked;
  ENGINE_CONFIG.allowResign = $('set-allow-resign').checked;
  ENGINE_CONFIG.allowDrawOffer = $('set-allow-draw').checked;
  persist();
  $('set-temperature-val').textContent = ENGINE_CONFIG.temperature.toFixed(2);
  $('set-topp-val').textContent = ENGINE_CONFIG.topP > 0 && ENGINE_CONFIG.topP < 1
    ? ENGINE_CONFIG.topP.toFixed(2) : 'off';
  $('set-temp-mid-val').textContent = ENGINE_CONFIG.tempMid.toFixed(2);
  $('set-temp-end-val').textContent = ENGINE_CONFIG.tempEnd.toFixed(2);
  $('set-minp-val').textContent = ENGINE_CONFIG.minP > 0 ? ENGINE_CONFIG.minP.toFixed(2) : 'off';
}

loadSaved();
syncUI();

$('settings').addEventListener('input', apply);
$('settings').addEventListener('change', apply);
$('set-reset').addEventListener('click', () => {
  localStorage.removeItem(STORAGE_KEY);
  for (const k of KEYS) ENGINE_CONFIG[k] = JSON.parse(JSON.stringify(DEFAULTS[k] ?? null));
  syncUI();
});
