// Adapter for the local inference server (tools/model_server.py) — the real
// trained checkpoint behind the same predict(ctx) seam the mock implements.
//
// The server owns context assembly (the bare-history contract documented in
// engine.js); this module just ships it the pieces: bare historyTokens, the
// current move's quality, the tokens of the move being decoded, whose turn it
// is, and the human's rating for the opponent Elo slot. The response is the
// full-vocabulary probability vector for the next token; the engine masks and
// renormalizes as usual.

// The backend base URL, in priority order:
//   1. `?model=<base-url>` in the page URL (explicit override)
//   2. the local dev server when the page itself is served from localhost
//   3. the Hugging Face Space (the hosted default — a phone opening the
//      GitHub Pages copy needs no query parameter)
// The server must speak HTTPS when the page does (mixed content is blocked)
// and send CORS headers (tools/model_server.py already does).
const LOCAL_BASE = 'http://127.0.0.1:8123';
const HOSTED_BASE = 'https://shazmate-gpct-server.hf.space';
function resolveBase() {
  const q = new URLSearchParams(window.location.search).get('model');
  if (q) return q.replace(/\/+$/, '');
  const h = window.location.hostname;
  return h === 'localhost' || h === '127.0.0.1' ? LOCAL_BASE : HOSTED_BASE;
}
export const SERVER_BASE = resolveBase();

export function createRemoteModel(base = SERVER_BASE, info = null) {
  return {
    name: info ? `${info.repo} @ iter ${info.iter_num}` : `remote:${base}`,
    async predict(ctx) {
      const res = await fetch(`${base}/predict`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          historyTokens: ctx.historyTokens,
          quality: ctx.quality,
          moveTokens: ctx.moveTokens,
          turn: ctx.turn,
          opponentRating: ctx.opponent?.rating ?? null,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(`model server ${res.status}: ${detail.error ?? 'unknown error'}`);
      }
      return new Float32Array(await res.json());
    },
  };
}

// Probe the server; resolves to a ready model or null if it isn't running.
export async function detectRemoteModel(base = SERVER_BASE) {
  try {
    // Generous timeout: a phone on mobile data, or a Space that is waking
    // from sleep, can take far longer than a LAN health check.
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    return createRemoteModel(base, await res.json());
  } catch {
    return null;
  }
}
