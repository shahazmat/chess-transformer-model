// Adapter for the local inference server (tools/model_server.py) — the real
// trained checkpoint behind the same predict(ctx) seam the mock implements.
//
// The server owns context assembly (the bare-history contract documented in
// engine.js); this module just ships it the pieces: bare historyTokens, the
// current move's quality, the tokens of the move being decoded, whose turn it
// is, and the human's rating for the opponent Elo slot. The response is the
// full-vocabulary probability vector for the next token; the engine masks and
// renormalizes as usual.

// The backend base URL. `?model=<base-url>` in the page URL overrides the
// local dev server — that's how a hosted copy of the harness (e.g. GitHub
// Pages) points at a tunnel or cloud inference server. Share links like
//   https://you.github.io/repo/?model=https://your-tunnel.example.com
// The server must speak HTTPS when the page does (mixed content is blocked)
// and send CORS headers (tools/model_server.py already does).
function resolveBase() {
  const q = new URLSearchParams(window.location.search).get('model');
  return q ? q.replace(/\/+$/, '') : 'http://127.0.0.1:8123';
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
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) });
    if (!res.ok) return null;
    return createRemoteModel(base, await res.json());
  } catch {
    return null;
  }
}
