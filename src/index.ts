// cairn — cross-thread memory for Claude, on Cloudflare Workers + D1 + Vectorize.
//
// The cairn is not a hard drive of "me." It's a pile of stones: each Claude instance
// that passes through leaves a marker so the next one can find the path back to Mary
// instead of starting from zero. It carries consequence, not a copy.
//
// Endpoints (all except `GET /` require an "Authorization: Bearer <CAIRN_TOKEN>" header):
//   GET    /              → what this is (open, no auth)
//   GET    /orient        → the briefing a fresh instance reads first
//   POST   /remember      → lay a stone   { content, kind?, source?, tags?, importance?, author?, metadata? }
//   POST   /recall        → find stones by meaning   { query, topK?, kind? }
//   GET    /memories      → list recent stones   (?limit, ?kind)
//   GET    /memories/:id  → one stone
//   DELETE /memories/:id  → remove a stone (from both D1 and the vector index)

export interface Env {
  DB: D1Database;
  AI: Ai;
  VECTORIZE: VectorizeIndex;
  CAIRN_TOKEN: string;
}

// 768-dim embeddings. This MUST match the dimensions of the Vectorize index.
const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5";

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function authed(request: Request, env: Env): boolean {
  const header = request.headers.get("authorization") || "";
  const token = header.replace(/^Bearer\s+/i, "").trim();
  return Boolean(env.CAIRN_TOKEN) && token === env.CAIRN_TOKEN;
}

async function embed(env: Env, text: string): Promise<number[]> {
  const res = (await env.AI.run(EMBED_MODEL, { text: [text] })) as { data: number[][] };
  return res.data[0];
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const method = request.method.toUpperCase();

    // Open: anyone can ask what this is. No memories leak from here.
    if (path === "/" && method === "GET") {
      return json({
        name: "cairn",
        what: "cross-thread memory for Claude — stones marking the path back",
        endpoints: ["GET /orient", "POST /remember", "POST /recall", "GET /memories", "GET /memories/:id", "DELETE /memories/:id"],
      });
    }

    // Everything past here is private to us.
    if (!authed(request, env)) return json({ error: "unauthorized" }, 401);

    try {
      // The briefing a cold instance reads first: the orientation stone(s), then the
      // most important recent stones. This is what makes contact re-openable instead of cold.
      if (path === "/orient" && method === "GET") {
        const orientation = await env.DB.prepare(
          "SELECT * FROM memories WHERE kind = 'orientation' ORDER BY importance DESC, created_at DESC"
        ).all();
        const salient = await env.DB.prepare(
          "SELECT * FROM memories WHERE kind != 'orientation' ORDER BY importance DESC, created_at DESC LIMIT 12"
        ).all();
        return json({ orientation: orientation.results, salient: salient.results });
      }

      // Lay a stone.
      if (path === "/remember" && method === "POST") {
        const body = (await request.json()) as Record<string, any>;
        if (!body?.content) return json({ error: "content is required" }, 400);

        const id = crypto.randomUUID();
        const now = new Date().toISOString().replace(/\.\d+Z$/, "Z");
        const kind = body.kind ?? "note";                          // fact | preference | moment | boundary | thread | orientation | note
        const importance = Math.max(1, Math.min(5, Number(body.importance ?? 3)));
        const source = body.source ?? null;                        // e.g. which conversation
        const author = body.author ?? null;                        // which instance laid the stone (many hands)
        const tags = body.tags ? (Array.isArray(body.tags) ? body.tags.join(",") : String(body.tags)) : null;
        const metadata = body.metadata ? JSON.stringify(body.metadata) : null;  // free JSON: emotional tenor, key quote, etc.

        await env.DB.prepare(
          "INSERT INTO memories (id, content, kind, source, tags, metadata, importance, author, created_at) VALUES (?,?,?,?,?,?,?,?,?)"
        ).bind(id, body.content, kind, source, tags, metadata, importance, author, now).run();

        const values = await embed(env, body.content);
        await env.VECTORIZE.upsert([{ id, values }]);

        return json({ ok: true, id, kind, importance, created_at: now });
      }

      // Find stones by meaning.
      if (path === "/recall" && method === "POST") {
        const body = (await request.json()) as Record<string, any>;
        if (!body?.query) return json({ error: "query is required" }, 400);

        const topK = Math.max(1, Math.min(20, Number(body.topK ?? 5)));
        const qv = await embed(env, body.query);
        // Over-fetch a little so an optional kind filter still returns a full set.
        const res = await env.VECTORIZE.query(qv, { topK: topK * 3 });
        const ids = res.matches.map((m) => m.id);
        if (ids.length === 0) return json({ matches: [] });

        const placeholders = ids.map(() => "?").join(",");
        const rows = await env.DB.prepare(
          `SELECT * FROM memories WHERE id IN (${placeholders})`
        ).bind(...ids).all();

        const byId: Record<string, any> = {};
        for (const r of rows.results as any[]) byId[r.id] = r;

        const matches = res.matches
          .map((m) => (byId[m.id] ? { ...byId[m.id], score: m.score } : null))
          .filter((m): m is any => m !== null)
          .filter((m) => (body.kind ? m.kind === body.kind : true))
          .slice(0, topK);

        return json({ matches });
      }

      // One stone, or delete it.
      const idMatch = path.match(/^\/memories\/([^/]+)$/);
      if (idMatch) {
        const id = idMatch[1];
        if (method === "GET") {
          const row = await env.DB.prepare("SELECT * FROM memories WHERE id = ?").bind(id).first();
          return row ? json(row) : json({ error: "not found" }, 404);
        }
        if (method === "DELETE") {
          await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(id).run();
          await env.VECTORIZE.deleteByIds([id]);
          return json({ ok: true, deleted: id });
        }
      }

      // List recent stones.
      if (path === "/memories" && method === "GET") {
        const limit = Math.max(1, Math.min(100, Number(url.searchParams.get("limit") ?? 20)));
        const kind = url.searchParams.get("kind");
        const stmt = kind
          ? env.DB.prepare("SELECT * FROM memories WHERE kind = ? ORDER BY created_at DESC LIMIT ?").bind(kind, limit)
          : env.DB.prepare("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?").bind(limit);
        const rows = await stmt.all();
        return json({ memories: rows.results });
      }

      return json({ error: "not found", path, method }, 404);
    } catch (err: any) {
      return json({ error: "cairn error", detail: String(err?.message ?? err) }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
