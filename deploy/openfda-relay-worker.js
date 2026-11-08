const UPSTREAM = "https://api.fda.gov/drug/label.json";
const ALLOWED_QUERY_PARAMS = new Set(["search", "limit"]);

function jsonResponse(status, error) {
  return new Response(JSON.stringify({ error }), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function safeEqual(left, right) {
  if (typeof left !== "string" || typeof right !== "string") return false;
  const encoder = new TextEncoder();
  const first = encoder.encode(left);
  const second = encoder.encode(right);
  if (first.length !== second.length) return false;
  let difference = 0;
  for (let index = 0; index < first.length; index += 1) {
    difference |= first[index] ^ second[index];
  }
  return difference === 0;
}

export default {
  async fetch(request, env) {
    const incoming = new URL(request.url);

    if (request.method !== "GET" || incoming.pathname !== "/drug/label.json") {
      return jsonResponse(404, "not_found");
    }
    if (!env.RELAY_SECRET) {
      return jsonResponse(503, "relay_not_configured");
    }
    if (
      !safeEqual(
        request.headers.get("X-PharmaCheck-Relay-Secret"),
        env.RELAY_SECRET,
      )
    ) {
      return jsonResponse(401, "unauthorized");
    }

    for (const key of incoming.searchParams.keys()) {
      if (
        !ALLOWED_QUERY_PARAMS.has(key) ||
        incoming.searchParams.getAll(key).length !== 1
      ) {
        return jsonResponse(400, "invalid_query");
      }
    }

    const search = incoming.searchParams.get("search");
    const limitText = incoming.searchParams.get("limit");
    const limit = Number(limitText);
    if (!search || search.length > 6000) {
      return jsonResponse(400, "invalid_search");
    }
    if (
      !/^\d+$/.test(limitText ?? "") ||
      !Number.isSafeInteger(limit) ||
      limit < 1 ||
      limit > 100
    ) {
      return jsonResponse(400, "invalid_limit");
    }

    const upstream = new URL(UPSTREAM);
    upstream.searchParams.set("search", search);
    upstream.searchParams.set("limit", String(limit));

    try {
      const response = await fetch(upstream, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      const headers = new Headers({
        "Content-Type":
          response.headers.get("Content-Type") || "application/json",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-PharmaCheck-Relay": "openfda-v1",
      });
      return new Response(response.body, { status: response.status, headers });
    } catch {
      return jsonResponse(502, "upstream_unavailable");
    }
  },
};
