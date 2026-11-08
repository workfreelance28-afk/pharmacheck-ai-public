const TELEGRAM_API_URL = "https://api.telegram.org";
const ALLOWED_TELEGRAM_METHODS = new Set([
  "getMe",
  "getWebhookInfo",
  "sendMessage",
  "setMyCommands",
  "setWebhook",
]);

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

function resolveUpstream(env) {
  try {
    const upstream = new URL(env.UPSTREAM_URL);
    if (
      upstream.protocol !== "https:" ||
      !upstream.hostname.endsWith(".apigw.yandexcloud.net") ||
      upstream.pathname !== "/telegram" ||
      upstream.search ||
      upstream.hash ||
      upstream.username ||
      upstream.password
    ) {
      return null;
    }
    return upstream.toString();
  } catch {
    return null;
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return new Response("ok", { headers: { "Cache-Control": "no-store" } });
    }
    if (request.method === "POST" && url.pathname === "/telegram") {
      return relayWebhook(request, env);
    }
    if (request.method === "POST" && url.pathname.startsWith("/telegram-api/")) {
      return relayTelegramApi(request, url, env);
    }
    return new Response("Not Found", { status: 404 });
  },
};

async function relayWebhook(request, env) {
  const upstream = resolveUpstream(env);
  if (!upstream) return new Response("Relay Misconfigured", { status: 503 });

  try {
    const headers = new Headers({
      "Content-Type": request.headers.get("Content-Type") || "application/json",
    });
    const telegramSecret = request.headers.get(
      "X-Telegram-Bot-Api-Secret-Token",
    );
    if (telegramSecret) {
      headers.set("X-Telegram-Bot-Api-Secret-Token", telegramSecret);
    }

    const response = await fetch(upstream, {
      method: "POST",
      headers,
      body: await request.arrayBuffer(),
      redirect: "manual",
    });
    return new Response(response.ok ? "OK" : "Upstream Error", {
      status: response.ok ? 200 : 502,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return new Response("Relay Error", { status: 502 });
  }
}

async function relayTelegramApi(request, url, env) {
  const method = url.pathname.slice("/telegram-api/".length);
  if (!ALLOWED_TELEGRAM_METHODS.has(method)) {
    return new Response("Not Found", { status: 404 });
  }
  if (!safeEqual(request.headers.get("X-Pharmacheck-Relay-Secret"), env.RELAY_SECRET)) {
    return new Response("Unauthorized", { status: 401 });
  }
  if (typeof env.TELEGRAM_BOT_TOKEN !== "string" || !env.TELEGRAM_BOT_TOKEN) {
    return new Response("Relay Misconfigured", { status: 503 });
  }

  try {
    const response = await fetch(`${TELEGRAM_API_URL}/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: await request.arrayBuffer(),
      redirect: "manual",
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("Relay Error", { status: 502 });
  }
}
