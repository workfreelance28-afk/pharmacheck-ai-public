const UPSTREAM_URL =
  "https://YOUR_GATEWAY_ID.example.apigw.yandexcloud.net/telegram";
const TELEGRAM_API_URL = "https://api.telegram.org";
const ALLOWED_TELEGRAM_METHODS = new Set([
  "getMe",
  "getWebhookInfo",
  "sendMessage",
  "setMyCommands",
  "setWebhook",
]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return new Response("ok");
    }

    if (request.method === "POST" && url.pathname === "/telegram") {
      return relayWebhook(request);
    }

    if (request.method === "POST" && url.pathname.startsWith("/telegram-api/")) {
      return relayTelegramApi(request, url, env);
    }

    return new Response("Not Found", { status: 404 });
  },
};

async function relayWebhook(request) {
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

    const upstreamResponse = await fetch(UPSTREAM_URL, {
      method: "POST",
      headers,
      body: await request.arrayBuffer(),
      redirect: "manual",
    });

    return new Response(upstreamResponse.ok ? "OK" : "Upstream Error", {
      status: upstreamResponse.ok ? 200 : 502,
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
  if (
    typeof env.RELAY_SECRET !== "string" ||
    request.headers.get("X-Pharmacheck-Relay-Secret") !== env.RELAY_SECRET
  ) {
    return new Response("Unauthorized", { status: 401 });
  }
  if (typeof env.TELEGRAM_BOT_TOKEN !== "string" || !env.TELEGRAM_BOT_TOKEN) {
    return new Response("Relay Misconfigured", { status: 503 });
  }

  try {
    const upstreamResponse = await fetch(
      `${TELEGRAM_API_URL}/bot${env.TELEGRAM_BOT_TOKEN}/${method}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: await request.arrayBuffer(),
        redirect: "manual",
      },
    );
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch {
    return new Response("Relay Error", { status: 502 });
  }
}
