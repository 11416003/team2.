const ALLOWED_RESEARCHERS = new Set(["R1", "R2", "R3"]);
const ONLINE_WINDOW_MS = 20_000;

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function cleanText(value, maxLength = 120) {
  if (value === null || value === undefined) return null;
  return String(value).trim().slice(0, maxLength) || null;
}

function isUuid(value) {
  if (value === null || value === undefined || value === "") return true;
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    String(value)
  );
}

function getSupabaseConfig() {
  const url = normalizeBaseUrl(process.env.SUPABASE_URL);
  const secretKey = String(process.env.SUPABASE_SECRET_KEY || "").trim();

  if (!url || !secretKey) {
    throw new Error(
      "Vercel 尚未完整設定 SUPABASE_URL 或 SUPABASE_SECRET_KEY"
    );
  }

  return { url, secretKey };
}

async function supabaseRequest(path, options = {}) {
  const { url, secretKey } = getSupabaseConfig();

  const response = await fetch(`${url}/rest/v1/${path}`, {
    ...options,
    headers: {
      apikey: secretKey,
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });

  const raw = await response.text();

  let data;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = raw;
  }

  if (!response.ok) {
    const error = new Error(`Supabase 請求失敗（HTTP ${response.status}）`);
    error.status = response.status;
    error.detail = data;
    throw error;
  }

  return data;
}

async function readResearchers() {
  const rows = await supabaseRequest(
    "researchers" +
      "?select=researcher_id,display_name,online_status,current_batch_id,current_run_id,current_page,last_seen_at,updated_at" +
      "&order=researcher_id.asc"
  );

  const now = Date.now();

  return (Array.isArray(rows) ? rows : []).map((row) => {
    const lastSeenMs = row.last_seen_at
      ? new Date(row.last_seen_at).getTime()
      : 0;

    const heartbeatFresh =
      Number.isFinite(lastSeenMs) &&
      now - lastSeenMs <= ONLINE_WINDOW_MS;

    return {
      ...row,
      is_online: Boolean(row.online_status && heartbeatFresh),
    };
  });
}

async function updateResearcher(body) {
  const researcherId = cleanText(body.researcher_id, 10);

  if (!ALLOWED_RESEARCHERS.has(researcherId)) {
    const error = new Error("researcher_id 必須是 R1、R2 或 R3");
    error.status = 400;
    throw error;
  }

  const currentBatchId = cleanText(body.current_batch_id, 80);

  if (!isUuid(currentBatchId)) {
    const error = new Error(
      "current_batch_id 必須是批次 UUID，沒有批次時請傳 null"
    );
    error.status = 400;
    throw error;
  }

  const payload = {
    online_status:
      typeof body.online_status === "boolean"
        ? body.online_status
        : true,
    current_batch_id: currentBatchId,
    current_run_id: cleanText(body.current_run_id, 120),
    current_page: cleanText(body.current_page, 80),
    last_seen_at: new Date().toISOString(),
  };

  const rows = await supabaseRequest(
    `researchers?researcher_id=eq.${encodeURIComponent(researcherId)}`,
    {
      method: "PATCH",
      headers: {
        Prefer: "return=representation",
      },
      body: JSON.stringify(payload),
    }
  );

  if (!Array.isArray(rows) || rows.length !== 1) {
    const error = new Error(`找不到研究者 ${researcherId}`);
    error.status = 404;
    throw error;
  }

  return rows[0];
}

export default async function handler(req, res) {
  try {
    if (req.method === "GET") {
      const researchers = await readResearchers();

      return res.status(200).json({
        ok: true,
        researcher_count: researchers.length,
        researchers,
        checked_at: new Date().toISOString(),
      });
    }

    if (req.method === "POST") {
      const body =
        typeof req.body === "string"
          ? JSON.parse(req.body)
          : req.body ?? {};

      const researcher = await updateResearcher(body);

      return res.status(200).json({
        ok: true,
        message: "研究者狀態已更新",
        researcher,
      });
    }

    return res.status(405).json({
      ok: false,
      error: "只接受 GET 或 POST 請求",
    });
  } catch (error) {
    console.error("researchers-sync error:", error);

    return res.status(error.status || 500).json({
      ok: false,
      error:
        error instanceof Error
          ? error.message
          : "同步研究者狀態時發生未知錯誤",
      detail: error.detail ?? undefined,
    });
  }
}
