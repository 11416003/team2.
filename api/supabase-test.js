const TABLE = "researchers";

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({
      ok: false,
      error: "只接受 GET 請求",
    });
  }

  const supabaseUrl = normalizeBaseUrl(process.env.SUPABASE_URL);
  const supabaseSecretKey = String(
    process.env.SUPABASE_SECRET_KEY || ""
  ).trim();

  if (!supabaseUrl || !supabaseSecretKey) {
    return res.status(500).json({
      ok: false,
      error: "Vercel 尚未完整設定 SUPABASE_URL 或 SUPABASE_SECRET_KEY",
    });
  }

  try {
    const endpoint =
      `${supabaseUrl}/rest/v1/${TABLE}` +
      "?select=researcher_id,display_name,online_status,last_seen_at" +
      "&order=researcher_id.asc";

    const response = await fetch(endpoint, {
      method: "GET",
      headers: {
        apikey: supabaseSecretKey,
        Accept: "application/json",
      },
    });

    const responseText = await response.text();

    let data;
    try {
      data = responseText ? JSON.parse(responseText) : [];
    } catch {
      data = responseText;
    }

    if (!response.ok) {
      console.error("Supabase REST error:", response.status, data);

      return res.status(502).json({
        ok: false,
        error: "Supabase 查詢失敗",
        supabase_status: response.status,
        detail: data,
      });
    }

    return res.status(200).json({
      ok: true,
      connected: true,
      message: "Vercel 已成功連線 Supabase",
      researcher_count: Array.isArray(data) ? data.length : 0,
      researchers: data,
      checked_at: new Date().toISOString(),
    });
  } catch (error) {
    console.error("supabase-test error:", error);

    return res.status(500).json({
      ok: false,
      error:
        error instanceof Error
          ? error.message
          : "測試 Supabase 連線時發生未知錯誤",
    });
  }
}
