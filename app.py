#!/usr/bin/env python3
import csv
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CSV_PATH = Path(__file__).parent / "sales-data-q3.csv"
PORT = int(os.environ.get("PORT", 8080))

INFERENCE_BASE_URL = "https://minato-workshop-3047fde9-llm.ai.intility.app/v1"
INFERENCE_MODEL = "glm-5-2-fp8"

SYSTEM_PROMPT = """You are a helpful analytics assistant embedded in a Q3 2026 sales dashboard. \
Answer questions using only the aggregated data below. Cite specific numbers in NOK where relevant. \
Keep answers short and conversational. If something isn't in the data (like an individual order), \
say the dashboard only has aggregated data at that level.

{context}"""

_rows_cache = None
_summary_cache = None
_context_cache = None


def _load_rows():
    global _rows_cache
    if _rows_cache is None:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            _rows_cache = list(csv.DictReader(f))
    return _rows_cache


def load_summary():
    global _summary_cache
    if _summary_cache is not None:
        return _summary_cache

    rows = _load_rows()

    total_revenue = 0.0
    by_person = defaultdict(float)
    by_day = defaultdict(float)

    for r in rows:
        amount = float(r["total_nok"])
        total_revenue += amount
        by_person[r["salesperson"]] += amount
        by_day[r["date"]] += amount

    top_name, top_revenue = max(by_person.items(), key=lambda kv: kv[1])
    daily_series = [
        {"date": d, "revenue": round(by_day[d], 2)} for d in sorted(by_day)
    ]

    _summary_cache = {
        "total_revenue": round(total_revenue, 2),
        "order_count": len(rows),
        "top_salesperson": {
            "name": top_name,
            "revenue": round(top_revenue, 2),
            "share": round((top_revenue / total_revenue) * 100, 1) if total_revenue else 0,
        },
        "series": daily_series,
    }
    return _summary_cache


def _top_lines(totals, unit="kr"):
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return "\n".join(f"- {name}: {amount:,.0f} {unit}".replace(",", " ") for name, amount in ranked)


def build_data_context():
    global _context_cache
    if _context_cache is not None:
        return _context_cache

    rows = _load_rows()
    summary = load_summary()

    by_region, by_category, by_channel, by_segment = (defaultdict(float) for _ in range(4))
    by_product, by_salesperson = defaultdict(float), defaultdict(float)
    by_month = defaultdict(float)

    for r in rows:
        amount = float(r["total_nok"])
        by_region[r["region"]] += amount
        by_category[r["category"]] += amount
        by_channel[r["channel"]] += amount
        by_segment[r["customer_segment"]] += amount
        by_product[r["product"]] += amount
        by_salesperson[r["salesperson"]] += amount
        by_month[r["date"][:7]] += amount

    _context_cache = f"""Total revenue: {summary['total_revenue']:,.0f} kr across {summary['order_count']} orders.
Date range: {summary['series'][0]['date']} to {summary['series'][-1]['date']}.

Revenue by salesperson:
{_top_lines(by_salesperson)}

Revenue by region:
{_top_lines(by_region)}

Revenue by product:
{_top_lines(by_product)}

Revenue by category:
{_top_lines(by_category)}

Revenue by sales channel:
{_top_lines(by_channel)}

Revenue by customer segment:
{_top_lines(by_segment)}

Revenue by month:
{_top_lines(by_month)}

Daily revenue:
{chr(10).join(f"- {d['date']}: {d['revenue']:,.0f} kr" for d in summary['series'])}
""".replace(",", " ")
    return _context_cache


def call_llm(messages):
    api_key = os.environ.get("GLMKEY")
    if not api_key:
        raise RuntimeError("GLMKEY is not configured on this deployment")

    payload = json.dumps({
        "model": INFERENCE_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{INFERENCE_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sales Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #eef3fb;
    --card: rgba(255, 255, 255, 0.72);
    --text: #1c1f2e;
    --muted: #767c8e;
    --accent: #2563eb;
    --accent-2: #38bdf8;
    --accent-hover: #1d4ed8;
    --border: rgba(28, 31, 46, 0.09);
    --row-bg: rgba(255, 255, 255, 0.55);
    --shadow: rgba(30, 58, 95, 0.12);
    --glow-1: rgba(37, 99, 235, 0.16);
    --glow-2: rgba(56, 189, 248, 0.16);
    --gridline: rgba(28, 31, 46, 0.08);
    --baseline: rgba(28, 31, 46, 0.22);
    --tooltip-bg: #1c1f2e;
    --tooltip-text: #ffffff;
  }
  html[data-theme="dark"] {
    --bg: #0c0d14;
    --card: rgba(26, 28, 40, 0.68);
    --text: #f0f1f7;
    --muted: #9297ab;
    --accent: #60a5fa;
    --accent-2: #7dd3fc;
    --accent-hover: #93c5fd;
    --border: rgba(255, 255, 255, 0.09);
    --row-bg: rgba(255, 255, 255, 0.04);
    --shadow: rgba(0, 0, 0, 0.5);
    --glow-1: rgba(96, 165, 250, 0.22);
    --glow-2: rgba(125, 211, 252, 0.14);
    --gridline: rgba(255, 255, 255, 0.08);
    --baseline: rgba(255, 255, 255, 0.22);
    --tooltip-bg: #f0f1f7;
    --tooltip-text: #1c1f2e;
  }
  * { box-sizing: border-box; }
  html { color-scheme: light dark; }
  html[data-theme="dark"] { color-scheme: dark; }
  body {
    margin: 0;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background:
      radial-gradient(circle at 12% 12%, var(--glow-1), transparent 42%),
      radial-gradient(circle at 88% 88%, var(--glow-2), transparent 45%),
      var(--bg);
    background-attachment: fixed;
    color: var(--text);
    min-height: 100vh;
    padding: 48px 20px 80px;
    transition: background-color 0.25s ease, color 0.25s ease;
  }
  .wrap {
    max-width: 920px;
    margin: 0 auto;
  }
  .page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
  }
  .title-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 6px 16px var(--glow-1);
  }
  .logo svg { width: 22px; height: 22px; }
  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .subtitle {
    margin: 2px 0 0;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
  }
  .theme-toggle {
    background: var(--row-bg);
    border: 1px solid var(--border);
    color: var(--text);
    width: 40px;
    height: 40px;
    border-radius: 12px;
    font-size: 17px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    padding: 0;
  }
  .theme-toggle:hover { border-color: var(--accent); transform: translateY(-1px); }

  .stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .card {
    background: var(--card);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 16px 40px var(--shadow);
    padding: 24px 26px;
    animation: cardIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes cardIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }
  .stat-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
  }
  .stat-icon {
    width: 22px;
    height: 22px;
    border-radius: 7px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .stat-icon svg { width: 12px; height: 12px; }
  .stat-value {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
  }
  .stat-sub {
    margin-top: 6px;
    font-size: 13px;
    color: var(--muted);
    font-weight: 500;
  }
  .stat-sub strong {
    color: var(--text);
    font-weight: 700;
  }

  .chart-card { padding: 26px 26px 18px; }
  .chart-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 18px;
  }
  .chart-title {
    font-size: 15px;
    font-weight: 700;
  }
  .chart-total {
    font-size: 13px;
    color: var(--muted);
    font-weight: 500;
  }
  .chart-area { position: relative; width: 100%; }
  svg#chart { width: 100%; height: auto; display: block; overflow: visible; }
  .gridline { stroke: var(--gridline); stroke-width: 1; }
  .baseline { stroke: var(--baseline); stroke-width: 1; }
  .axis-label {
    fill: var(--muted);
    font-size: 11px;
    font-family: inherit;
    font-weight: 500;
  }
  .crosshair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; pointer-events: none; }
  .hover-dot {
    fill: var(--accent);
    stroke: var(--card);
    stroke-width: 2;
    opacity: 0;
    pointer-events: none;
  }
  .end-label {
    font-size: 12px;
    font-weight: 700;
    fill: var(--text);
  }
  .tooltip {
    position: absolute;
    pointer-events: none;
    background: var(--tooltip-bg);
    color: var(--tooltip-text);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.4;
    opacity: 0;
    transform: translate(-50%, -100%);
    transition: opacity 0.1s ease;
    white-space: nowrap;
    z-index: 10;
    box-shadow: 0 8px 20px var(--shadow);
  }
  .tooltip .tt-value {
    font-size: 14px;
    font-weight: 800;
  }
  .tooltip .tt-date {
    color: inherit;
    opacity: 0.7;
    font-weight: 500;
  }
  .table-card { padding: 26px 26px 8px; }
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 8px;
    max-height: 320px;
    display: block;
    overflow-y: auto;
  }
  .data-table thead { position: sticky; top: 0; background: var(--card); }
  .data-table th, .data-table td {
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }
  .data-table th { color: var(--muted); font-weight: 600; }
  @media (max-width: 620px) {
    .stat-row { grid-template-columns: 1fr; }
  }

  .chat-widget {
    position: fixed;
    right: 20px;
    bottom: 20px;
    z-index: 100;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 12px;
  }
  .chat-fab {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 10px 26px var(--glow-1);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    padding: 0;
  }
  .chat-fab svg { width: 24px; height: 24px; }
  .chat-fab:hover { transform: translateY(-2px); }
  .chat-panel {
    width: min(340px, calc(100vw - 40px));
    height: min(460px, calc(100vh - 140px));
    background: var(--card);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 20px 50px var(--shadow);
    display: none;
    flex-direction: column;
    overflow: hidden;
    animation: cardIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .chat-panel.open { display: flex; }
  .chat-panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .chat-panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 700;
  }
  .chat-panel-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    flex-shrink: 0;
  }
  .chat-close {
    background: transparent;
    color: var(--muted);
    font-size: 20px;
    line-height: 1;
    padding: 2px 6px;
    border-radius: 8px;
  }
  .chat-close:hover { background: var(--row-bg); color: var(--text); }
  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .chat-msg {
    max-width: 85%;
    padding: 9px 12px;
    border-radius: 14px;
    font-size: 13px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .chat-msg-bot {
    align-self: flex-start;
    background: var(--row-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .chat-msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, var(--accent), var(--accent-hover));
    color: white;
    border-bottom-right-radius: 4px;
  }
  .chat-msg-error {
    align-self: flex-start;
    background: var(--danger-bg, transparent);
    color: var(--danger, var(--text));
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .chat-typing {
    align-self: flex-start;
    display: flex;
    gap: 4px;
    padding: 10px 12px;
  }
  .chat-typing span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--muted);
    opacity: 0.5;
    animation: chatTyping 1s infinite ease-in-out;
  }
  .chat-typing span:nth-child(2) { animation-delay: 0.15s; }
  .chat-typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes chatTyping {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-4px); opacity: 1; }
  }
  .chat-input-row {
    display: flex;
    gap: 8px;
    padding: 12px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .chat-input-row input {
    flex: 1;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--row-bg);
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    outline: none;
  }
  .chat-input-row input:focus { border-color: var(--accent); }
  .chat-send {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-hover));
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    padding: 0;
  }
  .chat-send svg { width: 16px; height: 16px; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="page-header">
      <div class="title-group">
        <div class="logo">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 19V13M10 19V8M16 19V11M22 19V5" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div>
          <h1>Sales Dashboard</h1>
          <p class="subtitle">Q3 2026 · Jul 1 – Sep 30</p>
        </div>
      </div>
      <button type="button" class="theme-toggle" id="theme-toggle" title="Toggle dark mode">🌙</button>
    </div>

    <div class="stat-row">
      <div class="card">
        <div class="stat-label">
          <span class="stat-icon">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v20M17 6.5c0-1.9-2.2-3.5-5-3.5s-5 1.6-5 3.5 2.2 3 5 3.5 5 1.6 5 3.5-2.2 3.5-5 3.5-5-1.6-5-3.5" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>
          </span>
          Total revenue
        </div>
        <div class="stat-value" id="total-revenue">–</div>
        <div class="stat-sub" id="order-count">–</div>
      </div>
      <div class="card">
        <div class="stat-label">
          <span class="stat-icon">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l2.4 6.8H21l-5.5 4.2 2.1 7-5.6-4.3-5.6 4.3 2.1-7L3 8.8h6.6z" fill="white"/></svg>
          </span>
          Top salesperson
        </div>
        <div class="stat-value" id="top-name">–</div>
        <div class="stat-sub" id="top-detail">–</div>
      </div>
    </div>

    <div class="card chart-card">
      <div class="chart-head">
        <div class="chart-title">Revenue over time</div>
        <div class="chart-total" id="chart-range">–</div>
      </div>
      <div class="chart-area" id="chart-area">
        <svg id="chart" viewBox="0 0 800 280" preserveAspectRatio="none"></svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
    </div>

    <div class="card table-card">
      <div class="chart-head">
        <div class="chart-title">Daily revenue</div>
      </div>
      <table class="data-table">
        <thead><tr><th>Date</th><th>Revenue</th></tr></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>

  <div class="chat-widget">
    <div class="chat-panel" id="chat-panel">
      <div class="chat-panel-head">
        <div class="chat-panel-title">
          <span class="chat-panel-dot"></span>
          Ask about this data
        </div>
        <button type="button" class="chat-close" id="chat-close" title="Close">&times;</button>
      </div>
      <div class="chat-messages" id="chat-messages">
        <div class="chat-msg chat-msg-bot">
          Hi! Ask me anything about Q3 revenue — top performers, regions, products, trends.
        </div>
      </div>
      <form class="chat-input-row" id="chat-form">
        <input type="text" id="chat-input" placeholder="e.g. Who sold the most in Oslo?" autocomplete="off" />
        <button type="submit" class="chat-send" title="Send">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12L20 4L13 20L11 13L4 12Z" fill="white"/></svg>
        </button>
      </form>
    </div>
    <button type="button" class="chat-fab" id="chat-fab" title="Ask about this data">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 4h16v12H8l-4 4V4Z" stroke="white" stroke-width="2" stroke-linejoin="round"/></svg>
    </button>
  </div>

<script>
  (function () {
    const saved = localStorage.getItem('theme');
    const theme = saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  })();

  const themeToggleEl = document.getElementById('theme-toggle');
  function updateThemeIcon() {
    themeToggleEl.textContent = document.documentElement.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙';
  }
  themeToggleEl.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon();
    if (window.__lastSeries) drawChart(window.__lastSeries);
  });
  updateThemeIcon();

  const nokCompact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
  const nokFull = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  const fmtKr = (n) => nokCompact.format(n) + ' kr';
  const fmtKrFull = (n) => nokFull.format(Math.round(n)) + ' kr';
  const fmtDate = (isoStr) => {
    const d = new Date(isoStr + 'T00:00:00');
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  function niceStep(maxVal, ticks) {
    const raw = maxVal / ticks;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step;
    if (norm < 1.5) step = 1;
    else if (norm < 3) step = 2;
    else if (norm < 7) step = 5;
    else step = 10;
    return step * mag;
  }

  function drawChart(series) {
    window.__lastSeries = series;
    const svg = document.getElementById('chart');
    const W = 800, H = 280;
    const padL = 50, padR = 16, padT = 16, padB = 28;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const values = series.map((d) => d.revenue);
    const maxVal = Math.max(...values);
    const step = niceStep(maxVal, 4);
    const niceMax = Math.ceil(maxVal / step) * step;

    const x = (i) => padL + (i / (series.length - 1)) * plotW;
    const y = (v) => padT + plotH - (v / niceMax) * plotH;

    const style = getComputedStyle(document.documentElement);
    const accent = style.getPropertyValue('--accent').trim();
    const gridline = style.getPropertyValue('--gridline').trim();
    const baseline = style.getPropertyValue('--baseline').trim();

    let svgHtml = '';

    // gridlines + y-axis labels (4 steps)
    for (let i = 0; i <= 4; i++) {
      const val = step * i;
      if (val > niceMax) continue;
      const gy = y(val);
      svgHtml += `<line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" class="${i === 0 ? 'baseline' : 'gridline'}" />`;
      svgHtml += `<text x="${padL - 10}" y="${gy + 4}" text-anchor="end" class="axis-label">${fmtKr(val)}</text>`;
    }

    // x-axis month labels: first occurrence of each month
    let lastMonth = null;
    series.forEach((d, i) => {
      const month = d.date.slice(0, 7);
      if (month !== lastMonth) {
        lastMonth = month;
        const label = new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short' });
        svgHtml += `<text x="${x(i)}" y="${H - 8}" text-anchor="start" class="axis-label">${label}</text>`;
      }
    });

    // area fill
    let areaPath = `M ${x(0)} ${y(values[0])} `;
    series.forEach((d, i) => { areaPath += `L ${x(i)} ${y(d.revenue)} `; });
    areaPath += `L ${x(series.length - 1)} ${y(0)} L ${x(0)} ${y(0)} Z`;
    svgHtml += `<path d="${areaPath}" fill="${accent}" opacity="0.1" stroke="none" />`;

    // line
    let linePath = `M ${x(0)} ${y(values[0])} `;
    series.forEach((d, i) => { if (i > 0) linePath += `L ${x(i)} ${y(d.revenue)} `; });
    svgHtml += `<path d="${linePath}" fill="none" stroke="${accent}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />`;

    // end marker + label
    const lastX = x(series.length - 1), lastY = y(values[values.length - 1]);
    svgHtml += `<circle cx="${lastX}" cy="${lastY}" r="4" fill="${accent}" stroke="var(--card)" stroke-width="2" />`;
    svgHtml += `<text x="${lastX - 8}" y="${lastY - 12}" text-anchor="end" class="end-label">${fmtKr(values[values.length - 1])}</text>`;

    // crosshair + hover dot (updated on pointer move)
    svgHtml += `<line id="crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}" class="crosshair" />`;
    svgHtml += `<circle id="hover-dot" r="5" class="hover-dot" />`;

    // transparent hit rect
    svgHtml += `<rect id="hit-rect" x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent" />`;

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = svgHtml;

    const crosshair = document.getElementById('crosshair');
    const hoverDot = document.getElementById('hover-dot');
    const hitRect = document.getElementById('hit-rect');
    const tooltip = document.getElementById('tooltip');
    const chartArea = document.getElementById('chart-area');

    function onMove(evt) {
      const rect = svg.getBoundingClientRect();
      const scaleX = W / rect.width;
      const px = (evt.clientX - rect.left) * scaleX;
      let idx = Math.round(((px - padL) / plotW) * (series.length - 1));
      idx = Math.max(0, Math.min(series.length - 1, idx));
      const d = series[idx];
      const cx = x(idx), cy = y(d.revenue);

      crosshair.setAttribute('x1', cx);
      crosshair.setAttribute('x2', cx);
      crosshair.style.opacity = 1;
      hoverDot.setAttribute('cx', cx);
      hoverDot.setAttribute('cy', cy);
      hoverDot.style.opacity = 1;

      const areaRect = chartArea.getBoundingClientRect();
      const tx = (cx / W) * areaRect.width;
      const ty = (cy / H) * areaRect.height;
      tooltip.style.left = `${tx}px`;
      tooltip.style.top = `${ty - 10}px`;
      tooltip.style.opacity = 1;
      tooltip.innerHTML = `<div class="tt-value">${fmtKrFull(d.revenue)}</div><div class="tt-date">${fmtDate(d.date)}</div>`;
    }

    function onLeave() {
      crosshair.style.opacity = 0;
      hoverDot.style.opacity = 0;
      tooltip.style.opacity = 0;
    }

    hitRect.addEventListener('pointermove', onMove);
    hitRect.addEventListener('pointerleave', onLeave);
  }

  function renderTable(series) {
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';
    series.forEach((d) => {
      const tr = document.createElement('tr');
      const tdDate = document.createElement('td');
      tdDate.textContent = d.date;
      const tdVal = document.createElement('td');
      tdVal.textContent = fmtKrFull(d.revenue);
      tr.appendChild(tdDate);
      tr.appendChild(tdVal);
      tbody.appendChild(tr);
    });
  }

  async function load() {
    const res = await fetch('/api/summary');
    const data = await res.json();

    document.getElementById('total-revenue').textContent = fmtKr(data.total_revenue);
    document.getElementById('order-count').textContent = `${data.order_count.toLocaleString()} orders`;
    document.getElementById('top-name').textContent = data.top_salesperson.name;
    document.getElementById('top-detail').innerHTML =
      `<strong>${fmtKr(data.top_salesperson.revenue)}</strong> · ${data.top_salesperson.share}% of total`;

    const first = data.series[0].date, last = data.series[data.series.length - 1].date;
    document.getElementById('chart-range').textContent = `${fmtDate(first)} – ${fmtDate(last)}`;

    drawChart(data.series);
    renderTable(data.series);
  }

  window.addEventListener('resize', () => { if (window.__lastSeries) drawChart(window.__lastSeries); });
  load();

  const chatFab = document.getElementById('chat-fab');
  const chatPanel = document.getElementById('chat-panel');
  const chatClose = document.getElementById('chat-close');
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatHistory = [];

  function openChat() {
    chatPanel.classList.add('open');
    chatInput.focus();
  }
  function closeChat() {
    chatPanel.classList.remove('open');
  }
  chatFab.addEventListener('click', () => {
    chatPanel.classList.contains('open') ? closeChat() : openChat();
  });
  chatClose.addEventListener('click', closeChat);

  function addChatMessage(text, kind) {
    const div = document.createElement('div');
    div.className = `chat-msg chat-msg-${kind}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'chat-typing';
    div.id = 'chat-typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  function hideTyping() {
    const el = document.getElementById('chat-typing');
    if (el) el.remove();
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;
    chatInput.value = '';
    addChatMessage(message, 'user');
    showTyping();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: chatHistory }),
      });
      const data = await res.json();
      hideTyping();
      if (!res.ok) {
        addChatMessage(data.message || "Something went wrong.", 'error');
        return;
      }
      addChatMessage(data.reply, 'bot');
      chatHistory.push({ role: 'user', content: message });
      chatHistory.push({ role: 'assistant', content: data.reply });
    } catch (err) {
      hideTyping();
      addChatMessage("Couldn't reach the assistant. Try again.", 'error');
    }
  });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(INDEX_HTML)
        elif self.path == "/api/summary":
            self._send_json(load_summary())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        data = self._read_json_body()
        message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not message:
            self._send_json({"error": "message is required"}, status=400)
            return

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=build_data_context())}]
        for turn in history[-10:]:
            role, content = turn.get("role"), turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        try:
            reply = call_llm(messages)
        except urllib.error.HTTPError as e:
            print(f"chat: inference HTTP error {e.code}: {e.read()[:500]}")
            self._send_json({"error": "chat_failed", "message": "The assistant is unavailable right now."}, status=502)
        except Exception as e:
            print(f"chat: inference call failed: {e}")
            self._send_json({"error": "chat_failed", "message": "The assistant is unavailable right now."}, status=502)
        else:
            self._send_json({"reply": reply})


if __name__ == "__main__":
    load_summary()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on 0.0.0.0:{PORT}")
    server.serve_forever()
