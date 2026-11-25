"""Local, causal two-output additive Gaussian process for mainland A-shares."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
from threading import Thread
from typing import NamedTuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Small Cholesky factorizations are faster without a large BLAS thread pool.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

CODE = "603011"
CN = timezone(timedelta(hours=8))
HORIZONS = (1, 5, 10)
FIELDS = ("date", "open", "close", "high", "low", "volume", "raw_close")
SOURCE = "Tencent Finance / hfq daily OHLCV + raw daily close"
EAST_SOURCE = "Eastmoney / hfq daily OHLCV + raw daily close"
FEATURES = ("ret1", "ret5", "ret20", "vol20", "intraday_range", "volume_ratio20")
WINDOW = 150
Z80 = 1.2815515655446004
RETURN_METHOD = "vagpr-return-v2"
CAL_MIN = 20
CAL_WINDOW = 40
WEIGHTS = np.array([0, .25, .50, .75, 1])
INITIAL = np.array([-1.4, -2.2, 0.0, 0.0, -0.8])
BOUNDS = [(-8, 3), (-8, 3), (-2.3, 2.3), (-2.3, 2.3), (-5, 2)]


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    close: float
    high: float
    low: float
    volume: float
    raw_close: float


def validate_bars(bars):
    if len(bars) < 220:
        raise ValueError("至少需要 220 条完整日线。")
    prev = date.min
    for b in bars:
        dt = date.fromisoformat(b.date)
        if dt.isoformat() != b.date or dt <= prev:
            raise ValueError("日期必须是 YYYY-MM-DD，且严格递增、不能重复。")
        prev = dt
        if not all(math.isfinite(v) and v > 0 for v in asdict(b).values() if not isinstance(v, str)):
            raise ValueError(f"{b.date} 含无效价格或成交量。")
        if not (b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high):
            raise ValueError(f"{b.date} 的 OHLC 关系无效。")
    return bars


def market_symbol(symbol):
    if not re.fullmatch(r"[036]\d{5}", symbol, flags=re.ASCII):
        raise ValueError("请输入以 0、3 或 6 开头的六位沪深股票代码。")
    return ("sh" if symbol[0] == "6" else "sz") + symbol


def tencent(adjustment, symbol=CODE, modern=False):
    key = market_symbol(symbol)
    param = f"{key},day,,,500," + ("hfq" if adjustment else "bfq")
    endpoint = "newfqkline" if modern else "fqkline"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/{endpoint}/get?" + urlencode({"param": param}, safe=",")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if result.get("code") != 0:
        raise ValueError("腾讯行情暂不可用。")
    return result["data"][key]


def last_complete_day(now):
    return now.date() - timedelta(days=1) if now.time() < time(15, 15) else now.date()


def fetch_bars(now=None, symbol=CODE, modern=False):
    now = now or datetime.now(CN)
    cutoff = last_complete_day(now).isoformat()
    adjusted, unadjusted = tencent(True, symbol, modern), tencent(False, symbol, modern)
    raw = {r[0]: float(r[2]) for r in unadjusted["day"] if r[0] <= cutoff}
    hfq = [r for r in adjusted["hfqday"] if r[0] <= cutoff]
    if not hfq or not raw or hfq[-1][0] != max(raw):
        raise ValueError("复权与未复权行情截止日不一致。")
    # The providers can return different lengths; use only their overlapping span.
    first_raw_day = min(raw)
    bars = [
        Bar(r[0], *map(float, r[1:6]), raw[r[0]])
        for r in hfq if r[0] >= first_raw_day
    ]
    quote = unadjusted.get("qt", {}).get(market_symbol(symbol), [])
    if len(quote) > 30 and str(quote[30])[:8] == bars[-1].date.replace("-", ""):
        if quote[2] != symbol or abs(float(quote[3]) - bars[-1].raw_close) > 0.011:
            raise ValueError("快照和日线的收盘价不一致，请稍后重试。")
    if (now.date() - date.fromisoformat(bars[-1].date)).days > 7:
        raise ValueError(f"在线行情停留在 {bars[-1].date}，请核对停牌或数据源。")
    return validate_bars(bars)


def eastmoney_bars(now, symbol):
    market_symbol(symbol)
    secid = ("1." if symbol[0] == "6" else "0.") + symbol
    cutoff = last_complete_day(now).isoformat()

    def rows(adjustment):
        host = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        query = dict(secid=secid, fields1="f1,f2,f3,f4,f5,f6",
                     fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                     klt=101, fqt=adjustment, lmt=500, end=20500101)
        request = Request(host + "?" + urlencode(query, safe=","),
                          headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=12) as handle:
            data = json.load(handle)["data"]
        if not data or data.get("code") != symbol:
            raise ValueError("东方财富日线与请求的代码不一致。")
        return [row.split(",") for row in data["klines"] if row[:10] <= cutoff]

    adjusted, unadjusted = rows(2), rows(0)
    raw = {r[0]: float(r[2]) for r in unadjusted}
    if not raw or not adjusted or adjusted[-1][0] != max(raw):
        raise ValueError("东方财富的两个复权口径截止日不一致。")
    first_raw = min(raw)
    series = [Bar(r[0], *map(float, r[1:6]), raw[r[0]]) for r in adjusted if r[0] >= first_raw]
    if (now.date() - date.fromisoformat(series[-1].date)).days > 7:
        raise ValueError("备用数据源的收盘数据已过期。")
    return validate_bars(series)


def fetch_market_data(now=None, symbol=CODE):
    now = now or datetime.now(CN)
    market_symbol(symbol)
    errors = []
    for modern in (False, True):
        try:
            return fetch_bars(now, symbol, modern), SOURCE
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
    try:
        return eastmoney_bars(now, symbol), EAST_SOURCE
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"腾讯：{' / '.join(errors)}；东方财富：{error}") from error


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(FIELDS) <= set(reader.fieldnames):
            raise ValueError("CSV 缺少必要列：" + ", ".join(FIELDS))
        bars = [Bar(r["date"], *(float(r[f]) for f in FIELDS[1:])) for r in reader]
    if bars and bars[-1].date > last_complete_day(datetime.now(CN)).isoformat():
        raise ValueError("CSV 包含尚未收盘或未来交易日。")
    return validate_bars(bars)


def save_csv(path, bars):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(asdict(b) for b in bars)


class Series(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    closes: np.ndarray
    dates: tuple


def make_series(bars, h):
    p = np.log([b.close for b in bars])
    v = np.log([b.volume for b in bars])
    day_return = np.diff(p)
    x, y = [], []
    for i in range(20, len(p)):
        x.append((p[i] - p[i-1], p[i] - p[i-5], p[i] - p[i-20],
                  np.sqrt(np.mean(day_return[i-20:i]**2)),
                  np.log(bars[i].high / bars[i].low),
                  v[i] - np.median(v[i-20:i])))
        if i + h < len(p):
            rms = np.sqrt(np.mean(day_return[i:i+h]**2))
            y.append((p[i+h] - p[i], np.log(max(rms, 1e-5))))
    return Series(np.array(x), np.array(y), np.array([b.raw_close for b in bars[20:]]),
                  tuple(b.date for b in bars[20:]))


def centered_rbf(a, b, ell=1.25):
    # Condition the prior on zero integral under the standardized N(0,1) marginal.
    mean = lambda t: ell / np.sqrt(ell**2 + 1) * np.exp(-t**2 / (2 * (ell**2 + 1)))
    return (np.exp(-(a-b)**2 / (2*ell**2))
            - mean(a)*mean(b)/(ell/np.sqrt(ell**2+2)))


def scalar_orders(a, b):
    parts = centered_rbf(a[:, None, :], b[None, :, :])
    sums = parts.sum(axis=-1)
    d = parts.shape[-1]
    pairs = (sums**2 - (parts**2).sum(axis=-1)) / (d*(d-1))
    return sums / d, pairs


def task_matrix(log_amplitude, corr):
    rho = np.tanh(corr)
    return np.exp(log_amplitude) * np.array([[1.0, rho], [rho, 1.0]])


def covariance(kernels, theta):
    return np.kron(kernels[0], task_matrix(theta[0], theta[2])) + np.kron(
        kernels[1], task_matrix(theta[1], theta[3]))


def stabilize_scale(values):
    return np.maximum(np.std(values, axis=0), 1e-8)


def gp_predict(x, y, point, theta, refit):
    loc, scale = x.mean(axis=0), stabilize_scale(x)
    x, point = (x-loc)/scale, (point[None]-loc)/scale
    yloc, yscale = y.mean(axis=0), stabilize_scale(y)
    target = ((y-yloc)/yscale).ravel()
    k = scalar_orders(x, x)
    diag = np.diag_indices(len(target))

    def factor(t):
        cov = covariance(k, t)
        cov[diag] += np.exp(t[4]) + 1e-8
        chol = cho_factor(cov, lower=True, check_finite=False)
        return chol, cho_solve(chol, target, check_finite=False)

    def objective(t):
        chol, alpha = factor(t)
        inverse = cho_solve(chol, np.eye(len(target)), check_finite=False)
        weight = (inverse - np.outer(alpha, alpha)).reshape(len(x), 2, len(x), 2) / 2
        reduced = [np.einsum("iajb,ij->ab", weight, component) for component in k]
        grad = np.empty(5)
        for i, reduced_i in enumerate(reduced):
            grad[i] = np.sum(reduced_i * task_matrix(t[i], t[i+2]))
            grad[i+2] = (reduced_i[0, 1] + reduced_i[1, 0]) * np.exp(t[i]) * (
                1 - np.tanh(t[i+2])**2)
        grad[4] = np.exp(t[4]) * (np.trace(inverse) - alpha @ alpha) / 2
        nll = (target @ alpha)/2 + np.log(np.diag(chol[0])).sum()
        return nll, grad

    if refit:
        result = minimize(objective, theta, method="L-BFGS-B", jac=True,
                          bounds=BOUNDS, options={"maxiter": 25})
        if np.isfinite(result.fun) and result.fun <= objective(theta)[0] + 1e-9:
            theta = result.x
    chol, alpha = factor(theta)
    cross = covariance(scalar_orders(point, x), theta)
    prior = covariance(scalar_orders(point, point), theta)
    solved = solve_triangular(chol[0], cross.T, lower=True, check_finite=False)
    posterior = prior - solved.T @ solved + np.eye(2)*np.exp(theta[4])
    mean = cross @ alpha * yscale + yloc
    sd = np.sqrt(np.maximum(np.diag(posterior), 1e-12)) * yscale
    return mean, sd, theta


def training_indices(origin, h):
    stop = origin-h+1  # Only labels whose final close is already observable.
    return slice(max(0, stop-WINDOW), stop)


def metrics(predictions, realized):
    errors = predictions - realized
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2)))}


class ReturnCalibration(NamedTuple):
    past_centers: np.ndarray
    past_widths: np.ndarray
    center: float
    width: float
    weight: float
    n: int
    ready: bool


def calibrate_returns(mu, sigma, actual, final_mu, final_sigma):
    """Replay calibration: each issued forecast uses only settled labels."""
    mu, sigma, actual = (np.asarray(a, dtype=float) for a in (mu, sigma, actual))
    realized = np.expm1(actual)
    grid = np.expm1(mu[:, None]*WEIGHTS)
    centers, widths, errors = [], [], []
    for i in range(len(actual)+1):
        start = max(0, i-CAL_WINDOW)
        ready = i >= CAL_MIN
        if ready:
            losses = np.mean(np.abs(grid[start:i]-realized[start:i, None]), axis=0)
            # In a tie, prefer the smaller weight, including the no-change baseline.
            weight = float(WEIGHTS[np.flatnonzero(losses <= losses.min() + 1e-12)[0]])
            residuals = sorted(errors[start:i])
            rank = min(len(residuals), math.ceil((len(residuals)+1)*.8))
            multiplier = residuals[rank-1]
        else:
            weight, multiplier = 0.0, Z80
        if i == len(actual):
            return ReturnCalibration(np.array(centers), np.array(widths), weight*final_mu,
                                     multiplier*max(float(final_sigma), 1e-6),
                                     weight, min(i, CAL_WINDOW), ready)
        center, scale = weight*mu[i], max(sigma[i], 1e-6)
        centers.append(center)
        widths.append(multiplier*scale)
        errors.append(abs(actual[i]-center)/scale)


def run(bars, h=5, source=SOURCE, symbol=CODE):
    if h not in HORIZONS:
        raise ValueError("不支持的预测期。")
    x, y, closes, dates = make_series(validate_bars(bars), h)
    current = len(x)-1
    origins = list(range(current-h, max(WINDOW+h-2, current-h*80)-1, -h))[::-1]
    theta = INITIAL.copy()
    records = []
    for pos, i in enumerate(origins):
        train = training_indices(i, h)
        mean, sd, theta = gp_predict(x[train], y[train], x[i], theta, pos % 6 == 0)
        records.append((mean[0], sd[0], y[i, 0], mean[1], y[i, 1], x[i, 3]))
    mu, sigma, actual, vol, actual_vol, vol_base = np.array(records).T
    train = training_indices(current, h)
    mean, sd, _ = gp_predict(x[train], y[train], x[current], theta, True)
    cal = calibrate_returns(mu, sigma, actual, mean[0], sd[0])
    terminal = np.expm1([cal.center-cal.width, cal.center, cal.center+cal.width])
    historical = slice(CAL_MIN, None)
    target = actual[historical]
    n = len(target)
    model = baseline = raw_model = hit = always_up = always_down = interval = raw_interval = None
    direction_n = 0
    if n:
        realized = np.expm1(target)
        predicted, raw_predicted = cal.past_centers[historical], mu[historical]
        model = metrics(np.expm1(predicted), realized)
        raw_model = metrics(np.expm1(raw_predicted), realized)
        baseline = metrics(np.zeros(n), realized)
        active = (target != 0) & (predicted != 0)
        direction_n = int(active.sum())
        if direction_n:
            directions = np.sign(target[active])
            hit = float(np.mean(np.sign(predicted[active]) == directions))
            always_up = float(np.mean(directions > 0))
            always_down = float(np.mean(directions < 0))
        interval = float(np.mean(np.abs(target-predicted) <= cal.past_widths[historical]))
        raw_interval = float(np.mean(np.abs(target-raw_predicted) <= Z80*sigma[historical]))
    validated = bool(n >= 30 and direction_n >= 30 and model["mae"] < baseline["mae"]
                     and model["rmse"] < baseline["rmse"] and hit > max(always_up, always_down))
    return {
        "forecast_method": RETURN_METHOD,
        "symbol": symbol, "source": source, "data_from": bars[0].date, "as_of": dates[-1],
        "daily_bars": len(bars), "horizon": h, "raw_close": float(closes[-1]),
        "return": float(terminal[1]), "raw_return": float(np.expm1(mean[0])),
        "return_lower": float(terminal[0]), "return_upper": float(terminal[2]),
        "reference_price": float(closes[-1]*(1+terminal[1])),
        "reference_lower": float(closes[-1]*(1+terminal[0])),
        "reference_upper": float(closes[-1]*(1+terminal[2])),
        "calibration": {"weight": cal.weight, "n": cal.n, "ready": cal.ready, "target_coverage": .8},
        "daily_rms_volatility": float(np.exp(mean[1])),
        "backtest": {"n": n, "vagpr_origins": len(origins),
                     "first_origin": dates[origins[CAL_MIN]] if n else None,
                     "last_origin": dates[origins[-1]] if n else None,
                     "model": model, "raw_model": raw_model, "flat": baseline, "direction_hit": hit,
                     "direction_n": direction_n, "always_up": always_up, "always_down": always_down,
                     "interval_coverage": interval, "raw_interval_coverage": raw_interval, "passes_screen": validated,
                     "volatility_model_mae": float(np.mean(np.abs(np.exp(vol)-np.exp(actual_vol)))),
                     "volatility_baseline_mae": float(np.mean(np.abs(vol_base-np.exp(actual_vol))))},
    }


def write_report(path, result):
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def price_ticks(prices, target):
    lo, hi = min(prices), max(prices)
    pad = max((hi-lo)*.07, abs(hi)*.003, .01)
    lo, hi = max(0, lo-pad), hi+pad
    interval = (hi-lo)/max(2, target-1)
    scale = 10**math.floor(math.log10(interval))
    step = next(k*scale for k in (1, 2, 2.5, 5, 10) if k*scale >= interval)
    return [i*step for i in range(math.floor(lo/step), math.ceil(hi/step)+1)]


def date_ticks(count, pixels):
    n = min(count, max(2, pixels//120+1))
    return [round(i*(count-1)/(n-1)) for i in range(n)] if n > 1 else [0]


def gui(initial_csv=None, initial_symbol=CODE, initial_horizon=5):
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("VAGPR A 股涨跌幅预测器")
    root.geometry("960x680")
    root.minsize(750, 610)
    bg, ink, muted, accent = "#f5f7f8", "#152731", "#4b626b", "#10685d"
    root.configure(background=bg)
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=bg)
    style.configure("TButton", padding=(12, 7), font=("Microsoft YaHei UI", 10))
    style.configure("TLabel", background=bg, foreground=ink, font=("Microsoft YaHei UI", 11))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 22, "bold"))
    style.configure("Muted.TLabel", foreground=muted, font=("Microsoft YaHei UI", 10))
    style.configure("Value.TLabel", foreground=accent, font=("Microsoft YaHei UI", 22, "bold"))
    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)
    title = tk.StringVar(value="合锻智能 / 603011" if initial_symbol == CODE else initial_symbol)
    ttk.Label(frame, textvariable=title, style="Title.TLabel").pack(anchor="w")
    status = tk.StringVar(value="尚未载入行情")
    ttk.Label(frame, textvariable=status, style="Muted.TLabel").pack(anchor="w", pady=(3, 15))
    toolbar = ttk.Frame(frame)
    toolbar.pack(fill="x")
    results = queue.Queue()
    state = {"bars": None, "result": None}
    symbol_var = tk.StringVar(value=initial_symbol)
    ttk.Label(toolbar, text="股票代码").pack(side="left", padx=(0, 5))
    symbol_input = ttk.Entry(toolbar, width=9, textvariable=symbol_var)
    symbol_input.pack(side="left", padx=(0, 9))
    horizon = ttk.Combobox(toolbar, values=HORIZONS, width=3, state="readonly")
    horizon.set(initial_horizon)
    summary, details = tk.StringVar(value="—"), tk.StringVar(value="")
    tools = ttk.Frame(frame)
    span_choice = ttk.Combobox(tools, values=(30, 60, 120, 250), width=4, state="readonly")
    span_choice.set(60)
    price_mode = ttk.Combobox(tools, values=("后复权", "未复权"), width=7, state="readonly")
    price_mode.set("后复权")
    ttk.Label(tools, text="日线").pack(side="left", padx=(0, 5))
    span_choice.pack(side="left", padx=(0, 12))
    price_mode.pack(side="left")
    graph = tk.Canvas(frame, background="#ffffff", highlightthickness=1, highlightbackground="#d9e2e6", height=190)
    axis_font = tkfont.Font(family="Microsoft YaHei UI", size=9)
    chart = {}

    def graph_draw(_=None):
        graph.delete("all")
        chart.clear()
        bars = state["bars"]
        if not bars:
            graph.create_text(16, 16, anchor="nw", fill=muted, text="尚无日线", font=axis_font)
            return
        bars = bars[-int(span_choice.get()):]
        mode = price_mode.get()
        prices = [b.raw_close if mode == "未复权" else b.close for b in bars]
        width, height = graph.winfo_width(), graph.winfo_height()
        if width < 280 or height < 125:
            return
        ticks = price_ticks(prices, min(7, max(3, (height-60)//30)))
        decimals = 3 if ticks[1]-ticks[0] < .10 else 2
        labels = [f"{v:,.{decimals}f}" for v in ticks]
        left, right = max(axis_font.measure(t) for t in labels)+22, width-34
        top, bottom = 41, height-48
        x = lambda i: left+(right-left)*i/(len(bars)-1)
        y = lambda p: bottom-(bottom-top)*(p-ticks[0])/(ticks[-1]-ticks[0])
        graph.create_text(12, 17, anchor="w", fill=ink, font=axis_font,
                          text=f"收盘价（元，{mode}）")
        graph.create_text(width-14, 17, anchor="e", fill=muted, font=axis_font,
                          text=f"{bars[0].date} 至 {bars[-1].date}")
        for level, label in zip(ticks, labels):
            yy = y(level)
            graph.create_line(left, yy, right, yy, fill="#e6edef")
            graph.create_text(left-8, yy, anchor="e", fill=muted, font=axis_font, text=label)
        for i in date_ticks(len(bars), max(0, right-left)):
            xx = x(i)
            graph.create_line(xx, top, xx, bottom, fill="#edf1f2", dash=(2, 4))
            anchor = "nw" if i == 0 else "ne" if i == len(bars)-1 else "n"
            graph.create_text(xx, bottom+6, anchor=anchor, fill=muted, font=axis_font, text=bars[i].date)
        graph.create_line(left, top, left, bottom, right, bottom, fill="#a7b8bd")
        graph.create_text((left+right)/2, height-6, anchor="s", fill=muted, font=axis_font,
                          text="交易日期（非交易日不占位）")
        points = [(x(i), y(p)) for i, p in enumerate(prices)]
        graph.create_line(*[coord for pair in points for coord in pair], fill=accent, width=2)
        xx, yy = points[-1]
        graph.create_oval(xx-3, yy-3, xx+3, yy+3, fill=accent, outline="")
        chart.update(bars=bars, prices=prices, points=points, bounds=(left, right, top, bottom), mode=mode)

    def graph_hover(event):
        graph.delete("cursor")
        if not chart:
            return
        left, right, top, bottom = chart["bounds"]
        if not (left <= event.x <= right and top <= event.y <= bottom):
            return
        i = round((event.x-left)/(right-left)*(len(chart["bars"])-1))
        xx, yy = chart["points"][i]
        label = f'{chart["bars"][i].date}   {chart["prices"][i]:,.3f} 元'
        graph.create_line(xx, top, xx, bottom, fill=muted, dash=(3, 3), tags="cursor")
        graph.create_oval(xx-4, yy-4, xx+4, yy+4, fill=accent, outline="white", tags="cursor")
        text_width = axis_font.measure(label)+14
        px = xx+10 if xx+text_width+10 <= right else xx-text_width-10
        py = max(top+3, min(yy-29, bottom-26))
        graph.create_rectangle(px, py, px+text_width, py+22, fill=ink, outline="", tags="cursor")
        graph.create_text(px+7, py+11, anchor="w", fill="white", font=axis_font, text=label, tags="cursor")

    def show(result, bars, source):
        state.update(result=result, bars=bars, source=source)
        code = result["symbol"]
        symbol_var.set(code)
        title.set("合锻智能 / 603011" if code == CODE else f"行情研究 / {code}")
        b, cal = result["backtest"], result["calibration"]
        movement = f'{result["return"]:+.2%}' if result["return"] else "0.00%"
        summary.set(f'{result["horizon"]} 日累计涨跌幅   {movement}')
        lines = [
            f'最新收盘 ¥{result["raw_close"]:.2f}   ·   折算参考 ¥{result["reference_price"]:.2f}'
            f'   ·   VAGPR 原始估计 {result["raw_return"]:+.2%}',
            f'80% {"目标区间" if cal["ready"] else "未校准参考区间"}：'
            f'{result["return_lower"]:+.2%} 至 {result["return_upper"]:+.2%}'
            + (f'   ·   样本外覆盖 {b["interval_coverage"]:.1%}' if b["n"] else ""),
        ]
        if b["n"]:
            lines.append(
                f'样本外 {b["n"]} 次 · MAE（百分点）：'
                f'校正 {100*b["model"]["mae"]:.2f} / 原始 {100*b["raw_model"]["mae"]:.2f}'
                f' / 不变 {100*b["flat"]["mae"]:.2f}'
            )
        if not cal["ready"]:
            note = "校正样本不足 · 主预测暂用不变基准"
        elif not cal["weight"]:
            note = "回测选择不变基准 · 0% 并不表示价格不会变"
        elif not b["passes_screen"] or result["return_lower"] <= 0 <= result["return_upper"]:
            note = f'原始估计权重 {cal["weight"]:.0%} · 方向证据不足'
        else:
            note = f'原始估计权重 {cal["weight"]:.0%} · 通过历史基准筛查，不构成保证'
        details.set("\n".join([*lines, note]))
        stale = (datetime.now(CN).date() - date.fromisoformat(result["as_of"])).days > 7
        status.set(f'{result["as_of"]} 收盘 · {len(bars)} 根日线 · {source}'
                   + (" · 历史截点" if stale else ""))
        graph_draw()

    def poll():
        try:
            value = results.get_nowait()
        except queue.Empty:
            root.after(75, poll)
            return
        online.configure(state="normal")
        local.configure(state="normal")
        rerun.configure(state="normal" if state["bars"] or not isinstance(value, Exception) else "disabled")
        horizon.configure(state="readonly")
        symbol_input.configure(state="normal")
        if isinstance(value, Exception):
            status.set("运行失败 · 保留上次结果" if state["result"] else "运行失败")
            messagebox.showerror("无法运行", str(value))
        else:
            show(*value)
            export.configure(state="normal")
            csv_export.configure(state="normal")

    def start(reader, source, symbol=None):
        h = int(horizon.get())
        code = symbol or symbol_var.get().strip()
        try:
            market_symbol(code)
        except ValueError as error:
            messagebox.showerror("股票代码无效", str(error))
            return
        for control in (online, local, rerun, horizon, symbol_input):
            control.configure(state="disabled")
        status.set("准备数据并进行滚动样本外回测…")

        def work():
            try:
                loaded = reader(code)
                if source is None:
                    bars, used = loaded
                    provider = "腾讯行情" if used == SOURCE else "东方财富（备用）"
                else:
                    bars, provider = loaded, source
                results.put((run(bars, h, provider, code), bars, provider))
            except Exception as error:
                results.put(error)
        Thread(target=work, daemon=True).start()
        root.after(75, poll)

    def choose_csv():
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            start(lambda _: load_csv(path), "本地 CSV（代码与口径由导入者确认）")

    def save_output(is_csv):
        suffix = ".csv" if is_csv else ".json"
        code = state["result"]["symbol"]
        path = filedialog.asksaveasfilename(defaultextension=suffix, initialfile=f"{code}{suffix}",
                                           filetypes=[(suffix[1:].upper(), f"*{suffix}")])
        if path:
            try:
                (save_csv if is_csv else write_report)(path, state["bars" if is_csv else "result"])
            except OSError as error:
                messagebox.showerror("无法写入", str(error))

    online = ttk.Button(toolbar, text="更新行情",
                        command=lambda: start(lambda code: fetch_market_data(symbol=code), None))
    local = ttk.Button(toolbar, text="导入 CSV", command=choose_csv)
    rerun = ttk.Button(toolbar, text="重新计算", state="disabled",
                      command=lambda: start(lambda _: state["bars"], state["source"], state["result"]["symbol"]))
    online.pack(side="left", padx=(0, 6))
    local.pack(side="left", padx=(0, 6))
    ttk.Label(toolbar, text="交易日").pack(side="left", padx=(6, 4))
    horizon.pack(side="left", padx=(0, 6))
    rerun.pack(side="left")
    ttk.Separator(frame).pack(fill="x", pady=14)
    ttk.Label(frame, textvariable=summary, style="Value.TLabel").pack(anchor="w")
    ttk.Label(frame, textvariable=details, justify="left").pack(anchor="w", pady=(4, 12))
    tools.pack(fill="x", pady=(0, 6))
    graph.pack(fill="both", expand=True)
    graph.bind("<Configure>", graph_draw)
    graph.bind("<Motion>", graph_hover)
    graph.bind("<Leave>", lambda _: graph.delete("cursor"))
    span_choice.bind("<<ComboboxSelected>>", graph_draw)
    price_mode.bind("<<ComboboxSelected>>", graph_draw)
    foot = ttk.Frame(frame)
    foot.pack(fill="x", pady=(12, 0))
    export = ttk.Button(foot, text="导出报告", state="disabled", command=lambda: save_output(False))
    csv_export = ttk.Button(foot, text="导出行情", state="disabled", command=lambda: save_output(True))
    export.pack(side="left", padx=(0, 6))
    csv_export.pack(side="left")
    ttk.Label(foot, text="历史研究 · 非投资建议", style="Muted.TLabel").pack(side="right")
    if initial_csv:
        root.after(50, lambda: start(lambda _: load_csv(initial_csv), "本地 CSV", initial_symbol))
    root.mainloop()


def main():
    if len(os.sys.argv) == 1:
        gui()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fetch", action="store_true")
    action.add_argument("--csv", type=Path)
    action.add_argument("--chart-csv", type=Path)
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--symbol", default=CODE)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=5)
    parser.add_argument("--save-csv", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    market_symbol(args.symbol)
    if args.chart_csv:
        gui(args.chart_csv, args.symbol, args.horizon)
        return
    if args.self_test:
        # Windowless bundles have no stdout. A report is usable in either build.
        samples = np.arange(280)
        prices = 15*np.exp(.001*samples + .025*np.sin(samples/4))
        dates = [date(2025, 1, 1)+timedelta(days=int(i)) for i in samples]
        bars = [Bar(day.isoformat(), p, p, p*1.01, p*.99, 1000+i%19, p)
                for i, (day, p) in enumerate(zip(dates, prices))]
        source = "synthetic self-test"
    else:
        bars, source = ((load_csv(args.csv), "user CSV") if args.csv
                        else fetch_market_data(symbol=args.symbol))
    result = run(bars, args.horizon, source, args.symbol)
    if args.save_csv:
        save_csv(args.save_csv, bars)
    if args.report:
        write_report(args.report, result)
    if os.sys.stdout is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
