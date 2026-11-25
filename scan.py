"""Run the same VAGPR model on CSI 300 constituents with 0 < P/E TTM < 20."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

import xlrd

from vagpr import CN, RETURN_METHOD, fetch_market_data, last_complete_day, market_symbol, run, write_report

CSI = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls"
TENCENT = "https://qt.gtimg.cn/q="


def get(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
    with urlopen(request, timeout=15) as response:
        return response.read()


def constituents():
    sheet = xlrd.open_workbook(file_contents=get(CSI)).sheet_by_index(0)
    rows = [sheet.row_values(i) for i in range(1, sheet.nrows)]
    members = {str(r[4]).zfill(6): r[5] for r in rows if r[1] == "000300"}
    dates = {str(r[0]) for r in rows}
    if len(members) != 300 or len(dates) != 1:
        raise ValueError("中证指数成分表格式或数量异常")
    return members, dates.pop()


def quotes(members):
    result = {}
    codes = sorted(members)
    for start in range(0, len(codes), 50):
        batch = codes[start:start+50]
        url = TENCENT + ",".join(map(market_symbol, batch))
        response = get(url).decode("gb18030")
        for code, raw in re.findall(r'v_(?:sh|sz)(\d{6})="([^"]*)"', response):
            row = raw.split("~")
            if code in batch and len(row) > 39 and row[2] == code:
                result[code] = (row[30], row[39], row[3])
    return result


def select_quotes(members, snapshot, now):
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    cutoff = last_complete_day(now)
    valid, exclusions = [], {"missing_quote": 0, "missing_pe_or_price": 0, "different_session": 0}
    dated = {}
    for code, (timestamp, _, _) in snapshot.items():
        try:
            stamp = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        if stamp.date() <= cutoff and stamp.hour >= 15:
            dated[code] = stamp.date()
    session = max(dated.values(), default=None)
    if session is None or (cutoff-session).days > 7:
        raise ValueError("找不到有效的收盘估值快照")
    for code, name in members.items():
        if code not in snapshot:
            exclusions["missing_quote"] += 1
            continue
        _, pe_text, close_text = snapshot[code]
        pe, close = number(pe_text), number(close_text)
        if dated.get(code) != session:
            exclusions["different_session"] += 1
        elif not (math.isfinite(pe) and math.isfinite(close) and close > 0):
            exclusions["missing_pe_or_price"] += 1
        elif 0 < pe < 20:
            valid.append((code, name, pe, close))
    return session.isoformat(), valid, exclusions


def forecast_member(candidate, session, now):
    code, name, pe, close = candidate
    bars, source = fetch_market_data(now, code)
    if bars[-1].date != session or abs(bars[-1].raw_close-close) > .011:
        raise ValueError(f"日线与估值快照的日期或收盘价不一致：{bars[-1].date} / {bars[-1].raw_close}")
    result = run(bars, 5, source, code)
    return dict(result, name=name, pe_ttm=pe)


def scan(workers, prior=None):
    now = datetime.now(CN)
    members, roster_date = constituents()
    session, candidates, exclusions = select_quotes(members, quotes(members), now)
    print(f"{session}: {len(candidates)}/{len(members)} matched 0 < PE TTM < 20", file=sys.stderr, flush=True)
    forecasts, failures = [], {}
    if prior and prior["as_of"] == session:
        by_code = {f["symbol"]: f for f in prior["forecasts"]}
        for code, name, pe, close in candidates:
            old = by_code.get(code)
            if (old and old.get("forecast_method") == RETURN_METHOD
                    and old["pe_ttm"] == pe and old["raw_close"] == close):
                forecasts.append(dict(old, name=name))
        cached = {f["symbol"] for f in forecasts}
        candidates = [row for row in candidates if row[0] not in cached]
        print(f"{len(cached)} validated results reused; {len(candidates)} to process", file=sys.stderr, flush=True)
    eligible_count = len(candidates) + len(forecasts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(forecast_member, row, session, now): row for row in candidates}
        for done, task in enumerate(as_completed(jobs), 1):
            row = jobs[task]
            try:
                forecasts.append(task.result())
            except (OSError, ValueError, KeyError) as exc:
                failures[row[0]] = str(exc)
            if done % 10 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} processed", file=sys.stderr, flush=True)
    forecasts.sort(key=lambda f: (-int(f["backtest"]["passes_screen"]), -f["return"], f["symbol"]))
    return dict(as_of=session, roster_date=roster_date, universe="CSI 300",
                forecast_method=RETURN_METHOD,
                valuation_source="Tencent Finance / PE TTM (field 39)", universe_size=len(members),
                pe_eligible=eligible_count, forecast_count=len(forecasts),
                positive_count=sum(f["return"] > 0 for f in forecasts),
                positive_pass_count=sum(f["return"] > 0 and f["backtest"]["passes_screen"] for f in forecasts),
                exclusions=exclusions, failures=failures, forecasts=forecasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    prior = json.loads(args.report.read_text(encoding="utf-8")) if args.resume else None
    result = scan(args.workers, prior)
    write_report(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if k != "forecasts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
