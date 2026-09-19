"""매일 한 번 실행 — 2007년(HYG 출시 직후)부터 지금까지 실제 가격으로 현재 실전 규칙을
그대로 돌려서, 그 결과(자산가치/낙폭/연간 배당 현금흐름)를 Supabase `backtest_snapshot`에
저장한다. fabot-trade-journal 대시보드의 "일별 수익 추이" 자리가 이 데이터로 매일 갱신된다
(2026-09-19 사용자 요청 — "일별을 기준으로" 갱신).

원본 로직은 fabot-auto-trade/simulate_real_history_2008_current_rules.py와 동일(2026-09-11
최초 작성) — 매번 새로 가격을 받아오므로, 이 스크립트를 매일 돌리기만 하면 end_date가
자연스럽게 "오늘"까지 늘어난다. 다른 점은 로컬 JSON 파일 대신 Supabase에 올린다는 것뿐.

- F&G 대체지표: 실제 VIX(지수) + QQQ + IEF + HYG + LQD로 계산
- TQQQ: 2010년 2월 이후는 실제 데이터, 그 이전(2008~2010)은 QQQ의 3배로 근사
- 닷컴버블(2000~2002)은 HYG가 그때 없어서 포함하지 않음
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from price_fetcher import fetch_price_history
from price_proxy import compute_price_based_fg, WEIGHTS, INTERCEPT

START = datetime(2007, 6, 1)  # HYG 출시(2007-04) 후 여유를 두고 시작, 1년 롤링 워밍업 고려

BUY_THRESHOLDS = (20, 25, 30)
SELL_THRESHOLDS = (72, 77)
COVERED_CALL_ZONE = (35, 65)
COVERED_CALL_ALLOCATION = 0.10
COOLDOWN_DAYS = 4
CC_DIV_MONTHLY = 0.15 / 12
TAX_RATE = 0.22
TAX_EXEMPTION = 2_500_000


def run_strategy(tqqq_r, cc_r, fg_series, start_capital=10_000_000):
    cash = start_capital * 0.6
    cc_value = start_capital * 0.4
    tqqq_value = 0.0
    tqqq_cost_basis = 0.0
    tqqq_cooldown = 0
    cc_cooldown = 0
    day_count = 0
    year_realized_gain = 0.0
    history = np.empty(len(tqqq_r))
    dividends = np.zeros(len(tqqq_r))  # 그날 지급된 배당(원), 대부분 0, 21거래일마다 값 있음

    def sell_tqqq(amount):
        nonlocal tqqq_value, tqqq_cost_basis
        if tqqq_value <= 0:
            return 0.0
        frac = amount / tqqq_value
        cost_removed = tqqq_cost_basis * frac
        gain = amount - cost_removed
        tqqq_value -= amount
        tqqq_cost_basis -= cost_removed
        return gain

    for i in range(len(tqqq_r)):
        f = fg_series[i]
        tqqq_value *= (1 + tqqq_r[i])
        cc_value *= (1 + cc_r[i])

        day_count += 1
        if day_count % 21 == 0:
            div = cc_value * CC_DIV_MONTHLY
            cash += div
            dividends[i] = div
        if day_count % 252 == 0:
            taxable = max(0.0, year_realized_gain - TAX_EXEMPTION)
            cash -= taxable * TAX_RATE
            year_realized_gain = 0.0

        if tqqq_cooldown > 0:
            tqqq_cooldown -= 1
        if cc_cooldown > 0:
            cc_cooldown -= 1

        s50, s100 = SELL_THRESHOLDS
        if tqqq_value > 0:
            if f >= s100:
                amt = tqqq_value
                year_realized_gain += sell_tqqq(amt)
                cash += amt
            elif f >= s50:
                amt = tqqq_value * 0.5
                year_realized_gain += sell_tqqq(amt)
                cash += amt

        t100, t50, t25 = BUY_THRESHOLDS
        if tqqq_cooldown == 0 and cash > 0:
            if f <= t100:
                buy = cash * 1.0
            elif f <= t50:
                buy = cash * 0.5
            elif f <= t25:
                buy = cash * 0.25
            else:
                buy = 0.0
            if buy > 0:
                cash -= buy
                tqqq_value += buy
                tqqq_cost_basis += buy
                tqqq_cooldown = COOLDOWN_DAYS

        cc_lo, cc_hi = COVERED_CALL_ZONE
        if cc_cooldown == 0 and cash > 0 and cc_lo <= f <= cc_hi:
            add = cash * COVERED_CALL_ALLOCATION
            cash -= add
            cc_value += add
            cc_cooldown = COOLDOWN_DAYS

        history[i] = cash + cc_value + tqqq_value

    return history, dividends


def build_result() -> dict:
    print("실제 가격 데이터 받는 중 (QQQ, VIX, IEF, HYG, LQD, TQQQ)...")
    qqq = fetch_price_history("QQQ", START)
    vix = fetch_price_history("^VIX", START)
    ief = fetch_price_history("IEF", START)
    hyg = fetch_price_history("HYG", START)
    lqd = fetch_price_history("LQD", START)
    tqqq_real = fetch_price_history("TQQQ", START)  # 2010-02-11부터만 실제로 값이 있음

    fg = compute_price_based_fg(qqq, vix, ief, hyg, lqd, weights=WEIGHTS, intercept=INTERCEPT)

    df = pd.DataFrame({"qqq": qqq, "fg": fg, "tqqq_real": tqqq_real}).dropna(subset=["qqq", "fg"])
    df["qqq_ret"] = df["qqq"].pct_change()
    df["tqqq_ret_real"] = df["tqqq_real"].pct_change()
    df["tqqq_ret"] = df["tqqq_ret_real"].fillna(df["qqq_ret"] * 3)
    df = df.dropna(subset=["qqq_ret", "tqqq_ret", "fg"])

    approx_days = int(df["tqqq_ret_real"].isna().sum())
    print(f"실제 데이터 구간: {df.index.min().date()} ~ {df.index.max().date()} "
          f"({len(df)/252:.1f}년, {len(df)}거래일, 그중 TQQQ 근사 구간 {approx_days}거래일)")

    tqqq_ret = df["tqqq_ret"].values
    cc_ret = df["qqq_ret"].values * 0.9  # 커버드콜 beta 0.9 근사
    fg_vals = df["fg"].values
    dates = df.index

    history, dividends = run_strategy(tqqq_ret, cc_ret, fg_vals)
    running_max = np.maximum.accumulate(history)
    drawdown = (history - running_max) / running_max

    final_value = history[-1]
    years = len(history) / 252
    cagr = (final_value / 10_000_000) ** (1 / years) - 1
    mdd = drawdown.min()
    mdd_date = dates[int(np.argmin(drawdown))]

    print(f"실제 {years:.1f}년({dates[0].date()}~{dates[-1].date()}) 결과: "
          f"최종 {final_value:,.0f}원 (배수 {final_value/10_000_000:.1f}배), "
          f"CAGR {cagr*100:.1f}%, MDD {mdd*100:.1f}% ({mdd_date.date()} 근처)")

    STEP = 21
    points = []
    for i in range(0, len(history), STEP):
        points.append({
            "date": dates[i].strftime("%Y-%m"),
            "value": round(float(history[i])),
            "drawdown": round(float(drawdown[i]), 4),
        })
    if (len(history) - 1) % STEP != 0:
        points.append({
            "date": dates[-1].strftime("%Y-%m"),
            "value": round(float(history[-1])),
            "drawdown": round(float(drawdown[-1]), 4),
        })

    dividend_by_year = pd.Series(dividends, index=dates).groupby(lambda d: d.year).sum()
    yearly_dividends = [
        {"year": int(year), "dividend": round(float(amount))}
        for year, amount in dividend_by_year.items()
    ]

    value_series = pd.Series(history, index=dates)
    drawdown_series = pd.Series(drawdown, index=dates)
    year_end_value = value_series.groupby(lambda d: d.year).last()
    year_mdd = drawdown_series.groupby(lambda d: d.year).min()

    yearly_summary = []
    prev_value = 10_000_000
    for year in year_end_value.index:
        end_value = float(year_end_value[year])
        yearly_summary.append({
            "year": int(year),
            "return": round(end_value / prev_value - 1, 4),
            "mdd": round(float(year_mdd[year]), 4),
            "dividend": round(float(dividend_by_year[year])),
            "year_end_value": round(end_value),
        })
        prev_value = end_value

    return {
        "generated_at": datetime.now().isoformat(),
        "start_capital": 10_000_000,
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "years": round(years, 1),
        "tqqq_approx_days": approx_days,
        "final_value": round(float(final_value)),
        "cagr": round(float(cagr), 4),
        "mdd": round(float(mdd), 4),
        "mdd_date": mdd_date.strftime("%Y-%m"),
        "points": points,
        "yearly_dividends": yearly_dividends,
        "yearly_summary": yearly_summary,
    }


def save_to_supabase(result: dict) -> None:
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    response = requests.post(
        f"{supabase_url}/rest/v1/backtest_snapshot?on_conflict=id",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json={"id": 1, "data": result, "updated_at": datetime.now().isoformat()},
        timeout=20,
    )
    response.raise_for_status()


if __name__ == "__main__":
    result = build_result()
    save_to_supabase(result)
    print("backtest_snapshot 저장 완료")
