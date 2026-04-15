"""
나스닥 하이브리드 오라클 전략 — 백테스트 & 검증 대시보드
개인 교육용 | 투자 권유 아님

전략 로직:
  Entry : VIX > threshold AND QQQ 20일 수익률 > momentum_threshold AND TNX 60일 변화율 < tnx_threshold
  Action: QQQ 보유분 50% → SOXL 스위칭
  Exit  : QQQ 20일 수익률 < 0 (추세 꺾임) → SOXL → QQQ 전량 복귀
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="오라클 전략 검증기",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"]          { background:#08080f; }
[data-testid="stSidebar"]                   { background:#0e0e1a; border-right:1px solid #1e1e30; }
[data-testid="stSidebar"] label             { color:#8888aa !important; }
section[data-testid="stSidebar"] h2         { color:#a0a8ff !important; }
h1,h2,h3                                    { color:#ddddf0 !important; letter-spacing:-0.02em; }
div[data-testid="stMetric"]                 { background:#10101e; border:0.5px solid #22223a; border-radius:10px; padding:.9rem 1rem !important; }
div[data-testid="stMetricValue"]            { color:#ddddf0 !important; }
.stTabs [data-baseweb="tab"]                { color:#5a5a7a; background:transparent; font-size:15px; }
.stTabs [aria-selected="true"]              { color:#a0a8ff !important; border-bottom:2px solid #6366f1 !important; }
.stTabs [data-baseweb="tab-list"]           { background:transparent; border-bottom:1px solid #1e1e30; gap:8px; }
.verdict-box                                { border-radius:12px; padding:1.25rem 1.5rem; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(start: str = "2014-01-01"):
    """Yahoo Finance에서 4개 심볼 다운로드 후 병합."""
    end = datetime.today().strftime("%Y-%m-%d")
    symbols = {"QQQ": "QQQ", "SOXL": "SOXL", "VIX": "^VIX", "TNX": "^TNX"}
    frames: dict[str, pd.Series] = {}

    for col, sym in symbols.items():
        try:
            raw = yf.download(sym, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                st.error(f"데이터 없음: {sym}")
                return None
            s = raw["Close"]
            # yfinance 0.2+ 는 MultiIndex 반환 가능 → squeeze
            if isinstance(s, pd.DataFrame):
                s = s.squeeze()
            frames[col] = s
        except Exception as exc:
            st.error(f"다운로드 실패: {sym} — {exc}")
            return None

    df = pd.DataFrame(frames)
    # QQQ 기준으로 align, 나머지는 ffill (공휴일 등 결측)
    df = df.ffill().dropna(subset=["QQQ", "VIX"])
    # SOXL 이전 기간(2010 이전)은 QQQ 대체
    df["SOXL"] = df["SOXL"].fillna(df["QQQ"])
    return df


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════
def compute_signals(
    df: pd.DataFrame,
    vix_thresh: float   = 25.0,
    mom_days:   int     = 20,
    mom_thresh: float   = 2.0,
    tnx_days:   int     = 60,
    tnx_thresh: float   = 5.0,
) -> pd.DataFrame:
    d = df.copy()

    # TNX 60일 변화율 (%)
    d["tnx_chg"]   = d["TNX"].pct_change(tnx_days) * 100

    # QQQ N일 모멘텀 (%)
    d["qqq_mom"]   = d["QQQ"].pct_change(mom_days) * 100

    # QQQ 5일 단기 모멘텀 (Exit 보조)
    d["qqq_mom5"]  = d["QQQ"].pct_change(5) * 100

    # 개별 필터
    d["f_market"]   = d["tnx_chg"]  < tnx_thresh   # 금리 안전
    d["f_momentum"] = d["qqq_mom"]  > mom_thresh    # 추세 전환 확인
    d["f_vix"]      = d["VIX"]      > vix_thresh    # 공포 구간

    # 진입 신호: 세 조건 동시 충족
    d["entry"]      = d["f_market"] & d["f_momentum"] & d["f_vix"]

    # 청산 신호: 20일 모멘텀 음전환
    d["exit_sig"]   = d["qqq_mom"]  < 0.0

    return d


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════
def run_backtest(
    df: pd.DataFrame,
    initial:    float = 10_000_000,
    monthly:    float =    500_000,
    tx_cost:    float =      0.001,   # 왕복 거래비용 0.1%
):
    """
    Returns
    -------
    results  : 일별 포트폴리오 가치 DataFrame
    trade_log: 스위칭 이력 DataFrame
    """
    # ── Oracle 포트폴리오 ──────────────────────────
    o_qqq   = initial / df["QQQ"].iloc[0]   # QQQ 주수
    o_soxl  = 0.0                           # SOXL 주수
    in_soxl = False

    # ── QQQ DCA 벤치마크 ──────────────────────────
    b_qqq   = initial / df["QQQ"].iloc[0]

    records   = []
    trade_log = []
    prev_month = df.index[0].month

    for i, (dt, row) in enumerate(df.iterrows()):
        qqq_px  = float(row["QQQ"])
        soxl_px = float(row["SOXL"]) if pd.notna(row["SOXL"]) else qqq_px

        # ── 월 적립 (매달 첫 거래일) ─────────────────
        if i > 0 and dt.month != prev_month:
            if in_soxl:
                o_qqq  += (monthly * 0.5) / qqq_px
                o_soxl += (monthly * 0.5) / soxl_px
            else:
                o_qqq  += monthly / qqq_px
            b_qqq += monthly / qqq_px
        prev_month = dt.month

        # ── 신호 처리 ─────────────────────────────
        if i >= 1:
            entry = bool(row.get("entry",    False))
            exit_ = bool(row.get("exit_sig", False))

            if entry and not in_soxl:
                # QQQ 50% → SOXL
                switch_krw  = o_qqq * qqq_px * 0.5
                o_qqq      -= switch_krw / qqq_px
                o_soxl      = switch_krw * (1 - tx_cost) / soxl_px
                in_soxl     = True
                trade_log.append({
                    "날짜":        dt.date(),
                    "액션":        "QQQ→SOXL",
                    "VIX":         round(float(row["VIX"]), 1),
                    "QQQ 모멘텀(%)": round(float(row["qqq_mom"]), 2),
                    "TNX 변화율(%)": round(float(row["tnx_chg"]), 2),
                })

            elif exit_ and in_soxl:
                # SOXL → QQQ 전환
                soxl_krw  = o_soxl * soxl_px * (1 - tx_cost)
                o_qqq    += soxl_krw / qqq_px
                o_soxl    = 0.0
                in_soxl   = False
                trade_log.append({
                    "날짜":        dt.date(),
                    "액션":        "SOXL→QQQ",
                    "VIX":         round(float(row["VIX"]), 1),
                    "QQQ 모멘텀(%)": round(float(row["qqq_mom"]), 2),
                    "TNX 변화율(%)": round(float(row["tnx_chg"]), 2),
                })

        # ── 가치 기록 ─────────────────────────────
        records.append({
            "date":     dt,
            "oracle":   o_qqq * qqq_px + o_soxl * soxl_px,
            "dca":      b_qqq * qqq_px,
            "in_soxl":  in_soxl,
            "entry":    row.get("entry",    False),
            "exit_sig": row.get("exit_sig", False),
            "VIX":      float(row["VIX"]),
            "qqq_mom":  float(row.get("qqq_mom",  np.nan)),
            "tnx_chg":  float(row.get("tnx_chg",  np.nan)),
        })

    results   = pd.DataFrame(records).set_index("date")
    trade_df  = pd.DataFrame(trade_log)
    return results, trade_df


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def build_contributions(index: pd.DatetimeIndex,
                        initial: float, monthly: float) -> pd.Series:
    """누적 투입금 시리즈."""
    c = pd.Series(0.0, index=index)
    c.iloc[0] = initial
    prev = index[0].month
    for i, dt in enumerate(index[1:], 1):
        if dt.month != prev:
            c.iloc[i] = monthly
        prev = dt.month
    return c.cumsum()


def calc_metrics(portfolio: pd.Series, contrib: pd.Series) -> dict:
    v  = portfolio.dropna()
    fv = v.iloc[-1]
    tv = contrib.iloc[-1]                                   # 총 투입금
    yr = (v.index[-1] - v.index[0]).days / 365.25

    # CAGR (DCA이므로 근사값)
    cagr = (fv / tv) ** (1 / yr) - 1 if yr > 0 and tv > 0 else 0

    # MDD
    mdd = ((v - v.cummax()) / v.cummax()).min()

    # Sharpe (무위험 4%)
    dr = v.pct_change().dropna()
    rf = 0.04 / 252
    sh = ((dr - rf).mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0

    # 월 승률
    monthly_ret = v.resample("ME").last().pct_change().dropna()
    win_rate    = (monthly_ret > 0).mean()

    return dict(
        total_invested = tv,
        final_value    = fv,
        total_return   = (fv / tv - 1) * 100,
        cagr           = cagr * 100,
        mdd            = mdd * 100,
        sharpe         = sh,
        years          = yr,
        win_months     = win_rate * 100,
    )


# ══════════════════════════════════════════════════════════════
# NUMBER FORMATTING
# ══════════════════════════════════════════════════════════════
def fmt(v: float) -> str:
    """원화 단위 포맷 — 억/만 단위 자동 변환 + 3자리 콤마"""
    if v >= 100_000_000:
        eok = v / 100_000_000
        return f"{eok:,.1f}억원"
    elif v >= 10_000:
        man = v / 10_000
        return f"{man:,.0f}만원"
    return f"{v:,.0f}원"

def fmt_full(v: float) -> str:
    """전체 콤마 표기 (예: 92,971만원)"""
    return f"{v:,.0f}원"

def pct(v: float) -> str:
    return f"{v:+.1f}%" if v != 0 else "0.0%"


# ══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════
DARK = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(8,8,15,0.95)",
    font          = dict(color="#6b7280", size=12),
    margin        = dict(l=60, r=20, t=30, b=40),
    hovermode     = "x unified",
)
DARK_AXIS = dict(gridcolor="#151520", showgrid=True, zeroline=False)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 파라미터 조정")

    st.markdown("#### 💰 투자 설정")
    initial_cap = st.number_input("초기 투자금 (원)", 1_000_000, 200_000_000,
                                  10_000_000, 1_000_000, format="%d")
    st.caption(f"→ {initial_cap:,}원")
    monthly_inv = st.number_input("월 적립금 (원)", 100_000, 20_000_000,
                                    500_000, 100_000, format="%d")
    st.caption(f"→ {monthly_inv:,}원")

    st.markdown("#### 📡 신호 임계값")
    vix_t   = st.slider("VIX 임계값",        15, 45, 25, 1)
    mom_t   = st.slider("모멘텀 기준 (%)",    0.0, 10.0, 2.0, 0.5)
    tnx_t   = st.slider("TNX 60일 한도 (%)  ", 1.0, 20.0, 5.0, 0.5)

    st.markdown("---")
    st.caption(
        "📌 **개인 교육용 도구입니다.**\n\n"
        "과거 수익률은 미래를 보장하지 않으며, "
        "레버리지 ETF는 변동성 잠식(Volatility Decay)으로 "
        "장기 보유 시 예상보다 수익이 낮을 수 있습니다."
    )


# ══════════════════════════════════════════════════════════════
# DATA LOAD  &  COMPUTE
# ══════════════════════════════════════════════════════════════
st.markdown("# 🔮 나스닥 하이브리드 오라클 전략")
st.markdown("QQQ ↔ SOXL 스위칭 전략 | 10년 백테스트 검증 대시보드 · 개인 교육용")

with st.spinner("📡 데이터 로딩 중 (약 10~20초)..."):
    raw = load_data("2014-01-01")

if raw is None:
    st.stop()

sig_df      = compute_signals(raw, vix_t, 20, mom_t, 60, tnx_t)
bt, trades  = run_backtest(sig_df, initial_cap, monthly_inv)
contrib     = build_contributions(bt.index, initial_cap, monthly_inv)
om          = calc_metrics(bt["oracle"], contrib)
dm          = calc_metrics(bt["dca"],    contrib)

# QQQ 가격 CAGR — DCA와 무관하게 "지수가 얼마나 올랐나"
# 사람들이 "QQQ 연 18%" 할 때 쓰는 그 숫자
_yr            = (sig_df.index[-1] - sig_df.index[0]).days / 365.25
qqq_price_cagr = ((sig_df["QQQ"].iloc[-1] / sig_df["QQQ"].iloc[0]) ** (1 / _yr) - 1) * 100


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 백테스트 결과", "🎯 현재 시장 상태", "🚀 시뮬레이터"])


# ──────────────────────────────────────────────────────────────
# TAB 1 : 백테스트 결과
# ──────────────────────────────────────────────────────────────
with tab1:

    gap    = om["cagr"] - dm["cagr"]
    years  = om["years"]
    tv     = om["total_invested"]

    # ── 전략 한 줄 설명 ────────────────────────────────────
    with st.expander("📖 이 전략이 뭔가요? (클릭해서 보기)", expanded=False):
        st.markdown(f"""
**오라클 전략**은 평소엔 QQQ를 적립하다가, 시장이 공황 상태일 때 반등 순간을 잡아 **SOXL(반도체 3배 레버리지)로 50% 스위칭**하는 전략입니다.

#### 언제 SOXL로 바꾸나요? (3가지 조건 동시 충족)
| 조건 | 기준 | 의미 |
|---|---|---|
| 🔴 VIX 공포지수 | {vix_t} 이상 | 시장이 충분히 겁먹은 상태 |
| 🟢 QQQ 20일 수익률 | {mom_t}% 이상 | 바닥 치고 회복 시작 신호 |
| 🟡 금리 안정 | 60일 변화율 {tnx_t}% 미만 | 금리 충격 없어 유동성 OK |

#### 언제 다시 QQQ로 돌아오나요?
- QQQ 20일 수익률이 **0% 아래**로 떨어지면 → 추세가 꺾인 것으로 판단, 즉시 복귀

#### ⚠️ 주의할 점
- SOXL은 **3배 레버리지**라 반등 시 수익은 3배지만, 하락 시 손실도 3배
- 장기 보유하면 **변동성 잠식(Volatility Decay)** 으로 기대보다 수익이 낮음
- 단기 스위칭 용도로만 설계된 전략
        """)

    # ── 검증 결과 배너 ─────────────────────────────────────
    if om["cagr"] >= 22 * 0.90:
        st.success(f"✅ 주장된 22% CAGR → 백테스트 실측 **{om['cagr']:.1f}%** 로 재현됩니다.")
    elif om["cagr"] > dm["cagr"]:
        st.warning(f"⚠️ 실측 CAGR **{om['cagr']:.1f}%** — 22% 주장보다 낮지만 QQQ 적립({dm['cagr']:.1f}%)은 초과")
    else:
        st.error(f"❌ 현재 파라미터 CAGR **{om['cagr']:.1f}%** < QQQ 적립 {dm['cagr']:.1f}% — 슬라이더 조정 필요")

    st.markdown("---")

    # ── 핵심 카드: 쉽게 말하면 ─────────────────────────────
    st.markdown("### 💡 쉽게 말하면 이렇습니다")
    st.markdown(
        f"**{int(years)}년간 {fmt(initial_cap)} 넣고, 매달 {fmt(monthly_inv)}씩 적립했을 때**  "
        f"(총 투입금 **{fmt(tv)}**)"
    )

    col_o, col_d, col_diff = st.columns(3)
    with col_o:
        st.markdown(f"""
        <div style="background:#0f1f16;border:1px solid #22c55e44;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">🔮 오라클 전략</div>
          <div style="font-size:28px;font-weight:700;color:#22c55e">{fmt(om['final_value'])}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">내 돈의 {om['final_value']/tv:.1f}배</div>
          <div style="font-size:12px;color:#4ade80;margin-top:2px">연평균 {om['cagr']:.1f}% 복리</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #374151;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">📊 QQQ 그냥 적립</div>
          <div style="font-size:28px;font-weight:700;color:#9ca3af">{fmt(dm['final_value'])}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">내 돈의 {dm['final_value']/tv:.1f}배</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px">연평균 {dm['cagr']:.1f}% 복리</div>
        </div>""", unsafe_allow_html=True)
    with col_diff:
        extra = om['final_value'] - dm['final_value']
        sign_color = "#22c55e" if extra >= 0 else "#f87171"
        sign_txt   = "더 벌었습니다" if extra >= 0 else "덜 벌었습니다"
        st.markdown(f"""
        <div style="background:#0f0f1a;border:1px solid #6366f144;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">📐 전략 초과 수익</div>
          <div style="font-size:28px;font-weight:700;color:{sign_color}">{fmt(abs(extra))}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">{sign_txt}</div>
          <div style="font-size:12px;color:#6366f1;margin-top:2px">스위칭 {len(trades)}회 (비용 0.1%)</div>
        </div>""", unsafe_allow_html=True)

    # ── 연평균 수익률 시각적 비교 ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 연평균 수익률(CAGR)이 뭔지 모르겠어요", expanded=False):
        st.markdown("""
**CAGR = 내 돈이 매년 평균 몇 %씩 불었나**

예시: 1,000만원을 10년 뒤 2,593만원으로 불렸다면 → CAGR **10%**
(1,000만원 × 1.10 × 1.10 × ... × 1.10 = 2,593만원)

> 은행 예금이 연 3.5%라면, CAGR 13%는 예금보다 **약 3.7배** 더 빠르게 돈이 불어난다는 뜻입니다.

---
**왜 "QQQ 주가가 연 17~18% 올랐다"는데 내 DCA 포트폴리오 수익률은 그보다 낮게 나오나요?**

→ 매달 조금씩 넣으면, **마지막에 넣은 돈은 1달밖에 못 굴립니다.**
10년 전에 넣은 돈은 17%를 10번 받지만, 1년 전 돈은 17%를 1번만 받습니다.
평균으로 따지면 7~8번 받은 셈이 되어, DCA CAGR은 가격 CAGR보다 낮게 나옵니다.
이건 계산이 잘못된 게 아니라 **DCA의 정상적인 특성**입니다.
        """)

    # ── 수치 비교 바 ───────────────────────────────────────
    metrics_data = [
        ("📈 QQQ 주가 상승률", qqq_price_cagr, "#818cf8",
         "주가 자체가 오른 비율 (적립과 무관)"),
        ("🔮 오라클 연평균", om["cagr"], "#22c55e",
         f"적립하면서 전략 적용 시 | MDD(최대낙폭): {om['mdd']:.1f}%"),
        ("📊 QQQ 그냥 적립", dm["cagr"], "#6b7280",
         f"아무 것도 안 하고 매달 QQQ 삼 | MDD(최대낙폭): {dm['mdd']:.1f}%"),
        ("🏦 정기예금 (참고)", 3.5, "#374151",
         "위험 없는 기준선"),
    ]
    max_val = max(m[1] for m in metrics_data) * 1.15
    for label, val, color, note in metrics_data:
        bar_pct = val / max_val * 100
        st.markdown(f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
            <span style="color:#c0c0d8;font-size:14px">{label}</span>
            <span style="color:{color};font-weight:700;font-size:18px;font-family:monospace">{val:.1f}%</span>
          </div>
          <div style="background:#151520;border-radius:4px;height:10px;margin-bottom:4px">
            <div style="background:{color};width:{bar_pct:.0f}%;height:100%;border-radius:4px"></div>
          </div>
          <div style="font-size:11px;color:#3e3e56">{note}</div>
        </div>""", unsafe_allow_html=True)

    # ── MDD 설명 ───────────────────────────────────────────
    oracle_worst  = om['final_value'] * (1 + om['mdd'] / 100)
    dca_worst     = dm['final_value'] * (1 + dm['mdd'] / 100)
    st.markdown(f"""
    <div style="background:#1a1010;border:0.5px solid #f8717133;border-radius:10px;padding:1rem 1.2rem;margin:8px 0">
      <div style="color:#f87171;font-size:13px;font-weight:600;margin-bottom:6px">
        📉 MDD(최대 낙폭) — 투자 중 가장 많이 떨어진 순간
      </div>
      <div style="color:#9ca3af;font-size:13px">
        오라클 MDD <b style="color:#f87171">{om['mdd']:.1f}%</b>
        &nbsp;·&nbsp;
        QQQ 적립 MDD <b style="color:#fbbf24">{dm['mdd']:.1f}%</b>
      </div>
      <div style="color:#6b7280;font-size:12px;margin-top:4px">
        즉, 오라클 전략은 최악의 순간 자산이 전고점 대비 {abs(om['mdd']):.0f}%까지 빠졌습니다.
        1,000만원이 {1000*(1+om['mdd']/100):.0f}만원이 됐던 순간이 있었다는 뜻입니다.
        이 구간을 버텨낼 심리적 준비가 필요합니다.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 더 나은 전략 추천 ──────────────────────────────────
    with st.expander("💡 이 전략보다 좋은 전략이 있나요?", expanded=False):
        st.markdown("""
백테스트 결과와 학술 연구를 기반으로 **더 단순하고 검증된 전략**들을 소개합니다.

---
#### 1. 📏 QQQ 200일선 필터 전략 (추천 ⭐⭐⭐⭐)
> 규칙: QQQ가 **200일 이동평균선 위** → QQQ 보유 / 아래 → 현금(or 단기채)

- **장점**: 2008년, 2022년 같은 큰 하락을 상당 부분 피함. MDD 크게 감소
- **단점**: 자주 신호가 바뀌면 세금·수수료 발생, 횡보장에서 손실 가능
- **연구**: 200MA 필터는 수십 년 데이터에서 일관성 있게 MDD를 줄이는 것으로 검증됨

---
#### 2. 🔄 듀얼 모멘텀 (Gary Antonacci, 추천 ⭐⭐⭐⭐)
> 규칙: 매달 QQQ vs 채권(AGG) 최근 12개월 수익률 비교 → 높은 쪽에 100% 투자

- **장점**: 단순하고, 학술 논문으로 검증된 전략. 주식 약세장에 자동 회피
- **단점**: 월 1회 리밸런싱 필요. 추세 전환 시점 약간 늦음
- **책**: "Dual Momentum Investing" (2014) — 검증된 퀀트 전략

---
#### 3. 🌦️ 올웨더 포트폴리오 (Ray Dalio, 추천 ⭐⭐⭐)
> 배분: 주식 30% / 장기채 40% / 중기채 15% / 금 7.5% / 원자재 7.5%

- **장점**: 어떤 경제 환경에서도 큰 손실 없이 안정적. MDD 매우 낮음
- **단점**: 장기 수익률은 주식 100%보다 낮음. 레버리지 없음
- **특징**: 큰 하락보다 안정성이 중요한 사람에게 적합

---
#### 🔮 오라클 전략의 실제 강점
오라클 전략은 위 전략들보다 **복잡하지만**, QQQ DCA를 초과하는 것으로 백테스트에서 확인됩니다.
다만 신호가 드물게 발생하고 SOXL의 변동성이 크다는 점에서 **심리적 관리가 어렵습니다**.

| 전략 | 기대 수익 | MDD | 복잡도 |
|---|---|---|---|
| 오라클 (현 전략) | 높음 | 높음 | 복잡 |
| 200MA 필터 | 중간 | 낮음 | 단순 |
| 듀얼 모멘텀 | 중간 | 낮음 | 단순 |
| 올웨더 | 낮음 | 매우 낮음 | 단순 |

**결론: 수익 극대화보단 수익/위험 균형이 중요하다면 듀얼 모멘텀이나 200MA 필터가 더 현실적입니다.**
        """)

    st.markdown("---")

    # ── 누적 자산 차트 ────────────────────────────────────
    st.markdown("### 📈 누적 자산 변화 (2014 → 현재)")

    fig = go.Figure()

    # SOXL 보유 구간 음영
    soxl_flag  = bt["in_soxl"].astype(int)
    transitions = soxl_flag.diff().fillna(0)
    s_starts   = bt.index[transitions == 1].tolist()
    s_ends     = bt.index[transitions == -1].tolist()
    if len(s_starts) > len(s_ends):
        s_ends.append(bt.index[-1])
    for s, e in zip(s_starts, s_ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="#6366f1",
                      opacity=0.07, line_width=0,
                      annotation_text="SOXL" if (e - s).days > 5 else "",
                      annotation_font_color="#6366f1",
                      annotation_font_size=9)

    # 투입 원금
    fig.add_trace(go.Scatter(
        x=bt.index, y=contrib,
        name="누적 투입금", mode="lines",
        line=dict(color="#2d2d4a", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(45,45,74,0.15)",
    ))
    # QQQ DCA
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["dca"],
        name="QQQ DCA", mode="lines",
        line=dict(color="#6b7280", width=2, dash="dash"),
    ))
    # Oracle
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["oracle"],
        name="오라클 전략", mode="lines",
        line=dict(color="#818cf8", width=2.5),
    ))
    # Entry markers
    entries = bt[bt["entry"] == True]
    fig.add_trace(go.Scatter(
        x=entries.index, y=entries["oracle"], mode="markers",
        name="SOXL 진입", marker=dict(
            color="#22c55e", size=9, symbol="triangle-up",
            line=dict(color="white", width=1),
        ),
    ))
    # Exit markers
    exits = bt[bt["exit_sig"] == True]
    fig.add_trace(go.Scatter(
        x=exits.index, y=exits["oracle"], mode="markers",
        name="QQQ 복귀", marker=dict(
            color="#f87171", size=7, symbol="triangle-down",
            line=dict(color="white", width=1),
        ),
    ))

    fig.update_layout(
        **DARK,
        height=460,
        yaxis=dict(gridcolor="#151520", showgrid=True, zeroline=False, title="포트폴리오 (원)", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(10,10,20,0.85)",
                    bordercolor="#2a2a3a", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MDD 차트 ──────────────────────────────────────────
    with st.expander("📉 낙폭(Drawdown) 비교 차트"):
        dd_oracle = (bt["oracle"] - bt["oracle"].cummax()) / bt["oracle"].cummax() * 100
        dd_dca    = (bt["dca"]    - bt["dca"].cummax())    / bt["dca"].cummax()    * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=bt.index, y=dd_dca,
            name="QQQ DCA", line=dict(color="#6b7280", width=1.5),
            fill="tozeroy", fillcolor="rgba(107,114,128,0.1)"))
        fig_dd.add_trace(go.Scatter(x=bt.index, y=dd_oracle,
            name="오라클", line=dict(color="#818cf8", width=2),
            fill="tozeroy", fillcolor="rgba(129,140,248,0.1)"))
        fig_dd.update_layout(**DARK, height=280,
            xaxis=dict(**DARK_AXIS),
            yaxis=dict(**DARK_AXIS, title="낙폭 (%)"))
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── 스위칭 이력 ───────────────────────────────────────
    st.markdown("### 🔔 스위칭 이력")
    if trades.empty:
        st.info("현재 파라미터에서 신호가 발생하지 않았습니다. 슬라이더를 조정해 보세요.")
    else:
        ca, cb = st.columns([3, 1])
        with ca:
            def color_action(val):
                if val == "QQQ→SOXL":
                    return "color: #22c55e; font-weight:600"
                elif val == "SOXL→QQQ":
                    return "color: #f87171; font-weight:600"
                return ""
            styled = trades.style.map(color_action, subset=["액션"])
            st.dataframe(styled, use_container_width=True, height=300)

        with cb:
            n_entry = (trades["액션"] == "QQQ→SOXL").sum()
            n_exit  = (trades["액션"] == "SOXL→QQQ").sum()
            avg_vix = trades.loc[trades["액션"]=="QQQ→SOXL", "VIX"].mean()
            st.metric("SOXL 진입 횟수", f"{n_entry}회")
            st.metric("QQQ 복귀 횟수",  f"{n_exit}회")
            if not np.isnan(avg_vix):
                st.metric("진입 시 평균 VIX", f"{avg_vix:.1f}")

    # ── 면책 고지 ─────────────────────────────────────────
    st.info(
        "**📌 백테스트 주의사항**  \n"
        "① 과거 수익률은 미래를 보장하지 않습니다  \n"
        "② 파라미터를 과거 데이터에 맞게 최적화하면 Overfitting이 발생합니다  \n"
        "③ 세금(양도소득세), 환전비용, 슬리피지는 미반영  \n"
        "④ SOXL은 3× 레버리지 ETF — 장기 보유 시 변동성 잠식 주의"
    )


# ──────────────────────────────────────────────────────────────
# TAB 2 : 현재 시장 상태
# ──────────────────────────────────────────────────────────────
with tab2:
    latest    = sig_df.iloc[-1]
    last_date = sig_df.index[-1]

    cur_vix  = float(latest["VIX"])
    cur_mom  = float(latest.get("qqq_mom", np.nan))
    cur_tnx  = float(latest.get("tnx_chg", np.nan))

    f_mkt = bool(latest.get("f_market",   False))
    f_mom = bool(latest.get("f_momentum", False))
    f_vx  = bool(latest.get("f_vix",      False))
    n_ok  = sum([f_mkt, f_mom, f_vx])

    # 상태 결정
    if n_ok == 3:
        state, col_s = "🚀 가속 (SOXL 스위칭 시점)", "#22c55e"
        advice = "모든 조건 충족 — 오라클 전략 기준 SOXL 50% 스위칭 시점입니다."
    elif n_ok >= 1:
        state, col_s = "⏳ 적립 대기", "#fbbf24"
        advice = f"{3 - n_ok}개 조건 미충족 — QQQ DCA 유지하며 신호를 기다립니다."
    else:
        state, col_s = "🛡️ 안전 모드 (DCA)", "#6366f1"
        advice = "신호 없음 — 일반 QQQ 적립을 유지하세요."

    col_g, col_f = st.columns([1, 1])

    # ── 게이지 ────────────────────────────────────────────
    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = n_ok,
            title = {"text": f"<b>{state}</b>",
                     "font": {"color": col_s, "size": 15}},
            number= {"suffix": " / 3 조건", "font": {"color": "#ddddf0", "size": 28}},
            gauge = {
                "axis":      {"range": [0, 3], "tickcolor": "#6b7280"},
                "bar":       {"color": col_s, "thickness": 0.28},
                "bgcolor":   "#10101e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 1], "color": "#141426"},
                    {"range": [1, 2], "color": "#141426"},
                    {"range": [2, 3], "color": "#141426"},
                ],
                "threshold": {
                    "line":      {"color": col_s, "width": 4},
                    "thickness": 0.8,
                    "value":     n_ok,
                },
            },
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9ca3af"),
                            height=260,
                            margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption(f"기준: {last_date.strftime('%Y년 %m월 %d일')}")

    # ── 필터 카드 ──────────────────────────────────────────
    with col_f:
        def fcard(label: str, val_str: str, ok: bool, note: str) -> str:
            icon  = "✅" if ok else "❌"
            bc    = "#22c55e33" if ok else "#f8717122"
            vc    = "#22c55e"  if ok else "#f87171"
            return f"""
            <div style="background:#10101e;border:0.5px solid {bc};
                        border-radius:10px;padding:12px 16px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:#ddddf0;font-size:14px;font-weight:500">{icon} {label}</span>
                <span style="color:{vc};font-family:monospace;font-size:18px;font-weight:600">{val_str}</span>
              </div>
              <div style="font-size:11px;color:#5a5a7a;margin-top:3px">{note}</div>
            </div>"""

        vix_str = f"{cur_vix:.1f}" if not np.isnan(cur_vix) else "N/A"
        mom_str = f"{cur_mom:+.1f}%" if not np.isnan(cur_mom) else "N/A"
        tnx_str = f"{cur_tnx:+.1f}%" if not np.isnan(cur_tnx) else "N/A"

        st.markdown(
            fcard("VIX 공포지수",     vix_str, f_vx,
                  f"임계값 {vix_t} 이상이면 공포 구간 → {'✔ 충족' if f_vx else '✘ 미충족'}") +
            fcard("QQQ 20일 모멘텀",  mom_str, f_mom,
                  f"20일 수익률 {mom_t}% 이상이면 추세 전환 → {'✔ 충족' if f_mom else '✘ 미충족'}") +
            fcard("TNX 60일 변화율", tnx_str,  f_mkt,
                  f"금리 변화율 {tnx_t}% 미만이면 유동성 안전 → {'✔ 충족' if f_mkt else '✘ 미충족'}"),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#10101e;border:0.5px solid {col_s}44;'
            f'border-radius:10px;padding:12px 16px;color:{col_s};font-size:14px;">'
            f"💬 {advice}</div>",
            unsafe_allow_html=True,
        )

    # ── 최근 90일 지표 차트 ───────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 최근 90일 지표 추이")
    recent = sig_df.iloc[-90:].copy()

    fig_r = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("VIX 공포지수", "QQQ 20일 모멘텀 (%)", "TNX 60일 변화율 (%)"),
        vertical_spacing=0.1, row_heights=[0.4, 0.3, 0.3],
    )
    # VIX
    fig_r.add_trace(go.Scatter(x=recent.index, y=recent["VIX"],
        line=dict(color="#f87171", width=2), name="VIX"), row=1, col=1)
    fig_r.add_hline(y=vix_t, line=dict(color="#22c55e", width=1.2, dash="dash"),
                    row=1, col=1, annotation_text=f"임계 {vix_t}",
                    annotation_font_color="#22c55e")
    # Momentum bars
    colors_m = ["#22c55e" if v > 0 else "#f87171"
                for v in recent["qqq_mom"].fillna(0)]
    fig_r.add_trace(go.Bar(x=recent.index, y=recent["qqq_mom"],
        marker_color=colors_m, name="모멘텀"), row=2, col=1)
    fig_r.add_hline(y=mom_t, line=dict(color="#6366f1", width=1.2, dash="dash"),
                    row=2, col=1, annotation_text=f"임계 {mom_t}%",
                    annotation_font_color="#6366f1")
    # TNX bars
    colors_t = ["#f87171" if v > tnx_t else "#22c55e"
                for v in recent["tnx_chg"].fillna(0)]
    fig_r.add_trace(go.Bar(x=recent.index, y=recent["tnx_chg"],
        marker_color=colors_t, name="TNX 변화율"), row=3, col=1)
    fig_r.add_hline(y=tnx_t, line=dict(color="#f87171", width=1.2, dash="dash"),
                    row=3, col=1, annotation_text=f"한도 {tnx_t}%",
                    annotation_font_color="#f87171")

    fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(8,8,15,0.95)",
                        font=dict(color="#6b7280"),
                        showlegend=False, height=520,
                        margin=dict(l=55, r=20, t=40, b=40))
    fig_r.update_xaxes(gridcolor="#151520")
    fig_r.update_yaxes(gridcolor="#151520")
    st.plotly_chart(fig_r, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 : 시뮬레이터
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🚀 목표 자산 시뮬레이터")
    st.markdown("목표일과 적립금을 입력하면 각 전략별 예상 자산을 계산합니다.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_init    = st.number_input("현재 보유 자산 (원)", 0, 500_000_000,
                                     0, 100_000, format="%d")
    with sc2:
        s_monthly = st.number_input("월 적립 가능액 (원)", 0, 10_000_000,
                                     300_000, 50_000, format="%d")
    with sc3:
        s_target  = st.date_input("목표일",
                                   value=date(2027, 6, 1),
                                   min_value=date.today())

    months  = max(1, (s_target.year - date.today().year) * 12
                   + (s_target.month - date.today().month))

    def fv(p0: float, pmt: float, n: int, ann_r: float) -> float:
        r = (1 + ann_r) ** (1/12) - 1
        return p0 * (1+r)**n + (pmt * (((1+r)**n - 1) / r) if r > 0.0001 else pmt * n)

    r_oracle = om["cagr"] / 100
    r_dca    = dm["cagr"] / 100
    r_safe   = 0.04

    v_oracle = fv(s_init, s_monthly, months, r_oracle)
    v_dca    = fv(s_init, s_monthly, months, r_dca)
    v_safe   = fv(s_init, s_monthly, months, r_safe)
    v_total  = s_init + s_monthly * months

    # ── 예상 자산 결과 카드 ───────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {s_target.strftime('%Y년 %m월 %d일')} 기준 예상 자산 ({months}개월 후)")
    st.caption(f"총 투입 예정금: {fmt_full(v_total)} (현재 {fmt_full(s_init)} + 월 {fmt_full(s_monthly)} × {months}개월)")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🛡️ 안전 투자 (연 4%)",
               fmt_full(v_safe),
               f"투입금 대비 +{(v_safe/v_total-1)*100:.0f}%" if v_total > 0 else "—")
    mc2.metric("📊 QQQ 그냥 적립",
               fmt_full(v_dca),
               f"투입금 대비 +{(v_dca/v_total-1)*100:.0f}%" if v_total > 0 else "—")
    mc3.metric(f"🔮 오라클 전략 (연 {r_oracle*100:.1f}%)",
               fmt_full(v_oracle),
               f"투입금 대비 +{(v_oracle/v_total-1)*100:.0f}%" if v_total > 0 else "—")

    # ── 성장 시뮬레이션 차트 ──────────────────────────────
    sim_months = list(range(0, months + 1))
    sim_oracle = [fv(s_init, s_monthly, m, r_oracle) for m in sim_months]
    sim_dca    = [fv(s_init, s_monthly, m, r_dca)    for m in sim_months]
    sim_safe   = [fv(s_init, s_monthly, m, r_safe)   for m in sim_months]
    sim_input  = [s_init + s_monthly * m              for m in sim_months]

    from datetime import timedelta
    sim_dates = [date.today().replace(day=1)]
    for _ in range(months):
        d = sim_dates[-1]
        m = d.month + 1 if d.month < 12 else 1
        y = d.year if d.month < 12 else d.year + 1
        sim_dates.append(d.replace(year=y, month=m))

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=sim_dates, y=sim_input,
        name="투입 원금", line=dict(color="#2d2d4a", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(45,45,74,0.15)"))
    fig_sim.add_trace(go.Scatter(x=sim_dates, y=sim_safe,
        name="안전 투자 (4%)", line=dict(color="#374151", width=1.5)))
    fig_sim.add_trace(go.Scatter(x=sim_dates, y=sim_dca,
        name="QQQ 적립", line=dict(color="#6b7280", width=2, dash="dash")))
    fig_sim.add_trace(go.Scatter(x=sim_dates, y=sim_oracle,
        name="오라클 전략", line=dict(color="#22c55e", width=2.5)))
    fig_sim.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,8,15,0.95)",
        font=dict(color="#6b7280"), height=320,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#151520"),
        yaxis=dict(gridcolor="#151520", tickformat=",.0f", title="예상 자산 (원)"),
        legend=dict(bgcolor="rgba(10,10,20,0.85)", bordercolor="#2a2a3a", borderwidth=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # ── 내 목표 설정 달성도 ───────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 내 목표 달성도")
    st.caption("달성하고 싶은 목표를 직접 입력하세요. 각 전략별로 얼마나 달성하는지 보여줍니다.")

    # 목표 개수 (session state)
    if "n_goals" not in st.session_state:
        st.session_state.n_goals = 2

    ga, gb = st.columns([5, 1])
    with gb:
        if st.button("+ 목표 추가"):
            st.session_state.n_goals = min(st.session_state.n_goals + 1, 6)

    goals = []
    for i in range(st.session_state.n_goals):
        g1, g2 = st.columns([2, 2])
        with g1:
            g_name = st.text_input(f"목표 이름 {i+1}", value=["첫 번째 목표", "두 번째 목표",
                "세 번째 목표", "네 번째 목표", "다섯 번째 목표", "여섯 번째 목표"][i],
                key=f"gname_{i}", label_visibility="collapsed",
                placeholder=f"목표 {i+1} 이름")
        with g2:
            g_amt = st.number_input(f"목표 금액 {i+1}", value=[30_000_000, 100_000_000,
                50_000_000, 200_000_000, 500_000_000, 1_000_000_000][i],
                step=1_000_000, format="%d", key=f"gamt_{i}",
                label_visibility="collapsed")
        goals.append((g_name, g_amt))

    st.markdown("<br>", unsafe_allow_html=True)
    for g_name, g_amt in goals:
        if g_amt <= 0:
            continue
        strategies = [
            ("🛡️ 안전(4%)", v_safe,   "#4b5563"),
            ("📊 QQQ적립",  v_dca,    "#6b7280"),
            ("🔮 오라클",   v_oracle, "#22c55e"),
        ]
        st.markdown(f"<div style='color:#c0c0d8;font-size:14px;font-weight:600;"
                    f"margin-bottom:8px'>🎯 {g_name} — 목표 {fmt_full(g_amt)}</div>",
                    unsafe_allow_html=True)
        for s_name, s_val, s_color in strategies:
            ratio  = s_val / g_amt
            bar_w  = min(ratio * 100, 100)
            color  = "#22c55e" if ratio >= 1 else "#fbbf24" if ratio >= 0.5 else s_color
            label  = "✅ 달성!" if ratio >= 1 else f"{ratio*100:.0f}%"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
              <span style="color:#6b7280;font-size:12px;min-width:72px">{s_name}</span>
              <div style="flex:1;background:#151520;border-radius:3px;height:7px">
                <div style="background:{color};width:{bar_w:.0f}%;height:100%;border-radius:3px"></div>
              </div>
              <span style="color:{color};font-size:12px;font-weight:600;min-width:88px;text-align:right">
                {fmt(s_val)} ({label})
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.caption(
        f"⚠️ 오라클 CAGR {r_oracle*100:.1f}%가 미래에도 동일하다는 낙관적 가정입니다. "
        f"실제 수익률은 크게 다를 수 있습니다."
    )


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#2d2d4a;font-size:11px;padding:.5rem 0">'
    "개인 교육용 백테스팅 도구 | 투자 권유 아님 | "
    "과거 수익률은 미래를 보장하지 않습니다 | "
    "Built with Streamlit · yfinance · Plotly"
    "</div>",
    unsafe_allow_html=True,
)
