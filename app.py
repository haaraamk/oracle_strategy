"""
나스닥 하이브리드 오라클 전략 — 백테스트 & 검증 대시보드
개인 교육용 | 투자 권유 아님
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
import itertools

st.set_page_config(page_title="오라클 전략 검증기", page_icon="🔮",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background:#08080f; }
[data-testid="stSidebar"]           { background:#0e0e1a; border-right:1px solid #1e1e30; }
[data-testid="stSidebar"] label     { color:#8888aa !important; }
h1,h2,h3                            { color:#ddddf0 !important; letter-spacing:-0.02em; }
div[data-testid="stMetric"]         { background:#10101e; border:0.5px solid #22223a; border-radius:10px; padding:.9rem 1rem !important; }
div[data-testid="stMetricValue"]    { color:#ddddf0 !important; }
.stTabs [data-baseweb="tab"]        { color:#5a5a7a; background:transparent; font-size:15px; }
.stTabs [aria-selected="true"]      { color:#a0a8ff !important; border-bottom:2px solid #6366f1 !important; }
.stTabs [data-baseweb="tab-list"]   { background:transparent; border-bottom:1px solid #1e1e30; gap:8px; }
</style>
""", unsafe_allow_html=True)


# ── DATA ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(start="2014-01-01"):
    end, frames = datetime.today().strftime("%Y-%m-%d"), {}
    for col, sym in {"QQQ":"QQQ","SOXL":"SOXL","VIX":"^VIX","TNX":"^TNX"}.items():
        try:
            raw = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                st.error(f"데이터 없음: {sym}"); return None
            s = raw["Close"]
            frames[col] = s.squeeze() if isinstance(s, pd.DataFrame) else s
        except Exception as e:
            st.error(f"다운로드 실패: {sym} — {e}"); return None
    df = pd.DataFrame(frames).ffill().dropna(subset=["QQQ","VIX"])
    df["SOXL"] = df["SOXL"].fillna(df["QQQ"])
    return df


# ── SIGNALS ──────────────────────────────────────────────────
def compute_signals(df, vix_thresh=25.0, mom_days=20,
                    mom_thresh=2.0, tnx_days=60, tnx_thresh=5.0):
    d = df.copy()
    d["tnx_chg"]    = d["TNX"].pct_change(tnx_days) * 100
    d["qqq_mom"]    = d["QQQ"].pct_change(mom_days)  * 100
    d["f_market"]   = d["tnx_chg"]  < tnx_thresh
    d["f_momentum"] = d["qqq_mom"]  > mom_thresh
    d["f_vix"]      = d["VIX"]      > vix_thresh
    d["entry"]      = d["f_market"] & d["f_momentum"] & d["f_vix"]
    d["exit_sig"]   = d["qqq_mom"]  < 0.0
    return d


# ── BACKTEST ─────────────────────────────────────────────────
def run_backtest(df, initial=10_000_000, monthly=500_000, tx_cost=0.001):
    o_qqq, o_soxl, in_soxl = initial/df["QQQ"].iloc[0], 0.0, False
    b_qqq      = initial/df["QQQ"].iloc[0]
    records, trade_log, prev_month = [], [], df.index[0].month

    for i, (dt, row) in enumerate(df.iterrows()):
        qp = float(row["QQQ"])
        sp = float(row["SOXL"]) if pd.notna(row["SOXL"]) else qp
        if i > 0 and dt.month != prev_month:
            if in_soxl:
                o_qqq += (monthly*.5)/qp; o_soxl += (monthly*.5)/sp
            else:
                o_qqq += monthly/qp
            b_qqq += monthly/qp
        prev_month = dt.month
        if i >= 1:
            if bool(row.get("entry",False)) and not in_soxl:
                sw=o_qqq*qp*.5; o_qqq-=sw/qp; o_soxl=sw*(1-tx_cost)/sp; in_soxl=True
                trade_log.append({"날짜":dt.date(),"액션":"QQQ→SOXL",
                    "VIX":round(float(row["VIX"]),1),
                    "QQQ 모멘텀(%)":round(float(row["qqq_mom"]),2),
                    "TNX 변화율(%)":round(float(row["tnx_chg"]),2)})
            elif bool(row.get("exit_sig",False)) and in_soxl:
                sv=o_soxl*sp*(1-tx_cost); o_qqq+=sv/qp; o_soxl=0.0; in_soxl=False
                trade_log.append({"날짜":dt.date(),"액션":"SOXL→QQQ",
                    "VIX":round(float(row["VIX"]),1),
                    "QQQ 모멘텀(%)":round(float(row["qqq_mom"]),2),
                    "TNX 변화율(%)":round(float(row["tnx_chg"]),2)})
        records.append({"date":dt,"oracle":o_qqq*qp+o_soxl*sp,"dca":b_qqq*qp,
                        "in_soxl":in_soxl,"entry":row.get("entry",False),
                        "exit_sig":row.get("exit_sig",False),
                        "VIX":float(row["VIX"]),
                        "qqq_mom":float(row.get("qqq_mom",np.nan)),
                        "tnx_chg":float(row.get("tnx_chg",np.nan))})
    return pd.DataFrame(records).set_index("date"), pd.DataFrame(trade_log)


# ── METRICS ──────────────────────────────────────────────────
def build_contributions(index, initial, monthly):
    c=pd.Series(0.0,index=index); c.iloc[0]=initial; prev=index[0].month
    for i,dt in enumerate(index[1:],1):
        if dt.month!=prev: c.iloc[i]=monthly
        prev=dt.month
    return c.cumsum()

def calc_metrics(portfolio, contrib):
    v=portfolio.dropna(); fv=v.iloc[-1]; tv=contrib.iloc[-1]
    yr=(v.index[-1]-v.index[0]).days/365.25
    cagr=(fv/tv)**(1/yr)-1 if (yr>0 and tv>0) else 0
    mdd=((v-v.cummax())/v.cummax()).min()
    dr=v.pct_change().dropna(); rf=0.04/252
    sh=((dr-rf).mean()/dr.std())*np.sqrt(252) if dr.std()>0 else 0
    wr=(v.resample("ME").last().pct_change().dropna()>0).mean()
    return dict(total_invested=tv,final_value=fv,total_return=(fv/tv-1)*100,
                cagr=cagr*100,mdd=mdd*100,sharpe=sh,years=yr,win_months=wr*100)


# ── WALK-FORWARD ─────────────────────────────────────────────
def walk_forward_test(df_train, df_test, initial, monthly,
                      vix_range, mom_range, tnx_range):
    best_cagr, best_params, rows = -999, None, []
    combos = list(itertools.product(vix_range, mom_range, tnx_range))
    bar = st.progress(0, text=f"0 / {len(combos)} 조합 탐색 중...")
    for idx, (v, m, t) in enumerate(combos):
        sig=compute_signals(df_train,v,20,m,60,t)
        bt,_=run_backtest(sig,initial,monthly)
        con=build_contributions(bt.index,initial,monthly)
        met=calc_metrics(bt["oracle"],con)
        rows.append({"VIX":v,"모멘텀(%)":m,"TNX 한도(%)":t,
                     "학습 CAGR":round(met["cagr"],2),
                     "학습 MDD":round(met["mdd"],2),
                     "학습 샤프":round(met["sharpe"],2)})
        if met["cagr"]>best_cagr: best_cagr,best_params=met["cagr"],(v,m,t)
        bar.progress((idx+1)/len(combos),text=f"{idx+1}/{len(combos)} 탐색 중...")
    bar.progress(100, text="탐색 완료!")
    bv,bm,bt_=best_params
    sig_tr=compute_signals(df_train,bv,20,bm,60,bt_)
    bt_tr,_=run_backtest(sig_tr,initial,monthly)
    con_tr=build_contributions(bt_tr.index,initial,monthly)
    sig_te=compute_signals(df_test,bv,20,bm,60,bt_)
    bt_te,tr_te=run_backtest(sig_te,initial,monthly)
    con_te=build_contributions(bt_te.index,initial,monthly)
    return dict(grid=pd.DataFrame(rows).sort_values("학습 CAGR",ascending=False),
                best_vix=bv,best_mom=bm,best_tnx=bt_,
                tr_oracle=calc_metrics(bt_tr["oracle"],con_tr),
                tr_dca=calc_metrics(bt_tr["dca"],con_tr),
                te_oracle=calc_metrics(bt_te["oracle"],con_te),
                te_dca=calc_metrics(bt_te["dca"],con_te),
                bt_te=bt_te,trades_te=tr_te,con_te=con_te)


# ── FORMAT ───────────────────────────────────────────────────
def fmt(v):
    if v>=100_000_000: return f"{v/100_000_000:,.1f}억원"
    if v>=10_000: return f"{v/10_000:,.0f}만원"
    return f"{v:,.0f}원"
def fmt_full(v): return f"{v:,.0f}원"

AX  = dict(gridcolor="#151520", showgrid=True, zeroline=False)
LAY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,8,15,0.95)",
           font=dict(color="#6b7280",size=12),
           margin=dict(l=60,r=20,t=30,b=40), hovermode="x unified")


# ── SIDEBAR ──────────────────────────────────────────────────
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
    vix_t = st.slider("VIX 임계값",        15, 45,  25, 1)
    mom_t = st.slider("모멘텀 기준 (%)",    0.0, 10.0, 2.0, 0.5)
    tnx_t = st.slider("TNX 60일 한도 (%)", 1.0, 20.0, 5.0, 0.5)
    st.markdown("---")
    st.caption("📌 개인 교육용 도구 | 과거 수익률은 미래를 보장하지 않습니다.")


# ── LOAD & COMPUTE ───────────────────────────────────────────
st.markdown("# 🔮 나스닥 하이브리드 오라클 전략")
st.markdown("QQQ ↔ SOXL 스위칭 전략 | 10년 백테스트 검증 대시보드 · 개인 교육용")

with st.spinner("📡 데이터 로딩 중 (약 10~20초)..."):
    raw = load_data("2014-01-01")
if raw is None: st.stop()

sig_df     = compute_signals(raw, vix_t, 20, mom_t, 60, tnx_t)
bt, trades = run_backtest(sig_df, initial_cap, monthly_inv)
contrib    = build_contributions(bt.index, initial_cap, monthly_inv)
om         = calc_metrics(bt["oracle"], contrib)
dm         = calc_metrics(bt["dca"],    contrib)
_yr        = (sig_df.index[-1]-sig_df.index[0]).days/365.25
qqq_cagr   = ((sig_df["QQQ"].iloc[-1]/sig_df["QQQ"].iloc[0])**(1/_yr)-1)*100


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 백테스트 결과", "🎯 현재 시장 상태", "🚀 시뮬레이터", "🔬 과적합 검증"])


# ════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════
with tab1:
    gap  = om["cagr"]-dm["cagr"]
    tv   = om["total_invested"]
    yrs  = om["years"]

    with st.expander("📖 이 전략이 뭔가요? (클릭해서 보기)"):
        st.markdown(f"""
**오라클 전략** — 평소엔 QQQ 적립, 공황 반등 시점에 **SOXL 50% 스위칭**

| 조건 | 현재 기준 | 의미 |
|---|---|---|
| 🔴 VIX | {vix_t} 이상 | 시장이 충분히 겁먹은 상태 |
| 🟢 QQQ 20일 수익률 | {mom_t}% 이상 | 바닥 치고 회복 시작 |
| 🟡 금리 안정 | 60일 변화율 {tnx_t}% 미만 | 유동성 안전 |

QQQ 20일 수익률이 **0% 아래**로 떨어지면 → SOXL 전량 QQQ 복귀
        """)

    if om["cagr"] >= 22*0.90:
        st.success(f"✅ 주장된 22% CAGR → 백테스트 실측 **{om['cagr']:.1f}%** 재현")
    elif om["cagr"] > dm["cagr"]:
        st.warning(f"⚠️ 실측 CAGR **{om['cagr']:.1f}%** — QQQ 적립({dm['cagr']:.1f}%)은 초과")
    else:
        st.error(f"❌ 현재 파라미터 CAGR **{om['cagr']:.1f}%** < QQQ 적립 {dm['cagr']:.1f}%")

    st.markdown("---")
    st.markdown("### 💡 쉽게 말하면 이렇습니다")
    st.markdown(f"{int(yrs)}년간 **{fmt(initial_cap)}** 투자 + 매달 **{fmt(monthly_inv)}** 적립 (총 투입금 **{fmt(tv)}**)")

    co, cd, cdiff = st.columns(3)
    with co:
        st.markdown(f"""<div style="background:#0f1f16;border:1px solid #22c55e44;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">🔮 오라클 전략</div>
          <div style="font-size:28px;font-weight:700;color:#22c55e">{fmt(om['final_value'])}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">내 돈의 {om['final_value']/tv:.1f}배</div>
          <div style="font-size:12px;color:#4ade80;margin-top:2px">연평균 {om['cagr']:.1f}% 복리</div>
        </div>""", unsafe_allow_html=True)
    with cd:
        st.markdown(f"""<div style="background:#111827;border:1px solid #374151;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">📊 QQQ 그냥 적립</div>
          <div style="font-size:28px;font-weight:700;color:#9ca3af">{fmt(dm['final_value'])}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">내 돈의 {dm['final_value']/tv:.1f}배</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px">연평균 {dm['cagr']:.1f}% 복리</div>
        </div>""", unsafe_allow_html=True)
    with cdiff:
        extra=om["final_value"]-dm["final_value"]
        sc="#22c55e" if extra>=0 else "#f87171"
        st.markdown(f"""<div style="background:#0f0f1a;border:1px solid #6366f144;border-radius:12px;padding:1.2rem;text-align:center">
          <div style="font-size:12px;color:#6b7280;margin-bottom:6px">📐 전략 초과 수익</div>
          <div style="font-size:28px;font-weight:700;color:{sc}">{fmt(abs(extra))}</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px">{'더' if extra>=0 else '덜'} 벌었습니다</div>
          <div style="font-size:12px;color:#6366f1;margin-top:2px">스위칭 {len(trades)}회 (비용 0.1%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 연평균 수익률(CAGR)이 뭔지 모르겠어요"):
        st.markdown("""
**CAGR = 내 돈이 매년 평균 몇 %씩 불었나**

예: 1,000만원 → 10년 뒤 2,593만원 = CAGR **10%**
은행 예금 3.5% 대비, CAGR 13%는 **약 3.7배** 더 빠르게 불어납니다.

---
**왜 "QQQ 주가가 연 18% 올랐다"는데 내 수익률은 낮게 나오나요?**

매달 조금씩 넣으면 마지막에 넣은 돈은 1달밖에 못 굴립니다.
10년 전 돈은 18%를 10번, 1년 전 돈은 1번만 받습니다.
DCA CAGR이 가격 CAGR보다 낮은 건 **정상**입니다.
        """)

    mrows = [
        ("📈 QQQ 주가 상승률", qqq_cagr, "#818cf8", "주가 자체 상승 (적립과 무관)"),
        ("🔮 오라클 연평균",   om["cagr"],"#22c55e", f"적립 + 전략 | MDD {om['mdd']:.1f}%"),
        ("📊 QQQ 그냥 적립",   dm["cagr"],"#6b7280", f"매달 QQQ만 삼 | MDD {dm['mdd']:.1f}%"),
        ("🏦 정기예금 (참고)", 3.5,       "#374151", "위험 없는 기준선"),
    ]
    mx = max(m[1] for m in mrows)*1.15
    for label, val, color, note in mrows:
        st.markdown(f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
            <span style="color:#c0c0d8;font-size:14px">{label}</span>
            <span style="color:{color};font-weight:700;font-size:18px;font-family:monospace">{val:.1f}%</span>
          </div>
          <div style="background:#151520;border-radius:4px;height:10px;margin-bottom:4px">
            <div style="background:{color};width:{val/mx*100:.0f}%;height:100%;border-radius:4px"></div>
          </div>
          <div style="font-size:11px;color:#3e3e56">{note}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#1a1010;border:0.5px solid #f8717133;border-radius:10px;padding:1rem 1.2rem;margin:8px 0">
      <div style="color:#f87171;font-size:13px;font-weight:600;margin-bottom:6px">📉 MDD(최대 낙폭) — 투자 중 가장 많이 떨어진 순간</div>
      <div style="color:#9ca3af;font-size:13px">
        오라클 MDD <b style="color:#f87171">{om['mdd']:.1f}%</b> &nbsp;·&nbsp; QQQ 적립 MDD <b style="color:#fbbf24">{dm['mdd']:.1f}%</b>
      </div>
      <div style="color:#6b7280;font-size:12px;margin-top:4px">
        오라클 전략은 최악의 순간 전고점 대비 {abs(om['mdd']):.0f}%까지 빠졌습니다.
        1,000만원이 {1000*(1+om['mdd']/100):.0f}만원이 됐던 순간입니다.
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("💡 이 전략보다 좋은 전략이 있나요?"):
        st.markdown("""
| 전략 | 기대 수익 | MDD | 복잡도 |
|---|---|---|---|
| 오라클 (현 전략) | 높음 | 높음 | 복잡 |
| **200MA 필터** ⭐⭐⭐⭐ | 중간 | **낮음** | 단순 |
| **듀얼 모멘텀** ⭐⭐⭐⭐ | 중간 | **낮음** | 단순 |
| 올웨더 ⭐⭐⭐ | 낮음 | 매우 낮음 | 단순 |

**200MA 필터**: QQQ가 200일선 위면 보유, 아래면 현금. 2008·2022 큰 하락 회피.
**듀얼 모멘텀**: 매달 QQQ vs 채권 12개월 수익률 비교해서 높은 쪽 투자. 학술 검증 완료.
**올웨더**: 주식30%·장기채40%·중기채15%·금7.5%·원자재7.5%. MDD 매우 낮음.

수익보다 심리적 안정이 중요하다면 듀얼 모멘텀이나 200MA 필터가 더 현실적입니다.
        """)

    st.markdown("---")
    st.markdown("### 📈 누적 자산 변화 (2014 → 현재)")
    fig = go.Figure()
    soxl_flag=bt["in_soxl"].astype(int); trans=soxl_flag.diff().fillna(0)
    ss=bt.index[trans==1].tolist(); se=bt.index[trans==-1].tolist()
    if len(ss)>len(se): se.append(bt.index[-1])
    for s,e in zip(ss,se):
        fig.add_vrect(x0=s,x1=e,fillcolor="#6366f1",opacity=0.07,line_width=0,
                      annotation_text="SOXL" if (e-s).days>5 else "",
                      annotation_font_color="#6366f1",annotation_font_size=9)
    fig.add_trace(go.Scatter(x=bt.index,y=contrib,name="누적 투입금",
        line=dict(color="#2d2d4a",width=1.5,dash="dot"),
        fill="tozeroy",fillcolor="rgba(45,45,74,0.15)"))
    fig.add_trace(go.Scatter(x=bt.index,y=bt["dca"],name="QQQ DCA",
        line=dict(color="#6b7280",width=2,dash="dash")))
    fig.add_trace(go.Scatter(x=bt.index,y=bt["oracle"],name="오라클 전략",
        line=dict(color="#818cf8",width=2.5)))
    en=bt[bt["entry"]==True]; ex=bt[bt["exit_sig"]==True]
    fig.add_trace(go.Scatter(x=en.index,y=en["oracle"],mode="markers",name="SOXL 진입",
        marker=dict(color="#22c55e",size=9,symbol="triangle-up",line=dict(color="white",width=1))))
    fig.add_trace(go.Scatter(x=ex.index,y=ex["oracle"],mode="markers",name="QQQ 복귀",
        marker=dict(color="#f87171",size=7,symbol="triangle-down",line=dict(color="white",width=1))))
    fig.update_layout(**LAY,height=460,xaxis=dict(**AX),
        yaxis=dict(**AX,title="포트폴리오 (원)",tickformat=",.0f"),
        legend=dict(bgcolor="rgba(10,10,20,0.85)",bordercolor="#2a2a3a",borderwidth=1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📉 낙폭(Drawdown) 비교 차트"):
        ddo=(bt["oracle"]-bt["oracle"].cummax())/bt["oracle"].cummax()*100
        ddd=(bt["dca"]-bt["dca"].cummax())/bt["dca"].cummax()*100
        fdd=go.Figure()
        fdd.add_trace(go.Scatter(x=bt.index,y=ddd,name="QQQ DCA",
            line=dict(color="#6b7280",width=1.5),fill="tozeroy",fillcolor="rgba(107,114,128,0.1)"))
        fdd.add_trace(go.Scatter(x=bt.index,y=ddo,name="오라클",
            line=dict(color="#818cf8",width=2),fill="tozeroy",fillcolor="rgba(129,140,248,0.1)"))
        fdd.update_layout(**LAY,height=280,xaxis=dict(**AX),yaxis=dict(**AX,title="낙폭 (%)"))
        st.plotly_chart(fdd, use_container_width=True)

    st.markdown("### 🔔 스위칭 이력")
    if trades.empty:
        st.info("현재 파라미터에서 신호가 발생하지 않았습니다.")
    else:
        ca,cb=st.columns([3,1])
        with ca:
            def ca_fn(v):
                if v=="QQQ→SOXL": return "color:#22c55e;font-weight:600"
                if v=="SOXL→QQQ": return "color:#f87171;font-weight:600"
                return ""
            st.dataframe(trades.style.map(ca_fn,subset=["액션"]),
                         use_container_width=True,height=300)
        with cb:
            ne=(trades["액션"]=="QQQ→SOXL").sum()
            av=trades.loc[trades["액션"]=="QQQ→SOXL","VIX"].mean()
            st.metric("SOXL 진입 횟수",f"{ne}회")
            st.metric("QQQ 복귀 횟수",f"{(trades['액션']=='SOXL→QQQ').sum()}회")
            if not np.isnan(av): st.metric("진입 시 평균 VIX",f"{av:.1f}")

    st.info("**📌 백테스트 주의사항**  \n"
            "① 과거 수익률은 미래를 보장하지 않습니다  \n"
            "② 파라미터 최적화 → Overfitting 위험  \n"
            "③ 세금(22%)·환전비용·슬리피지 미반영  \n"
            "④ SOXL 3× 레버리지 — 장기 보유 시 변동성 잠식 주의")


# ════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════
with tab2:
    latest=sig_df.iloc[-1]; ld=sig_df.index[-1]
    cvix=float(latest["VIX"])
    cmom=float(latest.get("qqq_mom",np.nan))
    ctnx=float(latest.get("tnx_chg",np.nan))
    fm=bool(latest.get("f_market",False))
    fmo=bool(latest.get("f_momentum",False))
    fv=bool(latest.get("f_vix",False))
    nok=sum([fm,fmo,fv])
    if nok==3:   state,cs="🚀 가속 (SOXL 스위칭 시점)","#22c55e"; adv="모든 조건 충족 — SOXL 50% 스위칭 시점입니다."
    elif nok>=1: state,cs="⏳ 적립 대기","#fbbf24"; adv=f"{3-nok}개 조건 미충족 — QQQ DCA 유지하며 기다립니다."
    else:        state,cs="🛡️ 안전 모드 (DCA)","#6366f1"; adv="신호 없음 — 일반 QQQ 적립을 유지하세요."

    cg,cf=st.columns(2)
    with cg:
        fg=go.Figure(go.Indicator(mode="gauge+number",value=nok,
            title={"text":f"<b>{state}</b>","font":{"color":cs,"size":15}},
            number={"suffix":" / 3 조건","font":{"color":"#ddddf0","size":28}},
            gauge={"axis":{"range":[0,3],"tickcolor":"#6b7280"},
                   "bar":{"color":cs,"thickness":0.28},"bgcolor":"#10101e","borderwidth":0,
                   "steps":[{"range":[0,3],"color":"#141426"}],
                   "threshold":{"line":{"color":cs,"width":4},"thickness":0.8,"value":nok}}))
        fg.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#9ca3af"),
                         height=260,margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fg, use_container_width=True)
        st.caption(f"기준: {ld.strftime('%Y년 %m월 %d일')}")
    with cf:
        def fcard(label,val_str,ok,note):
            ic="✅" if ok else "❌"; bc="#22c55e33" if ok else "#f8717122"; vc="#22c55e" if ok else "#f87171"
            return (f'<div style="background:#10101e;border:0.5px solid {bc};border-radius:10px;padding:12px 16px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:#ddddf0;font-size:14px;font-weight:500">{ic} {label}</span>'
                    f'<span style="color:{vc};font-family:monospace;font-size:18px;font-weight:600">{val_str}</span>'
                    f'</div><div style="font-size:11px;color:#5a5a7a;margin-top:3px">{note}</div></div>')
        vs=f"{cvix:.1f}" if not np.isnan(cvix) else "N/A"
        ms=f"{cmom:+.1f}%" if not np.isnan(cmom) else "N/A"
        ts=f"{ctnx:+.1f}%" if not np.isnan(ctnx) else "N/A"
        st.markdown(
            fcard("VIX 공포지수",vs,fv,f"임계값 {vix_t} 이상 → {'✔ 충족' if fv else '✘ 미충족'}")+
            fcard("QQQ 20일 모멘텀",ms,fmo,f"20일 수익률 {mom_t}% 이상 → {'✔ 충족' if fmo else '✘ 미충족'}")+
            fcard("TNX 60일 변화율",ts,fm,f"금리 변화율 {tnx_t}% 미만 → {'✔ 충족' if fm else '✘ 미충족'}"),
            unsafe_allow_html=True)
        st.markdown(f'<div style="background:#10101e;border:0.5px solid {cs}44;border-radius:10px;padding:12px 16px;color:{cs};font-size:14px">💬 {adv}</div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📉 최근 90일 지표 추이")
    rc=sig_df.iloc[-90:].copy()
    fr=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=("VIX 공포지수","QQQ 20일 모멘텀 (%)","TNX 60일 변화율 (%)"),
        vertical_spacing=0.1,row_heights=[0.4,0.3,0.3])
    fr.add_trace(go.Scatter(x=rc.index,y=rc["VIX"],line=dict(color="#f87171",width=2),name="VIX"),row=1,col=1)
    fr.add_hline(y=vix_t,line=dict(color="#22c55e",width=1.2,dash="dash"),row=1,col=1,annotation_text=f"임계 {vix_t}",annotation_font_color="#22c55e")
    cm=["#22c55e" if v>0 else "#f87171" for v in rc["qqq_mom"].fillna(0)]
    fr.add_trace(go.Bar(x=rc.index,y=rc["qqq_mom"],marker_color=cm,name="모멘텀"),row=2,col=1)
    fr.add_hline(y=mom_t,line=dict(color="#6366f1",width=1.2,dash="dash"),row=2,col=1,annotation_text=f"임계 {mom_t}%",annotation_font_color="#6366f1")
    ct=["#f87171" if v>tnx_t else "#22c55e" for v in rc["tnx_chg"].fillna(0)]
    fr.add_trace(go.Bar(x=rc.index,y=rc["tnx_chg"],marker_color=ct,name="TNX"),row=3,col=1)
    fr.add_hline(y=tnx_t,line=dict(color="#f87171",width=1.2,dash="dash"),row=3,col=1,annotation_text=f"한도 {tnx_t}%",annotation_font_color="#f87171")
    fr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(8,8,15,0.95)",
                     font=dict(color="#6b7280"),showlegend=False,height=520,margin=dict(l=55,r=20,t=40,b=40))
    fr.update_xaxes(gridcolor="#151520"); fr.update_yaxes(gridcolor="#151520")
    st.plotly_chart(fr, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🚀 목표 자산 시뮬레이터")
    s1,s2,s3=st.columns(3)
    with s1: si=st.number_input("현재 보유 자산 (원)",0,500_000_000,0,100_000,format="%d")
    with s2: sm=st.number_input("월 적립 가능액 (원)",0,10_000_000,300_000,50_000,format="%d")
    with s3: stg=st.date_input("목표일",value=date(2027,6,1),min_value=date.today())
    mths=max(1,(stg.year-date.today().year)*12+(stg.month-date.today().month))

    def fv_calc(p0,pmt,n,ar):
        r=(1+ar)**(1/12)-1
        return p0*(1+r)**n+(pmt*(((1+r)**n-1)/r) if r>0.0001 else pmt*n)

    ro,rd,rs=om["cagr"]/100,dm["cagr"]/100,0.04
    vo=fv_calc(si,sm,mths,ro); vd=fv_calc(si,sm,mths,rd)
    vsa=fv_calc(si,sm,mths,rs); vt=si+sm*mths

    st.markdown("---")
    st.markdown(f"#### {stg.strftime('%Y년 %m월 %d일')} 기준 예상 자산 ({mths}개월 후)")
    st.caption(f"총 투입 예정금: {fmt_full(vt)} (현재 {fmt_full(si)} + 월 {fmt_full(sm)} × {mths}개월)")
    m1,m2,m3=st.columns(3)
    m1.metric("🛡️ 안전 투자 (연 4%)",fmt_full(vsa),f"+{(vsa/vt-1)*100:.0f}%" if vt>0 else "—")
    m2.metric("📊 QQQ 그냥 적립",fmt_full(vd),f"+{(vd/vt-1)*100:.0f}%" if vt>0 else "—")
    m3.metric(f"🔮 오라클 (연 {ro*100:.1f}%)",fmt_full(vo),f"+{(vo/vt-1)*100:.0f}%" if vt>0 else "—")

    sim_m=[*range(mths+1)]; sd=[date.today().replace(day=1)]
    for _ in range(mths):
        d=sd[-1]; nm=d.month+1 if d.month<12 else 1; ny=d.year if d.month<12 else d.year+1
        sd.append(d.replace(year=ny,month=nm))
    fs=go.Figure()
    fs.add_trace(go.Scatter(x=sd,y=[si+sm*m for m in sim_m],name="투입 원금",
        line=dict(color="#2d2d4a",width=1.5,dash="dot"),fill="tozeroy",fillcolor="rgba(45,45,74,0.15)"))
    fs.add_trace(go.Scatter(x=sd,y=[fv_calc(si,sm,m,rs) for m in sim_m],name="안전(4%)",line=dict(color="#374151",width=1.5)))
    fs.add_trace(go.Scatter(x=sd,y=[fv_calc(si,sm,m,rd) for m in sim_m],name="QQQ 적립",line=dict(color="#6b7280",width=2,dash="dash")))
    fs.add_trace(go.Scatter(x=sd,y=[fv_calc(si,sm,m,ro) for m in sim_m],name="오라클",line=dict(color="#22c55e",width=2.5)))
    fs.update_layout(**LAY,height=300,xaxis=dict(**AX),yaxis=dict(**AX,tickformat=",.0f",title="예상 자산 (원)"),
        legend=dict(bgcolor="rgba(10,10,20,0.85)",bordercolor="#2a2a3a",borderwidth=1))
    st.plotly_chart(fs, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🎯 내 목표 달성도")
    st.caption("목표를 직접 입력하면 각 전략별 달성 가능 여부를 보여줍니다.")
    if "n_goals" not in st.session_state: st.session_state.n_goals=2
    _,gb=st.columns([5,1])
    with gb:
        if st.button("+ 목표 추가"): st.session_state.n_goals=min(st.session_state.n_goals+1,6)
    dn=["첫 번째 목표","두 번째 목표","세 번째 목표","네 번째 목표","다섯 번째 목표","여섯 번째 목표"]
    da=[30_000_000,100_000_000,50_000_000,200_000_000,500_000_000,1_000_000_000]
    goals=[]
    for i in range(st.session_state.n_goals):
        g1,g2=st.columns(2)
        with g1: gn=st.text_input(f"목표{i+1}이름",value=dn[i],key=f"gname_{i}",label_visibility="collapsed",placeholder=f"목표 {i+1} 이름")
        with g2: ga=st.number_input(f"목표{i+1}금액",value=da[i],step=1_000_000,format="%d",key=f"gamt_{i}",label_visibility="collapsed")
        goals.append((gn,ga))
    st.markdown("<br>",unsafe_allow_html=True)
    for gn,ga in goals:
        if ga<=0: continue
        st.markdown(f"<div style='color:#c0c0d8;font-size:14px;font-weight:600;margin-bottom:8px'>🎯 {gn} — 목표 {fmt_full(ga)}</div>",unsafe_allow_html=True)
        for sn,sv,sc in [("🛡️ 안전(4%)",vsa,"#4b5563"),("📊 QQQ적립",vd,"#6b7280"),("🔮 오라클",vo,"#22c55e")]:
            r=sv/ga; bw=min(r*100,100); c="#22c55e" if r>=1 else "#fbbf24" if r>=0.5 else sc; lb="✅ 달성!" if r>=1 else f"{r*100:.0f}%"
            st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
              <span style="color:#6b7280;font-size:12px;min-width:72px">{sn}</span>
              <div style="flex:1;background:#151520;border-radius:3px;height:7px"><div style="background:{c};width:{bw:.0f}%;height:100%;border-radius:3px"></div></div>
              <span style="color:{c};font-size:12px;font-weight:600;min-width:88px;text-align:right">{fmt(sv)} ({lb})</span>
            </div>""",unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    st.caption(f"⚠️ 오라클 CAGR {ro*100:.1f}%가 미래에도 동일하다는 낙관적 가정입니다.")


# ════════════════════════════════════════════════════════
# TAB 4 — WALK-FORWARD TEST
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔬 과적합 검증 — Walk-Forward Test")

    with st.expander("📖 이게 왜 필요한가요?", expanded=True):
        st.markdown("""
**슬라이더를 움직여 수익이 가장 높은 파라미터를 찾는 것 = Overfitting(과적합)**

과거 데이터에 파라미터를 맞추면 항상 좋아 보입니다.
미래에도 통하는지는 **보지 못한 데이터에서 검증**해야 알 수 있습니다.

#### Walk-Forward Test 방법
1. 전체 데이터를 **학습 구간 / 검증 구간**으로 분리
2. 학습 구간에서만 최적 파라미터 탐색
3. 찾은 파라미터를 검증 구간에 **그대로 (수정 없이)** 적용
4. 검증 구간에서도 QQQ 적립을 이기면 → 전략이 실제로 유효
5. 검증 구간에서 무너지면 → 과적합, 미래에 통하지 않을 가능성 높음

#### ⚠️ VIX=15, 모멘텀=0이 수익이 높게 나오는 이유
이 파라미터는 조건이 거의 항상 충족되어 SOXL을 계속 보유하는 것과 같습니다.
2014~2024가 반도체 역사상 최고의 불장이었기 때문에 좋아 보이는 것이지,
전략의 우수성이 아닙니다. 2022년 금리 급등기에도 통했는지 검증해 보세요.
        """)

    st.markdown("---")
    w1,w2=st.columns(2)
    with w1:
        tey=st.slider("학습 구간 종료 연도",2017,2022,2020)
        st.caption(f"학습: 2014-01-01 ~ {tey}-12-31  \n검증: {tey+1}-01-01 ~ 현재")
    with w2:
        ntr=(raw[raw.index<=f"{tey}-12-31"]).shape[0]
        nte=(raw[raw.index>f"{tey}-12-31"]).shape[0]
        st.metric("학습 구간 거래일",f"{ntr:,}일"); st.metric("검증 구간 거래일",f"{nte:,}일")

    st.markdown("#### 🔍 탐색할 파라미터 후보")
    p1,p2,p3=st.columns(3)
    with p1: vo_=st.multiselect("VIX 후보값",[15,20,25,30,35,40],default=[20,25,30,35])
    with p2: mo_=st.multiselect("모멘텀 후보 (%)",[0,1,2,3,5,8],default=[1,2,3,5])
    with p3: to_=st.multiselect("TNX 한도 후보 (%)",[3,5,8,10,15],default=[3,5,8,10])

    nc=len(vo_)*len(mo_)*len(to_)
    st.caption(f"총 {nc}개 조합 (약 {max(5,nc//8)}~{max(10,nc//4)}초 소요)")

    if st.button("🚀 검증 실행", type="primary", disabled=(nc==0)):
        dtr=raw[raw.index<=f"{tey}-12-31"].copy()
        dte=raw[raw.index>f"{tey}-12-31"].copy()
        if len(dte)<100:
            st.error("검증 구간이 너무 짧습니다. 학습 종료 연도를 앞으로 당겨주세요.")
        else:
            wf=walk_forward_test(dtr,dte,initial_cap,monthly_inv,vo_,mo_,to_)
            st.session_state["wf"]=wf

    if "wf" in st.session_state:
        wf=st.session_state["wf"]
        bv,bm,bt_=wf["best_vix"],wf["best_mom"],wf["best_tnx"]
        st.markdown("---")
        st.markdown("#### 🏆 학습 구간 최적 파라미터")
        x1,x2,x3=st.columns(3)
        x1.metric("VIX 임계값",f"{bv}"); x2.metric("모멘텀 기준",f"{bm}%"); x3.metric("TNX 60일 한도",f"{bt_}%")

        st.markdown("---")
        st.markdown("#### 📊 학습 vs 검증 성과 비교")
        st.markdown("**검증 구간에서도 QQQ 적립을 이기는가**가 핵심입니다.")

        cdf=pd.DataFrame({
            "구간":["학습","학습","검증","검증"],
            "전략":["오라클","QQQ 적립","오라클","QQQ 적립"],
            "CAGR (%)":[round(wf["tr_oracle"]["cagr"],1),round(wf["tr_dca"]["cagr"],1),
                         round(wf["te_oracle"]["cagr"],1),round(wf["te_dca"]["cagr"],1)],
            "MDD (%)":[round(wf["tr_oracle"]["mdd"],1),round(wf["tr_dca"]["mdd"],1),
                       round(wf["te_oracle"]["mdd"],1),round(wf["te_dca"]["mdd"],1)],
            "샤프":[round(wf["tr_oracle"]["sharpe"],2),round(wf["tr_dca"]["sharpe"],2),
                   round(wf["te_oracle"]["sharpe"],2),round(wf["te_dca"]["sharpe"],2)],
        })
        st.dataframe(cdf, use_container_width=True, hide_index=True)

        tg=wf["te_oracle"]["cagr"]-wf["te_dca"]["cagr"]
        trg=wf["tr_oracle"]["cagr"]-wf["tr_dca"]["cagr"]
        if tg>0 and (trg-tg)<trg*0.5:
            st.success(f"✅ 과적합 없음 — 검증 구간에서도 QQQ 적립 대비 +{tg:.1f}%p 초과 (학습 +{trg:.1f}%p → 검증 +{tg:.1f}%p)")
        elif tg>0:
            st.warning(f"⚠️ 성과 감소 — 검증 구간에서 여전히 QQQ를 이기지만 학습보다 줄었습니다 (학습 +{trg:.1f}%p → 검증 +{tg:.1f}%p)")
        else:
            st.error(f"❌ 과적합 의심 — 검증 구간에서 QQQ 단순 적립보다 {abs(tg):.1f}%p 낮습니다. 학습 구간 과적합 가능성 높음")

        st.markdown("---")
        st.markdown("#### 📈 검증 구간 자산 비교 차트")
        st.caption(f"최적 파라미터 (VIX={bv}, 모멘텀={bm}%, TNX={bt_}%)를 검증 구간에 그대로 적용한 결과")
        bte=wf["bt_te"]; cte=wf["con_te"]
        fw=go.Figure()
        fw.add_trace(go.Scatter(x=bte.index,y=cte,name="누적 투입금",
            line=dict(color="#2d2d4a",width=1.5,dash="dot"),fill="tozeroy",fillcolor="rgba(45,45,74,0.15)"))
        fw.add_trace(go.Scatter(x=bte.index,y=bte["dca"],name="QQQ 적립 (검증)",
            line=dict(color="#6b7280",width=2,dash="dash")))
        fw.add_trace(go.Scatter(x=bte.index,y=bte["oracle"],name="오라클 (검증)",
            line=dict(color="#22c55e",width=2.5)))
        ete=bte[bte["entry"]==True]; xte=bte[bte["exit_sig"]==True]
        fw.add_trace(go.Scatter(x=ete.index,y=ete["oracle"],mode="markers",name="SOXL 진입",
            marker=dict(color="#22c55e",size=9,symbol="triangle-up",line=dict(color="white",width=1))))
        fw.add_trace(go.Scatter(x=xte.index,y=xte["oracle"],mode="markers",name="QQQ 복귀",
            marker=dict(color="#f87171",size=7,symbol="triangle-down",line=dict(color="white",width=1))))
        fw.update_layout(**LAY,height=400,xaxis=dict(**AX),
            yaxis=dict(**AX,title="포트폴리오 (원)",tickformat=",.0f"),
            legend=dict(bgcolor="rgba(10,10,20,0.85)",bordercolor="#2a2a3a",borderwidth=1))
        st.plotly_chart(fw, use_container_width=True)

        with st.expander("🗺️ 파라미터 탐색 결과 (상위 20개)"):
            st.dataframe(wf["grid"].head(20).reset_index(drop=True),
                         use_container_width=True)
            st.caption("학습 구간 기준 정렬. 검증 구간에서도 좋은 파라미터가 진짜 유효합니다.")


# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;color:#2d2d4a;font-size:11px;padding:.5rem 0">'
            "개인 교육용 백테스팅 도구 | 투자 권유 아님 | 과거 수익률은 미래를 보장하지 않습니다 | "
            "Built with Streamlit · yfinance · Plotly</div>", unsafe_allow_html=True)
