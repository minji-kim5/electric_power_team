import time
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="12월 예측 - 컨트롤", layout="wide")
st.title("⚡ 실시간 전력 및 전기요금 모니터링")

# ---- 데이터 로드 (초기 1회만) ----
@st.cache_data
def load_data():
    df = pd.read_csv('../data/test5.csv')
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    return df

# ---- 세션 상태 초기화 ----
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("step", 0)
ss.setdefault("accumulated_data", pd.DataFrame())
ss.setdefault("data_loaded", False)

# 데이터 로드
if not ss.data_loaded:
    ss.full_data = load_data()
    ss.data_loaded = True

# ---- 사이드바 컨트롤 ----
st.sidebar.header("⚙️ 제어판")
start = st.sidebar.button("▶ 재생", type="primary", use_container_width=True)
stop = st.sidebar.button("⏸ 정지", use_container_width=True)
reset = st.sidebar.button("⟲ 리셋", use_container_width=True)

if start:
    ss.running = True
if stop:
    ss.running = False
if reset:
    ss.running = False
    ss.step = 0
    ss.accumulated_data = pd.DataFrame()

# ---- 데이터 누적 로직 ----
if ss.running and ss.step < len(ss.full_data):
    # 현재 step에 해당하는 데이터 1개 추가
    current_row = ss.full_data.iloc[ss.step:ss.step+1]
    ss.accumulated_data = pd.concat([ss.accumulated_data, current_row], ignore_index=True)
    ss.step += 1

# ---- 메인 대시보드 ----
if len(ss.accumulated_data) > 0:
    df = ss.accumulated_data
    latest = df.iloc[-1]  # 최신 데이터
    
    # === KPI 카드 4개 ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 누적 전력사용량",
            value=f"{df['전력사용량_예측'].sum():.2f} kWh",
            delta=f"+{latest['전력사용량_예측']:.2f} kWh"
        )
    
    with col2:
        st.metric(
            label="💰 누적 전기요금",
            value=f"{df['전기요금_예측'].sum():,.0f} 원",
            delta=f"+{latest['전기요금_예측']:,.0f} 원"
        )
    
    with col3:
        st.metric(
            label="🌱 누적 탄소배출량",
            value=f"{df['탄소배출량_예측'].sum():.4f} tCO2",
            delta=f"+{latest['탄소배출량_예측']:.4f} tCO2"
        )
    
    with col4:
        status_color = "🟢" if latest['작업휴무'] == '가동' else "🔴"
        load_emoji = {"Light_Load": "🔵", "Medium_Load": "🟡", "Maximum_Load": "🔴"}
        st.metric(
            label="⚙️ 운영 상태",
            value=f"{status_color} {latest['작업휴무']}",
            delta=f"{load_emoji.get(latest['작업유형'], '⚪')} {latest['작업유형']}"
        )
    
    st.divider()
    
    # === 실시간 전력사용량 라인차트 ===
    st.subheader("📈 실시간 전력사용량 추이")
    fig = px.line(
        df, 
        x='측정일시', 
        y='전력사용량_예측',
        labels={'측정일시': '시간', '전력사용량_예측': '전력사용량 (kWh)'},
        template='plotly_white'
    )
    fig.update_traces(line_color='#1f77b4', line_width=2)
    fig.update_layout(
        height=400,
        xaxis_title='측정일시',
        yaxis_title='전력사용량 (kWh)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # === 데이터 로그 (최신 5개) ===
    st.subheader("📋 최근 데이터 로그")
    
    # 최신 5개 데이터
    recent_data = df.tail(5)[['측정일시', '작업유형', '작업휴무', '지상역률(%)', '진상역률(%)']].reset_index(drop=True)
    recent_data_full = df.tail(5).reset_index(drop=True)
    
    # 데이터 테이블 표시
    st.dataframe(
        recent_data, 
        use_container_width=True,
        hide_index=False,
        height=220
    )
    
    # 행 선택
    selected_row = st.selectbox(
        "🔍 상세 정보를 확인할 데이터 행 선택:",
        options=range(len(recent_data)),
        format_func=lambda x: f"행 {x} - {recent_data.iloc[x]['측정일시']}"
    )
    
    if selected_row is not None:
        selected_detail = recent_data_full.iloc[selected_row]
        
        st.info("💡 선택된 데이터 상세 정보")
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric("전력사용량 예측", f"{selected_detail['전력사용량_예측']:.2f} kWh")
        with detail_col2:
            st.metric("탄소배출량 예측", f"{selected_detail['탄소배출량_예측']:.6f} tCO2")
        with detail_col3:
            st.metric("전기요금 예측", f"{selected_detail['전기요금_예측']:,.0f} 원")
    
    # 진행 상태 표시
    st.divider()
    progress_text = f"진행 상황: {ss.step}/{len(ss.full_data)} ({ss.step/len(ss.full_data)*100:.1f}%)"
    st.progress(ss.step / len(ss.full_data), text=progress_text)

else:
    st.info("▶ '재생' 버튼을 눌러 모니터링을 시작하세요.")
    st.caption("📍 데이터가 2초마다 자동으로 업데이트됩니다.")

# ---- 자동 반복 (2초 간격) ----
if ss.running and ss.step < len(ss.full_data):
    time.sleep(2.0)  # 1초 → 2초로 변경
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
elif ss.running and ss.step >= len(ss.full_data):
    st.success("✅ 모든 데이터 처리 완료!")
    ss.running = False