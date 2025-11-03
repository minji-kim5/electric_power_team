import time
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="12월 예측 - 컨트롤", layout="wide")

# ---- 커스텀 CSS (카드 스타일) ----
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .metric-card-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-delta {
        font-size: 12px;
        opacity: 0.8;
    }
    
    /* 상세 정보 팝업의 metric 글씨 크기 축소 */
    [data-testid="stMetric"] {
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ 실시간 전력 및 전기요금 모니터링")

# ---- 데이터 로드 (초기 1회만) ----
@st.cache_data
def load_data():
    df = pd.read_csv('../data/test6.csv')
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    return df

# ---- 세션 상태 초기화 ----
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("step", 0)
ss.setdefault("accumulated_data", pd.DataFrame())
ss.setdefault("data_loaded", False)
ss.setdefault("popup_open", False)

# 데이터 로드
if not ss.data_loaded:
    ss.full_data = load_data()
    ss.data_loaded = True

# ---- 사이드바 컨트롤 ----
st.sidebar.header("⚙️ 제어판")
start = st.sidebar.button("▶ 재생", type="primary", use_container_width=True)
stop = st.sidebar.button("⏸ 정지", use_container_width=True)
reset = st.sidebar.button("⟲ 리셋", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("📊 차트 옵션")

# 데이터 출력 간격 조정 슬라이더
update_interval = st.sidebar.slider(
    "데이터 출력 간격 (초)",
    min_value=0.1,
    max_value=3.0,
    value=2.0,
    step=0.1,
    key="update_interval"
)

show_peak_line = st.sidebar.checkbox("피크전력선 표시", value=False, key="show_peak")
show_pf_line = st.sidebar.checkbox("기준역률선 표시", value=False, key="show_pf")

if start:
    ss.running = True
    ss.popup_open = False
if stop:
    ss.running = False
if reset:
    ss.running = False
    ss.step = 0
    ss.accumulated_data = pd.DataFrame()
    ss.popup_open = False

# ---- 데이터 누적 로직 ----
if ss.running and ss.step < len(ss.full_data) and not ss.popup_open:
    current_row = ss.full_data.iloc[ss.step:ss.step+1]
    ss.accumulated_data = pd.concat([ss.accumulated_data, current_row], ignore_index=True)
    ss.step += 1

# ---- 메인 대시보드 ----
if len(ss.accumulated_data) > 0:
    df = ss.accumulated_data.copy()
    
    # 탄소배출량을 kg으로 변환 (사용 시점에 계산)
    df['탄소배출량_kg'] = df['탄소배출량_예측'] * 1000
    
    latest = df.iloc[-1]
    
    # === KPI 카드 4개 (커스텀 HTML 카드) ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <div class="metric-label">📊 누적 전력사용량</div>
            <div class="metric-value"><strong>{df['전력사용량_예측'].sum():.2f}</strong> kWh</div>
            <div class="metric-delta">+{latest['전력사용량_예측']:.2f} kWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">💰 누적 전기요금</div>
            <div class="metric-value"><strong>{df['전기요금_예측'].sum():,.0f}</strong> 원</div>
            <div class="metric-delta">+{latest['전기요금_예측']:,.0f} 원</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-orange">
            <div class="metric-label">🌱 누적 탄소배출량</div>
            <div class="metric-value"><strong>{df['탄소배출량_kg'].sum():.2f}</strong> kgCO2</div>
            <div class="metric-delta">+{latest['탄소배출량_kg']:.2f} kgCO2</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status_emoji = "🟢" if latest['작업휴무'] == '가동' else "🔴"
        load_text = latest['작업유형'].replace('_', ' ')
        
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">⚙️ 운영 상태</div>
            <div class="metric-value">{status_emoji} <strong>{latest['작업휴무']}</strong></div>
            <div class="metric-delta">{load_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # === 실시간 전력사용량 라인차트 + 당일 전력사용량 게이지 ===
    st.subheader("📈 실시간 전력사용량 추이 및 당일 전력사용량")
    
    chart_col, gauge_col = st.columns([3, 1])
    
    with chart_col:
        # 최근 30개 데이터만 표시
        df_chart = df.tail(30).copy()
        
        # 라인차트 전용: 자정(00:00) 시간 데이터의 날짜를 다음날로 수정
        mask = df_chart['측정일시'].dt.time == pd.Timestamp('00:00:00').time()
        df_chart.loc[mask, '측정일시'] = df_chart.loc[mask, '측정일시'] + pd.Timedelta(days=1)
        
        # 피크 전력 계산 (기존 최대값 157.18 kWh 기준)
        BASE_PEAK = 157.18
        current_max = df['전력사용량_예측'].max()
        peak_power = max(BASE_PEAK, current_max)  # 157.18을 넘으면 갱신
        
        fig = px.line(
            df_chart, 
            x='측정일시', 
            y='전력사용량_예측',
            labels={'측정일시': '시간', '전력사용량_예측': '전력사용량 (kWh)'},
            template='plotly_white'
        )
        fig.update_traces(
            line_color='#1f77b4', 
            line_width=2,
            mode='lines+markers',
            marker=dict(size=6, color='#1f77b4')
        )
        
        # 피크 전력 기준선 추가 (체크박스로 제어)
        if show_peak_line:
            fig.add_hline(
                y=peak_power, 
                line_dash="dash", 
                line_color="red", 
                line_width=2,
                annotation_text=f"피크: {peak_power:.2f} kWh",
                annotation_position="right"
            )
        
        # y축 범위 동적 조정
        if show_peak_line:
            # 피크선 표시 시: 피크값까지 표시
            y_max = peak_power * 1.1
        else:
            # 피크선 미표시 시: 현재 차트 데이터 기준
            y_max = df_chart['전력사용량_예측'].max() * 1.15
        
        fig.update_layout(
            height=450,
            xaxis_title='측정일시',
            yaxis_title='전력사용량 (kWh)',
            yaxis_range=[0, y_max],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_col:
        # 현재 날짜 추출
        current_date = latest['측정일시'].date()
        date_str = current_date.strftime('%Y년 %m월 %d일')
        
        # 당일 데이터만 필터링하여 누적 전력사용량 계산
        df_today = df[df['측정일시'].dt.date == current_date]
        total_power = df_today['전력사용량_예측'].sum()
        
        # 현재 날짜의 작업휴무 상태 확인
        current_status = latest['작업휴무']
        
        # 작업휴무에 따른 기준 설정 (3~11월 데이터 기준)
        if current_status == '가동':
            threshold_95 = 4270  # 가동일 95% 분위수
            max_range = 6500     # 게이지 최대 범위
            status_text = "가동일"
            bar_color = "#1f77b4"
            # 눈금 위치 (피크 기준 포함)
            tick_vals = [0, 1300, 2600, threshold_95, 5200, max_range]
        else:  # 휴무
            threshold_95 = 360   # 휴무일 95% 분위수
            max_range = 550      # 게이지 최대 범위
            status_text = "휴무일"
            bar_color = "#90CAF9"
            # 눈금 위치 (피크 기준 포함)
            tick_vals = [0, 110, 220, threshold_95, 440, max_range]
        
        # 세로 게이지 차트
        import plotly.graph_objects as go
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_power,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"<b>당일 전력사용량<br>({status_text}, {date_str})</b>", 
                'font': {'size': 14, 'color': '#000000'}
            },
            number={
                'suffix': ' kWh', 
                'font': {'size': 42, 'color': '#000000'},
                'valueformat': '.0f'
            },
            gauge={
                'axis': {
                    'range': [None, max_range], 
                    'tickwidth': 1, 
                    'tickcolor': "darkblue",
                    'tickmode': 'array',
                    'tickvals': tick_vals,
                    'ticktext': [str(int(v)) for v in tick_vals],
                    'tickfont': {'size': 11}
                },
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold_95], 'color': '#E8F5E9'},
                    {'range': [threshold_95, max_range], 'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_95
                }
            }
        ))
        
        gauge_fig.update_layout(
            height=450,
            margin=dict(l=30, r=30, t=80, b=30)
        )
        
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.divider()
    
    # === 역률 실시간 추이 + 시간대별 부하 ===
    st.subheader("📶 실시간 역률 추이 및 시간대별 부하")
    
    pf_col, load_col = st.columns([3, 1])
    
    with pf_col:
        # 역률 통합 차트 (지상역률 + 진상역률)
        df_chart_pf = df.tail(30).copy()
        
        # 라인차트 전용: 자정(00:00) 시간 데이터의 날짜를 다음날로 수정
        mask = df_chart_pf['측정일시'].dt.time == pd.Timestamp('00:00:00').time()
        df_chart_pf.loc[mask, '측정일시'] = df_chart_pf.loc[mask, '측정일시'] + pd.Timedelta(days=1)
        
        # 데이터 재구성 (wide to long)
        df_pf_long = pd.melt(
            df_chart_pf,
            id_vars=['측정일시'],
            value_vars=['지상역률(%)', '진상역률(%)'],
            var_name='역률 유형',
            value_name='역률 값'
        )
        
        fig_pf = px.line(
            df_pf_long,
            x='측정일시',
            y='역률 값',
            color='역률 유형',
            labels={'측정일시': '시간', '역률 값': '역률 (%)'},
            template='plotly_white',
            color_discrete_map={'지상역률(%)': '#FF6B6B', '진상역률(%)': '#4ECDC4'}
        )
        fig_pf.update_traces(
            line_width=2,
            mode='lines+markers',
            marker=dict(size=5)
        )
        
        # 기준역률선 추가 (체크박스로 제어)
        if show_pf_line:
            # 현재 시각(최신 데이터) 기준으로 판단
            time_val = latest['측정일시']
            hour = time_val.hour
            minute = time_val.minute
            
            # 시간을 소수로 변환 (09:15 = 9.25)
            time_decimal = hour + minute / 60.0
            
            # 9:15 AM (9.25) ~ 10:00 PM (22.0): 지상역률 90%
            # 그 외 시간: 진상역률 95%
            if 9.25 <= time_decimal < 22.0:
                # 지상역률 기준선 (90%)
                fig_pf.add_hline(
                    y=90,
                    line_dash="dash",
                    line_color="#FF6B6B",
                    line_width=2,
                    annotation_text="지상역률 기준 90%",
                    annotation_position="right"
                )
            else:
                # 진상역률 기준선 (95%)
                fig_pf.add_hline(
                    y=95,
                    line_dash="dash",
                    line_color="#4ECDC4",
                    line_width=2,
                    annotation_text="진상역률 기준 95%",
                    annotation_position="right"
                )
        
        fig_pf.update_layout(
            height=450,
            xaxis_title='측정일시',
            yaxis_title='역률 (%)',
            yaxis_range=[0, 105],
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_pf, use_container_width=True)
    
    with load_col:
        # 시간대별 부하 원형 차트 (Barpolar 사용)
        import plotly.graph_objects as go
        import numpy as np
        
        # 현재 날짜와 작업휴무 상태 확인
        current_date = latest['측정일시'].date()
        current_status = latest['작업휴무']
        current_time = latest['측정일시'].time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # 가동일/휴무일에 따른 시간대별 부하 설정
        if current_status == '가동':
            # 가동일 부하 구간
            load_segments = [
                {'start': 0, 'end': 9, 'load': '경부하', 'color': '#90EE90'},
                {'start': 9, 'end': 10, 'load': '중간부하', 'color': '#FFD700'},
                {'start': 10, 'end': 12, 'load': '최대부하', 'color': '#FF6B6B'},
                {'start': 12, 'end': 17, 'load': '중간부하', 'color': '#FFD700'},
                {'start': 17, 'end': 20, 'load': '최대부하', 'color': '#FF6B6B'},
                {'start': 20, 'end': 22, 'load': '중간부하', 'color': '#FFD700'},
                {'start': 22, 'end': 23, 'load': '최대부하', 'color': '#FF6B6B'},
                {'start': 23, 'end': 24, 'load': '경부하', 'color': '#90EE90'}
            ]
            status_display = '가동일'
        else:
            # 휴무일 부하 구간
            load_segments = [
                {'start': 0, 'end': 24, 'load': '경부하', 'color': '#90EE90'}
            ]
            status_display = '휴무일'
        
        # Barpolar 차트 생성
        fig_load = go.Figure()
        
        # 각 부하 구간을 bar로 추가
        load_types = {'경부하': True, '중간부하': True, '최대부하': True}
        
        for segment in load_segments:
            start_hour = segment['start']
            end_hour = segment['end']
            duration = end_hour - start_hour
            
            # 중심 각도 계산 (0시 = 0도 위쪽, 시계방향)
            center_hour = (start_hour + end_hour) / 2
            theta = center_hour * 15  # 시간당 15도, 시계방향
            
            # 각도 폭 계산
            width = duration * 15
            
            # 범례 표시 여부 (각 부하 유형당 한 번만)
            show_legend = load_types.get(segment['load'], False)
            if show_legend:
                load_types[segment['load']] = False
            
            fig_load.add_trace(go.Barpolar(
                r=[1],  # 반지름
                theta=[theta],
                width=[width],
                base=0.8,  # 0.8~1.0 범위로 복구
                marker=dict(
                    color=segment['color'],
                    line=dict(color='white', width=2)
                ),
                name=segment['load'],
                showlegend=show_legend,
                hovertemplate=f"{start_hour:02d}:00-{end_hour:02d}:00<br>{segment['load']}<extra></extra>"
            ))
        
        # 시간 표기 추가 (0~23시 모두 표시, 굵게)
        for hour in range(24):
            theta = hour * 15  # 시계방향
            fig_load.add_trace(go.Scatterpolar(
                r=[1.35],
                theta=[theta],
                mode='text',
                text=[f'<b>{hour}</b>'],  # 굵게 표시
                textfont=dict(size=11, color='#333333'),  # 크기 증가 및 진한 색상
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 시계바늘 추가
        time_in_hours = current_hour + current_minute / 60.0
        needle_theta = time_in_hours * 15  # 시계방향
        
        fig_load.add_trace(go.Scatterpolar(
            r=[0, 0.85],
            theta=[needle_theta, needle_theta],
            mode='lines',
            line=dict(color='gray', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 시계바늘 끝 화살표
        fig_load.add_trace(go.Scatterpolar(
            r=[0.85],
            theta=[needle_theta],
            mode='markers',
            marker=dict(
                size=12,
                color='gray',
                symbol='arrow',
                angle=needle_theta,
                angleref='up'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 중심 점
        fig_load.add_trace(go.Scatterpolar(
            r=[0],
            theta=[0],
            mode='markers',
            marker=dict(size=10, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_load.update_layout(
            title=dict(
                text=f"시간대별 부하<br>({status_display})",
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, 1.5]
                ),
                angularaxis=dict(
                    visible=False,
                    direction='clockwise',
                    rotation=90
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=11)  # 범례 크기 증가
            ),
            height=450,
            margin=dict(l=10, r=10, t=80, b=40)
        )
        
        st.plotly_chart(fig_load, use_container_width=True)
    
    st.divider()
    
    # === 데이터 로그 (행 선택 가능) ===
    st.subheader("📋 최근 데이터 로그")
    
    # 최신 5개 데이터
    recent_data = df.tail(5)[['측정일시', '작업유형', '작업휴무', '지상역률(%)', '진상역률(%)']].copy()
    recent_data_full = df.tail(5).copy().reset_index(drop=True)
    recent_data = recent_data.reset_index(drop=True)
    
    # 데이터프레임 표시 (선택 모드, 인덱스 숨김)
    event = st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True,
        height=220,
        selection_mode="single-row",
        on_select="rerun",
        key="data_table"
    )
    
    # 행 선택 시 팝업 표시
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_detail = recent_data_full.iloc[selected_idx]
        
        # 팝업이 열려있음을 표시
        ss.popup_open = True
        
        # 팝업 다이얼로그
        @st.dialog("📊 상세 정보")
        def show_detail():
            st.markdown(f"### 측정일시: {selected_detail['측정일시']}")
            
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.metric("⚡ 전력사용량 예측", f"{selected_detail['전력사용량_예측']:.2f} kWh")
            with detail_col2:
                st.metric("🌱 탄소배출량 예측", f"{selected_detail['탄소배출량_kg']:.2f} kgCO2")
            with detail_col3:
                st.metric("💰 전기요금 예측", f"{selected_detail['전기요금_예측']:,.0f} 원")
            
            st.divider()
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"**작업유형:** {selected_detail['작업유형']}")
                st.info(f"**작업휴무:** {selected_detail['작업휴무']}")
            with info_col2:
                st.info(f"**지상역률:** {selected_detail['지상역률(%)']:.2f}%")
                st.info(f"**진상역률:** {selected_detail['진상역률(%)']:.2f}%")
            
            # 팝업 닫기 버튼
            if st.button("닫기", type="primary", use_container_width=True):
                ss.popup_open = False
                st.rerun()
        
        show_detail()
    else:
        # 팝업이 닫혔음을 표시
        ss.popup_open = False
    
    # 진행 상태 표시
    st.divider()
    progress_text = f"진행 상황: {ss.step}/{len(ss.full_data)} ({ss.step/len(ss.full_data)*100:.1f}%)"
    st.progress(ss.step / len(ss.full_data), text=progress_text)

else:
    st.info("▶ '재생' 버튼을 눌러 모니터링을 시작하세요.")
    st.caption(f"📍 데이터가 {update_interval}초마다 자동으로 업데이트됩니다.")

# ---- 자동 반복 (슬라이더로 조정 가능한 간격, 팝업 열려있을 때는 정지) ----
if ss.running and ss.step < len(ss.full_data) and not ss.popup_open:
    time.sleep(update_interval)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
elif ss.running and ss.step >= len(ss.full_data):
    st.success("✅ 모든 데이터 처리 완료!")
    ss.running = False