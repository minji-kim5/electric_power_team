import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================================================================================
# 설정 및 상수 정의
# ================================================================================
st.set_page_config(page_title="12월 예측 - 컨트롤", layout="wide")

# 차트 관련 상수
CHART_RECENT_POINTS = 30  # 차트에 표시할 최근 데이터 포인트 수
CHART_HEIGHT = 450
DATA_LOG_ROWS = 5

# 전력 관련 상수
BASE_PEAK_POWER = 157.18  # 기준 피크 전력 (kWh)
POWER_FACTOR_LAGGING = 90  # 지상역률 기준 (%)
POWER_FACTOR_LEADING = 95  # 진상역률 기준 (%)
POWER_FACTOR_THRESHOLD_START = 9.25  # 지상역률 적용 시작 시간
POWER_FACTOR_THRESHOLD_END = 22.0    # 지상역률 적용 종료 시간

# 일일 전력 기준 (가동일/휴무일)
DAILY_POWER_LIMITS = {
    '가동': {'threshold': 4270, 'max': 6500, 'ticks': [0, 1300, 2600, 4270, 5200, 6500]},
    '휴무': {'threshold': 360, 'max': 550, 'ticks': [0, 110, 220, 360, 440, 550]}
}

# 시간대별 부하 정의 (가동일)
LOAD_SEGMENTS_WORKING = [
    {'start': 0, 'end': 9, 'load': '경부하', 'color': '#90EE90'},
    {'start': 9, 'end': 10, 'load': '중간부하', 'color': '#FFD700'},
    {'start': 10, 'end': 12, 'load': '최대부하', 'color': '#FF6B6B'},
    {'start': 12, 'end': 17, 'load': '중간부하', 'color': '#FFD700'},
    {'start': 17, 'end': 20, 'load': '최대부하', 'color': '#FF6B6B'},
    {'start': 20, 'end': 22, 'load': '중간부하', 'color': '#FFD700'},
    {'start': 22, 'end': 23, 'load': '최대부하', 'color': '#FF6B6B'},
    {'start': 23, 'end': 24, 'load': '경부하', 'color': '#90EE90'}
]

# 시간대별 부하 정의 (휴무일)
LOAD_SEGMENTS_HOLIDAY = [
    {'start': 0, 'end': 24, 'load': '경부하', 'color': '#90EE90'}
]

# 차트 색상
CHART_COLORS = {
    'power': '#1f77b4',
    'lagging_pf': '#FF6B6B',
    'leading_pf': '#4ECDC4',
    'gauge_working': '#1f77b4',
    'gauge_holiday': '#90CAF9'
}

# ================================================================================
# 커스텀 CSS
# ================================================================================
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

# ================================================================================
# 유틸리티 함수
# ================================================================================
@st.cache_data
def load_data():
    """CSV 데이터 로드 및 전처리"""
    df = pd.read_csv('../data/test6.csv')
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    return df


def fix_midnight_dates(df_chart):
    """자정(00:00) 데이터의 날짜를 다음날로 수정"""
    mask = df_chart['측정일시'].dt.time == pd.Timestamp('00:00:00').time()
    df_chart.loc[mask, '측정일시'] = df_chart.loc[mask, '측정일시'] + timedelta(days=1)
    return df_chart


def create_metric_card(label, value, delta, card_class):
    """커스텀 메트릭 카드 HTML 생성"""
    return f"""
    <div class="metric-card {card_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value"><strong>{value}</strong></div>
        <div class="metric-delta">{delta}</div>
    </div>
    """


# ================================================================================
# 차트 생성 함수
# ================================================================================
def create_power_usage_chart(df, show_peak_line):
    """전력사용량 라인차트 생성"""
    df_chart = df.tail(CHART_RECENT_POINTS).copy()
    df_chart = fix_midnight_dates(df_chart)
    
    # 피크 전력 계산
    current_max = df['전력사용량_예측'].max()
    peak_power = max(BASE_PEAK_POWER, current_max)
    
    # 차트 생성
    fig = px.line(
        df_chart, 
        x='측정일시', 
        y='전력사용량_예측',
        labels={'측정일시': '시간', '전력사용량_예측': '전력사용량 (kWh)'},
        template='plotly_white'
    )
    
    fig.update_traces(
        line_color=CHART_COLORS['power'], 
        line_width=2,
        mode='lines+markers',
        marker=dict(size=6, color=CHART_COLORS['power'])
    )
    
    # 피크 전력 기준선
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
    y_max = peak_power * 1.1 if show_peak_line else df_chart['전력사용량_예측'].max() * 1.15
    
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title='측정일시',
        yaxis_title='전력사용량 (kWh)',
        yaxis_range=[0, y_max],
        hovermode='x unified',
        uirevision='power_chart',
        transition={'duration': 0}
    )
    
    return fig


def create_daily_power_gauge(df, latest):
    """당일 전력사용량 게이지 차트 생성"""
    current_date = latest['측정일시'].date()
    date_str = current_date.strftime('%Y년 %m월 %d일')
    
    # 당일 데이터 필터링
    df_today = df[df['측정일시'].dt.date == current_date]
    total_power = df_today['전력사용량_예측'].sum()
    
    # 작업휴무 상태에 따른 설정
    current_status = latest['작업휴무']
    config = DAILY_POWER_LIMITS[current_status]
    status_text = "가동일" if current_status == '가동' else "휴무일"
    bar_color = CHART_COLORS['gauge_working'] if current_status == '가동' else CHART_COLORS['gauge_holiday']
    
    # 게이지 차트 생성
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
                'range': [None, config['max']], 
                'tickwidth': 1, 
                'tickcolor': "darkblue",
                'tickmode': 'array',
                'tickvals': config['ticks'],
                'ticktext': [str(int(v)) for v in config['ticks']],
                'tickfont': {'size': 11}
            },
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, config['threshold']], 'color': '#E8F5E9'},
                {'range': [config['threshold'], config['max']], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': config['threshold']
            }
        }
    ))
    
    gauge_fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=30, r=30, t=80, b=30),
        uirevision='gauge_chart',
        transition={'duration': 0}
    )
    
    return gauge_fig


def create_power_factor_chart(df, show_pf_line, latest):
    """역률 추이 차트 생성"""
    df_chart_pf = df.tail(CHART_RECENT_POINTS).copy()
    df_chart_pf = fix_midnight_dates(df_chart_pf)
    
    # 데이터 재구성 (wide to long)
    df_pf_long = pd.melt(
        df_chart_pf,
        id_vars=['측정일시'],
        value_vars=['지상역률(%)', '진상역률(%)'],
        var_name='역률 유형',
        value_name='역률 값'
    )
    
    # 차트 생성
    fig_pf = px.line(
        df_pf_long,
        x='측정일시',
        y='역률 값',
        color='역률 유형',
        labels={'측정일시': '시간', '역률 값': '역률 (%)'},
        template='plotly_white',
        color_discrete_map={
            '지상역률(%)': CHART_COLORS['lagging_pf'], 
            '진상역률(%)': CHART_COLORS['leading_pf']
        }
    )
    
    fig_pf.update_traces(
        line_width=2,
        mode='lines+markers',
        marker=dict(size=5)
    )
    
    # 기준역률선 추가
    if show_pf_line:
        time_val = latest['측정일시']
        time_decimal = time_val.hour + time_val.minute / 60.0
        
        if POWER_FACTOR_THRESHOLD_START <= time_decimal < POWER_FACTOR_THRESHOLD_END:
            fig_pf.add_hline(
                y=POWER_FACTOR_LAGGING,
                line_dash="dash",
                line_color=CHART_COLORS['lagging_pf'],
                line_width=2,
                annotation_text=f"지상역률 기준 {POWER_FACTOR_LAGGING}%",
                annotation_position="right"
            )
        else:
            fig_pf.add_hline(
                y=POWER_FACTOR_LEADING,
                line_dash="dash",
                line_color=CHART_COLORS['leading_pf'],
                line_width=2,
                annotation_text=f"진상역률 기준 {POWER_FACTOR_LEADING}%",
                annotation_position="right"
            )
    
    fig_pf.update_layout(
        height=CHART_HEIGHT,
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
        ),
        uirevision='pf_chart',
        transition={'duration': 0}
    )
    
    return fig_pf


def create_load_clock_chart(latest):
    """시간대별 부하 원형 차트 생성"""
    current_status = latest['작업휴무']
    current_time = latest['측정일시'].time()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # 가동일/휴무일에 따른 부하 설정
    load_segments = LOAD_SEGMENTS_WORKING if current_status == '가동' else LOAD_SEGMENTS_HOLIDAY
    status_display = '가동일' if current_status == '가동' else '휴무일'
    
    fig_load = go.Figure()
    load_types = {'경부하': True, '중간부하': True, '최대부하': True}
    
    # 부하 세그먼트 추가
    for segment in load_segments:
        start_hour = segment['start']
        end_hour = segment['end']
        duration = end_hour - start_hour
        center_hour = (start_hour + end_hour) / 2
        theta = center_hour * 15
        width = duration * 15
        
        show_legend = load_types.get(segment['load'], False)
        if show_legend:
            load_types[segment['load']] = False
        
        fig_load.add_trace(go.Barpolar(
            r=[1],
            theta=[theta],
            width=[width],
            base=0.8,
            marker=dict(
                color=segment['color'],
                line=dict(color='white', width=2)
            ),
            name=segment['load'],
            showlegend=show_legend,
            hovertemplate=f"{start_hour:02d}:00-{end_hour:02d}:00<br>{segment['load']}<extra></extra>"
        ))
    
    # 시간 표기
    for hour in range(24):
        theta = hour * 15
        fig_load.add_trace(go.Scatterpolar(
            r=[1.35],
            theta=[theta],
            mode='text',
            text=[f'<b>{hour}</b>'],
            textfont=dict(size=11, color='#333333'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 시계바늘
    time_in_hours = current_hour + current_minute / 60.0
    needle_theta = time_in_hours * 15
    
    fig_load.add_trace(go.Scatterpolar(
        r=[0, 0.85],
        theta=[needle_theta, needle_theta],
        mode='lines',
        line=dict(color='gray', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
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
            font=dict(size=11)
        ),
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=80, b=40),
        uirevision='load_chart',
        transition={'duration': 0}
    )
    
    return fig_load


# ================================================================================
# 세션 상태 초기화
# ================================================================================
def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        "running": False,
        "step": 0,
        "accumulated_data": pd.DataFrame(),
        "data_loaded": False,
        "prev_show_peak": False,
        "prev_show_pf": False,
        "table_key": 0
    }
    
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ================================================================================
# 메인 앱
# ================================================================================
st.title("실시간 전력 및 전기요금 모니터링")

# 세션 상태 초기화
initialize_session_state()
ss = st.session_state

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

# 데이터 출력 간격 조정
update_interval = st.sidebar.slider(
    "데이터 출력 간격 (초)",
    min_value=0.1,
    max_value=4.0,
    value=2.0,
    step=0.1,
    key="update_interval"
)

# 차트 옵션
with st.sidebar.expander("실시간 전력사용량 추이"):
    show_peak_line = st.checkbox("피크전력선 표시", value=False, key="show_peak")

with st.sidebar.expander("실시간 역률 추이"):
    show_pf_line = st.checkbox("기준역률선 표시", value=False, key="show_pf")

# 체크박스 상태 변화 감지
if ss.prev_show_peak != show_peak_line or ss.prev_show_pf != show_pf_line:
    ss.table_key += 1
    ss.prev_show_peak = show_peak_line
    ss.prev_show_pf = show_pf_line

# 컨트롤 버튼 처리
if start:
    ss.running = True
if stop:
    ss.running = False
if reset:
    ss.running = False
    ss.step = 0
    ss.accumulated_data = pd.DataFrame()
    import os
    try:
        if os.path.exists('data_dash\\december_streaming.csv'):
            os.remove('data_dash\\december_streaming.csv')
    except:
        pass
    
    st.rerun()    

# ---- 데이터 누적 로직 ----
if ss.running and ss.step < len(ss.full_data):
    current_row = ss.full_data.iloc[ss.step:ss.step+1]
    ss.accumulated_data = pd.concat([ss.accumulated_data, current_row], ignore_index=True)
    ss.step += 1
    ss.accumulated_data.to_csv('data_dash\\december_streaming.csv', index=False, encoding='utf-8-sig')
# ================================================================================
# 메인 대시보드
# ================================================================================
if len(ss.accumulated_data) > 0:
    df = ss.accumulated_data.copy()
    df['탄소배출량_kg'] = df['탄소배출량_예측'] * 1000
    latest = df.iloc[-1]
    
    # === KPI 카드 ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "📊 누적 전력사용량",
            f"{df['전력사용량_예측'].sum():.2f} kWh",
            f"+{latest['전력사용량_예측']:.2f} kWh",
            "metric-card-blue"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "💰 누적 전기요금",
            f"{df['전기요금_예측'].sum():,.0f} 원",
            f"+{latest['전기요금_예측']:,.0f} 원",
            "metric-card-green"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "🌱 누적 탄소배출량",
            f"{df['탄소배출량_kg'].sum():.2f} kgCO2",
            f"+{latest['탄소배출량_kg']:.2f} kgCO2",
            "metric-card-orange"
        ), unsafe_allow_html=True)
    
    with col4:
        status_emoji = "🟢" if latest['작업휴무'] == '가동' else "🔴"
        load_text = latest['작업유형'].replace('_', ' ')
        st.markdown(create_metric_card(
            "⚙️ 운영 상태",
            f"{status_emoji} {latest['작업휴무']}",
            load_text,
            "metric-card-purple"
        ), unsafe_allow_html=True)
    
    st.divider()
    
    # === 전력사용량 섹션 ===
    st.subheader("실시간 전력사용량 추이 및 당일 전력사용량")
    chart_col, gauge_col = st.columns([3, 1])
    
    with chart_col:
        fig_power = create_power_usage_chart(df, show_peak_line)
        st.plotly_chart(fig_power, use_container_width=True, key="power_chart", config={'displayModeBar': False})
    
    with gauge_col:
        fig_gauge = create_daily_power_gauge(df, latest)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart", config={'displayModeBar': False})
    
    st.divider()
    
    # === 역률 섹션 ===
    st.subheader("실시간 역률 추이 및 시간대별 부하")
    pf_col, load_col = st.columns([3, 1])
    
    with pf_col:
        fig_pf = create_power_factor_chart(df, show_pf_line, latest)
        st.plotly_chart(fig_pf, use_container_width=True, key="pf_chart", config={'displayModeBar': False})
    
    with load_col:
        fig_load = create_load_clock_chart(latest)
        st.plotly_chart(fig_load, use_container_width=True, key="load_chart", config={'displayModeBar': False})
    
    st.divider()
    
    # === 데이터 로그 ===
    st.subheader("최근 데이터 로그")
    
    recent_data = df.tail(DATA_LOG_ROWS)[['측정일시', '작업유형', '작업휴무', '지상역률(%)', '진상역률(%)']].copy()
    recent_data_full = df.tail(DATA_LOG_ROWS).copy().reset_index(drop=True)
    recent_data = recent_data.reset_index(drop=True)
    
    event = st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True,
        height=220,
        selection_mode="single-row",
        on_select="rerun",
        key=f"data_table_{ss.table_key}"
    )
    
    # 행 선택 시 상세 정보 팝업
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_detail = recent_data_full.iloc[selected_idx]
        
        @st.dialog("상세 정보")
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
        
        show_detail()
    
    # 진행 상태
    st.divider()
    progress_text = f"진행 상황: {ss.step}/{len(ss.full_data)} ({ss.step/len(ss.full_data)*100:.1f}%)"
    st.progress(ss.step / len(ss.full_data), text=progress_text)

else:
    st.info("▶ '재생' 버튼을 눌러 모니터링을 시작하세요.")
    st.caption(f"📍 데이터가 {update_interval}초마다 자동으로 업데이트됩니다.")

# ---- 자동 반복 ----
if ss.running and ss.step < len(ss.full_data):
    time.sleep(update_interval)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
elif ss.running and ss.step >= len(ss.full_data):
    st.success("✅ 모든 데이터 처리 완료!")
    ss.running = False