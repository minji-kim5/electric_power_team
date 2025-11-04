import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================================================================================
# 설정 및 상수 정의
# ================================================================================
st.set_page_config(
    page_title="전력 모니터링 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 차트 관련 상수
CHART_RECENT_POINTS = 30
CHART_HEIGHT = 450
DATA_LOG_ROWS = 5

# 전력 관련 상수
BASE_PEAK_POWER = 157.18
POWER_FACTOR_LAGGING = 90
POWER_FACTOR_LEADING = 95
POWER_FACTOR_THRESHOLD_START = 9.25
POWER_FACTOR_THRESHOLD_END = 22.0

# 일일 전력 기준
DAILY_POWER_LIMITS = {
    '가동': {'threshold': 4270, 'max': 6500, 'ticks': [0, 1300, 2600, 4270, 5200, 6500]},
    '휴무': {'threshold': 360, 'max': 550, 'ticks': [0, 110, 220, 360, 440, 550]}
}

# 시간대별 부하 정의
LOAD_SEGMENTS_WORKING = [
    {'start': 0, 'end': 9, 'load': '경부하', 'color': '#4CAF50'},
    {'start': 9, 'end': 10, 'load': '중간부하', 'color': '#FFC107'},
    {'start': 10, 'end': 12, 'load': '최대부하', 'color': '#EF5350'},
    {'start': 12, 'end': 17, 'load': '중간부하', 'color': '#FFC107'},
    {'start': 17, 'end': 20, 'load': '최대부하', 'color': '#EF5350'},
    {'start': 20, 'end': 22, 'load': '중간부하', 'color': '#FFC107'},
    {'start': 22, 'end': 23, 'load': '최대부하', 'color': '#EF5350'},
    {'start': 23, 'end': 24, 'load': '경부하', 'color': '#4CAF50'}
]

LOAD_SEGMENTS_HOLIDAY = [
    {'start': 0, 'end': 24, 'load': '경부하', 'color': '#4CAF50'}
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
# 최적화된 CSS
# ================================================================================
st.markdown("""
<style>
    /* 전역 설정 */
    .main {
        background-color: #F5F7FA;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* 제목 최적화 */
    h1 {
        color: #2C3E50;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    
    h2, h3 {
        color: #34495E;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* KPI 카드 - 기존 그라데이션 유지하되 최적화 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
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
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        opacity: 0.85;
        font-weight: 500;
    }
    
    /* 차트 컨테이너 */
    .stPlotlyChart {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* 사이드바 최적화 */
    [data-testid="stSidebar"] {
        background: #f1f2f6;
        border-right: 1px solid #e0e6ed;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        color: #2C3E50;
        font-weight: 700;
        padding: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] h2 {
        font-size: 1rem;
        color: #34495E;
        margin-top: 1rem;
        font-weight: 600;
    }
    
    /* 버튼 최적화 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        font-size: 0.95rem;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:not([kind="primary"]) {
        background-color: white;
        border: 2px solid #e0e6ed;
        color: #2C3E50;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        background-color: #f8f9fa;
        border-color: #cbd5e0;
        transform: translateY(-1px);
    }
    
    /* 슬라이더 최적화 */
    .stSlider {
        padding: 0.5rem 0;
    }
    
    /* 체크박스 최적화 */
    .stCheckbox {
        padding: 0.3rem 0;
    }
    
    .stCheckbox label {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Expander 최적화 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border: 1px solid #e0e6ed;
        border-radius: 6px;
        font-weight: 600;
        color: #2C3E50;
        font-size: 0.9rem;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }
    
    /* 데이터프레임 최적화 */
    [data-testid="stDataFrame"] {
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* 구분선 */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e6ed;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Info 박스 */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* 캡션 */
    .stCaption {
        color: #7F8C8D;
        font-size: 0.85rem;
    }
    
    /* 다이얼로그 최적화 */
    [data-testid="stDialog"] {
        border-radius: 12px;
    }
    
    /* 성능 최적화 - 애니메이션 최소화 */
    .js-plotly-plot {
        transition: none !important;
    }
    
    /* 메트릭 최적화 */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# 유틸리티 함수
# ================================================================================
@st.cache_data(ttl=3600)
def load_data():
    """CSV 데이터 로드 및 전처리 (캐싱 최적화)"""
    df = pd.read_csv('../data/test6.csv')
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    return df


def fix_midnight_dates(df_chart):
    """자정(00:00) 데이터의 날짜를 다음날로 수정"""
    mask = df_chart['측정일시'].dt.time == pd.Timestamp('00:00:00').time()
    if mask.any():
        df_chart.loc[mask, '측정일시'] = df_chart.loc[mask, '측정일시'] + timedelta(days=1)
    return df_chart


def create_metric_card(label, value, delta, card_class):
    """최적화된 메트릭 카드 HTML 생성"""
    return f"""
    <div class="metric-card {card_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value"><strong>{value}</strong></div>
        <div class="metric-delta">{delta}</div>
    </div>
    """


# ================================================================================
# 최적화된 차트 생성 함수
# ================================================================================
def create_power_usage_chart(df, show_peak_line):
    """전력사용량 라인차트 - 최적화"""
    df_chart = df.tail(CHART_RECENT_POINTS).copy()
    df_chart = fix_midnight_dates(df_chart)
    
    current_max = df['전력사용량_예측'].max()
    peak_power = max(BASE_PEAK_POWER, current_max)
    
    # 최적화된 차트 생성
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_chart['측정일시'],
        y=df_chart['전력사용량_예측'],
        mode='lines+markers',
        name='전력사용량',
        line=dict(color=CHART_COLORS['power'], width=2.5, shape='spline'),
        marker=dict(size=6, color=CHART_COLORS['power'], symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
        hovertemplate='<b>%{x|%m/%d %H:%M}</b><br>전력: %{y:.2f} kWh<extra></extra>'
    ))
    
    if show_peak_line:
        fig.add_hline(
            y=peak_power,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"피크: {peak_power:.2f} kWh",
            annotation_position="top right",
            annotation=dict(font_size=11, font_color="red", bgcolor="rgba(255,255,255,0.8)")
        )
    
    y_max = peak_power * 1.1 if show_peak_line else df_chart['전력사용량_예측'].max() * 1.15
    
    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            title='측정일시',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#e0e6ed'
        ),
        yaxis=dict(
            title='전력사용량 (kWh)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#e0e6ed',
            range=[0, y_max]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=False,
        uirevision='power_chart',
        transition={'duration': 0}
    )
    
    return fig


def create_daily_power_gauge(df, latest):
    """당일 전력사용량 게이지 - 최적화 (깜빡임 방지)"""
    current_date = latest['측정일시'].date()
    date_str = current_date.strftime('%Y년 %m월 %d일')
    
    df_today = df[df['측정일시'].dt.date == current_date]
    total_power = df_today['전력사용량_예측'].sum()
    
    current_status = latest['작업휴무']
    config = DAILY_POWER_LIMITS[current_status]
    status_text = "가동일" if current_status == '가동' else "휴무일"
    bar_color = CHART_COLORS['gauge_working'] if current_status == '가동' else CHART_COLORS['gauge_holiday']
    
    # 깜빡임 방지: 날짜별 uirevision
    ui_revision = f"gauge_{current_date}_{current_status}"
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_power,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>당일 전력사용량<br>({status_text}, {date_str})</b>",
            'font': {'size': 14, 'color': '#2C3E50', 'family': 'Arial, sans-serif'}
        },
        number={
            'suffix': ' kWh',
            'font': {'size': 40, 'color': '#2C3E50'},
            'valueformat': '.0f'
        },
        gauge={
            'axis': {
                'range': [None, config['max']],
                'tickwidth': 1,
                'tickcolor': "#cbd5e0",
                'tickmode': 'array',
                'tickvals': config['ticks'],
                'ticktext': [str(int(v)) for v in config['ticks']],
                'tickfont': {'size': 10, 'color': '#7F8C8D'}
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, config['threshold']], 'color': '#E8F5E9'},
                {'range': [config['threshold'], config['max']], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': config['threshold']
            }
        }
    ))
    
    gauge_fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='white',
        uirevision=ui_revision,
        transition={'duration': 0}
    )
    
    return gauge_fig


def create_power_factor_chart(df, show_pf_line, latest):
    """역률 추이 차트 - 최적화"""
    df_chart_pf = df.tail(CHART_RECENT_POINTS).copy()
    df_chart_pf = fix_midnight_dates(df_chart_pf)
    
    fig = go.Figure()
    
    # 지상역률
    fig.add_trace(go.Scatter(
        x=df_chart_pf['측정일시'],
        y=df_chart_pf['지상역률(%)'],
        mode='lines+markers',
        name='지상역률',
        line=dict(color=CHART_COLORS['lagging_pf'], width=2.5, shape='spline'),
        marker=dict(size=5, color=CHART_COLORS['lagging_pf']),
        hovertemplate='<b>%{x|%m/%d %H:%M}</b><br>지상역률: %{y:.2f}%<extra></extra>'
    ))
    
    # 진상역률
    fig.add_trace(go.Scatter(
        x=df_chart_pf['측정일시'],
        y=df_chart_pf['진상역률(%)'],
        mode='lines+markers',
        name='진상역률',
        line=dict(color=CHART_COLORS['leading_pf'], width=2.5, shape='spline'),
        marker=dict(size=5, color=CHART_COLORS['leading_pf']),
        hovertemplate='<b>%{x|%m/%d %H:%M}</b><br>진상역률: %{y:.2f}%<extra></extra>'
    ))
    
    if show_pf_line:
        time_val = latest['측정일시']
        time_decimal = time_val.hour + time_val.minute / 60.0
        
        if POWER_FACTOR_THRESHOLD_START <= time_decimal < POWER_FACTOR_THRESHOLD_END:
            fig.add_hline(
                y=POWER_FACTOR_LAGGING,
                line_dash="dash",
                line_color=CHART_COLORS['lagging_pf'],
                line_width=2,
                annotation_text=f"기준: {POWER_FACTOR_LAGGING}%",
                annotation_position="top right",
                annotation=dict(font_size=11, bgcolor="rgba(255,255,255,0.8)")
            )
        else:
            fig.add_hline(
                y=POWER_FACTOR_LEADING,
                line_dash="dash",
                line_color=CHART_COLORS['leading_pf'],
                line_width=2,
                annotation_text=f"기준: {POWER_FACTOR_LEADING}%",
                annotation_position="top right",
                annotation=dict(font_size=11, bgcolor="rgba(255,255,255,0.8)")
            )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            title='측정일시',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#e0e6ed'
        ),
        yaxis=dict(
            title='역률 (%)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#e0e6ed',
            range=[0, 105]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e0e6ed',
            borderwidth=1
        ),
        uirevision='pf_chart',
        transition={'duration': 0}
    )
    
    return fig


def create_load_clock_chart(latest):
    """시간대별 부하 차트 - 최적화 (깜빡임 방지)"""
    current_status = latest['작업휴무']
    current_time = latest['측정일시'].time()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    load_segments = LOAD_SEGMENTS_WORKING if current_status == '가동' else LOAD_SEGMENTS_HOLIDAY
    status_display = '가동일' if current_status == '가동' else '휴무일'
    
    # 깜빡임 방지: 시간별 uirevision
    ui_revision = f"load_{current_status}_{current_hour}"
    
    fig_load = go.Figure()
    load_types = {'경부하': True, '중간부하': True, '최대부하': True}
    
    # 부하 세그먼트
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
            base=0.75,
            marker=dict(
                color=segment['color'],
                line=dict(color='white', width=2)
            ),
            name=segment['load'],
            showlegend=show_legend,
            hovertemplate=f"{start_hour:02d}:00~{end_hour:02d}:00<br>{segment['load']}<extra></extra>"
        ))
    
    # 시간 라벨 최적화 (한 번에 추가)
    time_labels_r = [1.3] * 24
    time_labels_theta = [hour * 15 for hour in range(24)]
    time_labels_text = [f'<b>{hour}</b>' for hour in range(24)]
    
    fig_load.add_trace(go.Scatterpolar(
        r=time_labels_r,
        theta=time_labels_theta,
        mode='text',
        text=time_labels_text,
        textfont=dict(size=10, color='#2C3E50', family='Arial, sans-serif'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 시계바늘
    time_in_hours = current_hour + current_minute / 60.0
    needle_theta = time_in_hours * 15
    
    fig_load.add_trace(go.Scatterpolar(
        r=[0, 0.8],
        theta=[needle_theta, needle_theta],
        mode='lines',
        line=dict(color='#2C3E50', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_load.add_trace(go.Scatterpolar(
        r=[0],
        theta=[0],
        mode='markers',
        marker=dict(size=8, color='#2C3E50'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_load.update_layout(
        title=dict(
            text=f"<b>시간대별 부하 ({status_display})</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=13, color='#2C3E50')
        ),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.5]),
            angularaxis=dict(visible=False, direction='clockwise', rotation=90),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e0e6ed',
            borderwidth=1
        ),
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=70, b=30),
        paper_bgcolor='white',
        uirevision=ui_revision,
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
st.caption("공장 전력 사용량 실시간 대시보드")

# 세션 상태 초기화
initialize_session_state()
ss = st.session_state

# 데이터 로드
if not ss.data_loaded:
    with st.spinner('데이터 로딩 중...'):
        ss.full_data = load_data()
        ss.data_loaded = True

# ---- 사이드바 ----
with st.sidebar:
    st.header("제어판")
    
    # 컨트롤 버튼
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("▶ 재생", type="primary", use_container_width=True)
    with col2:
        stop = st.button("⏸ 정지", use_container_width=True)
    
    reset = st.button("⟲ 리셋", use_container_width=True)
    
    st.divider()
    
    # 재생 설정
    st.subheader("재생 설정")
    update_interval = st.slider(
        "업데이트 간격 (초)",
        min_value=0.1,
        max_value=4.0,
        value=2.0,
        step=0.1,
        key="update_interval",
        help="데이터 업데이트 주기를 설정합니다"
    )
    
    st.divider()
    
    # 차트 옵션
    st.subheader("차트 옵션")
    
    with st.expander("실시간 전력사용량 추이", expanded=False):
        show_peak_line = st.checkbox(
            "피크전력선 표시",
            value=False,
            key="show_peak",
            help="한 해동안의 전력 사용량 중 가장 높은 값으로 산정된" \
                 "산정된 피크전력은 전기 요금 중 기본 요금의 산정에 " \
                 "적용, 피크전력 산출 대상 시간은 24시간 중 경부하" \
                 "시간을 제외한 시간이 대상이 된다."
        )
    
    with st.expander("실시간 역률 추이", expanded=False):
        show_pf_line = st.checkbox(
            "기준역률선 표시",
            value=False,
            key="show_pf",
            help="한전 규정에 따라 09시부터 22시까지 지상역률의 평균이 " \
                 "90% 미달인 경우 역률 60%까지 매 1%당 기본요금의 " \
                 "0.2% 추가하고 반대로 평균역률이 90%를 초과하는 " \
                 "경우 역률 95%까지 매 1%당 기본요금의 0.2% 감액 " \
                 "22시부터 09시까지 진상역률의 평균이 95% 미달인 " \
                 "경우 역률 60%까지 매 1%당 기본요금의 0.2% 추가한다."
        )
    
    # 상태 표시
    st.divider()
    if ss.running:
        st.success("🟢 **실행 중**")
    else:
        st.info("⚪ **대기 중**")

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
    try:
        ss.accumulated_data.to_csv('data_dash\\december_streaming.csv', index=False, encoding='utf-8-sig')
    except:
        pass

# ================================================================================
# 메인 대시보드
# ================================================================================
if len(ss.accumulated_data) > 0:
    df = ss.accumulated_data.copy()
    df['탄소배출량_kg'] = df['탄소배출량_예측'] * 1000
    latest = df.iloc[-1]
    
    # === KPI 카드 ===
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
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
    
    chart_col, gauge_col = st.columns([3, 1], gap="medium")
    
    with chart_col:
        fig_power = create_power_usage_chart(df, show_peak_line)
        st.plotly_chart(fig_power, use_container_width=True, key="power_chart", config={'displayModeBar': False})
    
    with gauge_col:
        fig_gauge = create_daily_power_gauge(df, latest)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart", config={'displayModeBar': False})
    
    st.divider()
    
    # === 역률 섹션 ===
    st.subheader("실시간 역률 추이 및 시간대별 부하")
    
    pf_col, load_col = st.columns([3, 1], gap="medium")
    
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
    
    # 행 선택 시 상세 정보
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_detail = recent_data_full.iloc[selected_idx]
        
        @st.dialog("상세 정보", width="large")
        def show_detail():
            st.markdown(f"### {selected_detail['측정일시']}")
            st.markdown("---")
            
            # 메트릭 섹션
            st.markdown("#### 주요 지표")
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.metric(
                    "⚡ 전력사용량", 
                    f"{selected_detail['전력사용량_예측']:.2f} kWh",
                    help="해당 시간의 전력 사용량"
                )
            with detail_col2:
                st.metric(
                    "🌱 탄소배출량", 
                    f"{selected_detail['탄소배출량_kg']:.2f} kgCO2",
                    help="해당 시간의 탄소 배출량"
                )
            with detail_col3:
                st.metric(
                    "💰 전기요금", 
                    f"{selected_detail['전기요금_예측']:,.0f} 원",
                    help="해당 시간의 전기 요금"
                )
            
            st.markdown("---")
            
            # 운영 정보 섹션
            st.markdown("#### ⚙️ 운영 정보")
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
    st.write(f"진행 상황: {ss.step}/{len(ss.full_data)} ({ss.step/len(ss.full_data)*100:.1f}%)")
    st.progress(ss.step / len(ss.full_data))


else:
    # 초기 화면
    st.info("**사이드바에서 '재생' 버튼을 눌러 모니터링을 시작하세요.**")
    st.caption(f"데이터가 {ss.get('update_interval', 2.0)}초마다 자동으로 업데이트됩니다.")
    
    # 가이드
    with st.expander("사용 가이드", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 제어 방법
            - **▶ 재생**: 실시간 데이터 스트리밍 시작
            - **⏸ 정지**: 현재 상태에서 일시정지
            - **⟲ 리셋**: 처음부터 다시 시작
            """)
        
        with col2:
            st.markdown("""
            #### 주요 기능
            - **실시간 KPI**: 전력, 요금, 탄소배출량 모니터링
            - **트렌드 분석**: 시간대별 사용 패턴 확인
            - **역률 관리**: 기준 역률 대비 현재 상태
            - **데이터 로그**: 최근 데이터 상세 확인
            """)

# ---- 자동 반복 ----
if ss.running and ss.step < len(ss.full_data):
    time.sleep(update_interval)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
elif ss.running and ss.step >= len(ss.full_data):
    st.success("✅ 모든 데이터 처리 완료!")
    st.balloons()
    ss.running = False