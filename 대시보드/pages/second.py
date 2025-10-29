import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(page_title="전력 데이터 분석", page_icon="📊", layout="wide")

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\USER\Desktop\electric_power_-team\data\train_df.csv")
    # df = pd.read_csv("data\\train_df.csv") # 데이터 불러오는 것 수정 (각자)
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['month'] = df['측정일시'].dt.month
    df['year'] = df['측정일시'].dt.year
    df['day'] = df['측정일시'].dt.day
    df['hour'] = df['측정일시'].dt.hour
    df['date'] = df['측정일시'].dt.date
    return df

# 데이터 로드
df = load_data()

# 페이지 제목
st.title("📊 LS ELECTRIC 청주 공장 전력 사용 현황")
st.divider()

# ===== 사이드바 필터 =====
st.sidebar.header("🔍 필터 선택")
mode = st.sidebar.radio("보기 방식", ["월별", "기간"])

# 필터링
if mode == "월별":
    month_options = ["전체"] + list(range(1, 12))
    selected_month = st.sidebar.selectbox(
        "분석할 월을 선택하세요",
        options=month_options,
        format_func=lambda x: "전체" if x == "전체" else f"{x}월"
    )
    
    if selected_month == "전체":
        filtered_df = df.copy()
        label = "전체(1~11월)"
    else:
        filtered_df = df[df['month'] == selected_month].copy()
        label = f"{selected_month}월"
else:
    min_date = df['측정일시'].min().date()
    max_date = df['측정일시'].max().date()
    
    date_range = st.sidebar.date_input(
        "기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        label = f"{start_date} ~ {end_date}"
    else:
        filtered_df = df.copy()
        label = "전체"

# ===== 주요 지표 =====
st.markdown(f"## 📅 {label} 주요 지표")
st.markdown(
    f"**데이터 기간**: {filtered_df['측정일시'].min().strftime('%Y-%m-%d')} ~ "
    f"{filtered_df['측정일시'].max().strftime('%Y-%m-%d')}"
)

# KPI 계산
total_power = filtered_df['전력사용량(kWh)'].sum()
total_cost = filtered_df['전기요금(원)'].sum()
total_carbon = filtered_df['탄소배출량(tCO2)'].sum()
total_lag = filtered_df['지상무효전력량(kVarh)'].sum()
total_lead = filtered_df['진상무효전력량(kVarh)'].sum()

# KPI 스타일
st.markdown("""
<style>
.kpi-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-title {
    font-size: 16px;
    color: #666;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 5px;
}
.kpi-unit {
    font-size: 14px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# KPI 카드
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">총 전력사용량</div>
        <div class="kpi-value">{total_power:,.0f}</div>
        <div class="kpi-unit">kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">총 전기요금</div>
        <div class="kpi-value">{total_cost:,.0f}</div>
        <div class="kpi-unit">원</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">총 탄소배출량</div>
        <div class="kpi-value">{total_carbon:,.2f}</div>
        <div class="kpi-unit">tCO2</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">지상무효전력량</div>
        <div class="kpi-value">{total_lag:,.1f}</div>
        <div class="kpi-unit">kVarh</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">진상무효전력량</div>
        <div class="kpi-value">{total_lead:,.1f}</div>
        <div class="kpi-unit">kVarh</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ===== 메인 그래프 =====
if mode == "월별":
    st.subheader("📊 월별 전력사용량 + 월 평균 전기요금")
    
    # 월별 집계
    monthly = df.groupby('month').agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'mean'
    }).reset_index()
    monthly = monthly[monthly['month'] <= 11]
    monthly['label'] = monthly['month'].apply(lambda x: f"2024-{x:02d}")
    
    # 그래프 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 선택된 월 강조
    if selected_month != "전체":
        other_months = monthly[monthly['month'] != selected_month]
        selected = monthly[monthly['month'] == selected_month]
        
        # 다른 월 (회색)
        fig.add_trace(go.Bar(
            x=other_months['label'],
            y=other_months['전력사용량(kWh)'],
            name='월별 사용량',
            marker_color='lightgray',
            text=other_months['전력사용량(kWh)'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside'
        ), secondary_y=False)
        
        # 선택 월 (파란색)
        fig.add_trace(go.Bar(
            x=selected['label'],
            y=selected['전력사용량(kWh)'],
            name=f'{selected_month}월',
            marker_color='#1f77b4',
            text=selected['전력사용량(kWh)'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside'
        ), secondary_y=False)
    else:
        # 전체 (회색)
        fig.add_trace(go.Bar(
            x=monthly['label'],
            y=monthly['전력사용량(kWh)'],
            name='월별 사용량',
            marker_color='lightgray',
            text=monthly['전력사용량(kWh)'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside'
        ), secondary_y=False)
    
    # 평균 요금 라인
    fig.add_trace(go.Scatter(
        x=monthly['label'],
        y=monthly['전기요금(원)'],
        name='월 평균 전기요금',
        mode='lines+markers',
        line=dict(color='crimson', width=2),
        marker=dict(size=8)
    ), secondary_y=True)
    
    fig.update_yaxes(title_text="전력사용량 (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="평균 전기요금 (원)", secondary_y=True)
    fig.update_layout(
        height=450,
        font=dict(color='black'),
        xaxis=dict(tickfont=dict(color='black')),
        yaxis=dict(tickfont=dict(color='black')),
        yaxis2=dict(tickfont=dict(color='black'))
    )

else:  # 기간 모드
    st.subheader("📊 시간대별 전력 사용량 패턴")
    
    # 시간대별 집계
    hourly = filtered_df.groupby('hour').agg({
        '전력사용량(kWh)': ['mean', 'min', 'max']
    }).reset_index()
    hourly.columns = ['hour', 'avg', 'min', 'max']
    hourly['label'] = hourly['hour'].apply(lambda x: f"{x:02d}:00")
    
    # 그래프 생성
    fig = go.Figure()
    
    # 범위 (면적)
    fig.add_trace(go.Scatter(
        x=hourly['label'],
        y=hourly['max'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly['label'],
        y=hourly['min'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        name='최소-최대 범위'
    ))
    
    # 평균 라인
    fig.add_trace(go.Scatter(
        x=hourly['label'],
        y=hourly['avg'],
        mode='lines+markers',
        name='평균 전력사용량',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        height=450,
        xaxis_title='시간',
        yaxis_title='전력사용량 (kWh)',
        font=dict(color='black'),
        xaxis=dict(tickfont=dict(color='black')),
        yaxis=dict(tickfont=dict(color='black'))
    )

# 그래프와 표 배치
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### 🔎 표시 데이터 표")
    
    # CSV 다운로드 버튼
    csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "⬇️ CSV 다운로드",
        data=csv,
        file_name=f"전력데이터_{label}.csv",
        mime="text/csv"
    )
    
    # 데이터 테이블
    st.dataframe(filtered_df, height=400, use_container_width=True)
    st.caption(f"행 수: {len(filtered_df):,}")

st.divider()

# ===== 시간대별 부하 발생 빈도 + 일별 전력량 분석 (나란히 배치) =====
col_polar, col_daily = st.columns(2)

# ===== 왼쪽: 시간대별 부하 발생 빈도 =====
with col_polar:
    st.subheader("📊 시간대별 부하 발생 빈도")
    
    # 부하 유형 선택
    load_type_select = st.selectbox(
        "부하 유형 선택",
        options=['경부하', '중간부하', '최대부하'],
        index=2  # 기본값: 최대부하
    )
    
    # 역매핑
    reverse_map = {
        '경부하': 'Light_Load',
        '중간부하': 'Medium_Load',
        '최대부하': 'Maximum_Load'
    }
    
    # 선택한 부하 유형으로 필터링
    selected_load = reverse_map[load_type_select]
    load_filtered = filtered_df[filtered_df['작업유형'] == selected_load]
    
    # 시간대별 빈도 계산
    if len(load_filtered) > 0:
        hour_counts = load_filtered.groupby('hour').size().reindex(range(24), fill_value=0)
    else:
        hour_counts = pd.Series([0] * 24, index=range(24))
    
    # Polar 차트
    fig_polar = go.Figure()
    
    # 부하 유형별 색상
    polar_colors = {
        '경부하': {'line': '#4CAF50', 'fill': 'rgba(76, 175, 80, 0.3)'},
        '중간부하': {'line': '#FFC107', 'fill': 'rgba(255, 193, 7, 0.3)'},
        '최대부하': {'line': '#EF5350', 'fill': 'rgba(239, 83, 80, 0.3)'}
    }
    
    fig_polar.add_trace(go.Scatterpolar(
        r=hour_counts.values,
        theta=[f"{h:02d}:00" for h in range(24)],
        fill='toself',
        fillcolor=polar_colors[load_type_select]['fill'],
        line=dict(color=polar_colors[load_type_select]['line'], width=2),
        marker=dict(size=8, color=polar_colors[load_type_select]['line']),
        name=load_type_select
    ))
    
    max_val = hour_counts.max() if hour_counts.max() > 0 else 10
    
    fig_polar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.1],
                tickfont=dict(color='black')
            ),
            angularaxis=dict(
                tickfont=dict(color='black'),
                direction='clockwise',  # 시계 방향
                rotation=90  # 0시를 위쪽(북쪽)으로
            )
        ),
        height=550,
        font=dict(color='black'),
        showlegend=False
    )
    
    st.plotly_chart(fig_polar, use_container_width=True)
    st.caption(f"📌 선택한 기간 내 **{load_type_select}** 발생 건수: **{len(load_filtered):,}건**")

# ===== 오른쪽: 일별 전력량 분석 =====
with col_daily:
    st.subheader("📊 일별 전력량 분석")
    
    # 작업유형 매핑
    load_map = {
        'Light_Load': '경부하',
        'Medium_Load': '중간부하',
        'Maximum_Load': '최대부하'
    }
    
    analysis_df = filtered_df.copy()
    analysis_df['부하타입'] = analysis_df['작업유형'].map(load_map)
    
    # 일별 집계 (측정일시의 날짜 부분만 사용)
    analysis_df['날짜'] = analysis_df['측정일시'].dt.date
    daily = analysis_df.groupby(['날짜', '부하타입'])['전력사용량(kWh)'].sum().reset_index()
    
    # 피벗 테이블 생성
    daily_pivot = daily.pivot(index='날짜', columns='부하타입', values='전력사용량(kWh)').fillna(0)
    daily_pivot = daily_pivot.reset_index()
    
    # 날짜를 정렬
    daily_pivot = daily_pivot.sort_values('날짜')
    
    # x축 라벨 생성 (월-일 형식)
    daily_pivot['날짜_str'] = pd.to_datetime(daily_pivot['날짜']).dt.strftime('%m-%d')
    
    # Stacked Bar 차트
    fig_daily = go.Figure()
    
    colors = {
        '경부하': '#4CAF50',
        '중간부하': '#FFC107',
        '최대부하': '#EF5350'
    }
    
    for load_type in ['경부하', '중간부하', '최대부하']:
        if load_type in daily_pivot.columns:
            fig_daily.add_trace(go.Bar(
                name=load_type,
                x=daily_pivot['날짜_str'],
                y=daily_pivot[load_type],
                marker_color=colors[load_type],
                hovertemplate='날짜: %{x}<br>' + load_type + ': %{y:,.0f} kWh<extra></extra>'
            ))
    
    fig_daily.update_layout(
        barmode='stack',
        height=550,
        xaxis_title='날짜',
        yaxis_title='전력사용량 (kWh)',
        font=dict(color='black'),
        xaxis=dict(
            tickfont=dict(color='black'),
            tickangle=-45,
            type='category'  # 카테고리 타입으로 설정
        ),
        yaxis=dict(tickfont=dict(color='black')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    st.caption(f"📌 총 **{len(daily_pivot)}일** 데이터")

st.divider()



# ===============================================================
# 역률 관리도 그래프
# ==============================================================
st.divider()
st.subheader("⚡ 작업휴무별 역률 일일 사이클 분석 (선택 필터 적용)")

# 1. 작업휴무 선택 체크박스 추가
st.markdown("##### 📌 분석 대상 선택")
col_check_1, col_check_2 = st.columns(2)

with col_check_1:
    show_on_work = st.checkbox("✅ 가동일 패턴 보기", value=True) 
with col_check_2:
    show_on_off = st.checkbox("❌ 휴무일 패턴 보기", value=True) 
    
# 2. 일일 사이클 집계 및 데이터 준비 (기존 로직 유지)
cycle_df = filtered_df.copy()

# 15분 단위로 그룹핑
cycle_df['time_15min'] = ((cycle_df['hour'] * 60 + cycle_df['minute']) // 15) * 15 
cycle_df['time_label'] = cycle_df['time_15min'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")

# 작업휴무, 15분 단위 시간별 평균 역률 계산
# 주의: Plotly에서 연속적인 X축을 위해 전체 0~96 인덱스를 모두 포함해야 함.
# 여기서는 time_label을 X축으로 사용하고, Plotly가 자동으로 카테고리 순서를 처리하도록 함.
daily_cycle = cycle_df.groupby(['작업휴무', 'time_15min', 'time_label']).agg(
    avg_lag_pf=('지상역률(%)', 'mean'),
    avg_lead_pf=('진상역률(%)', 'mean')
).reset_index().sort_values('time_15min')

# X축에 사용할 모든 15분 단위 레이블 생성 (00:00 ~ 23:45)
all_time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]

# 3. 차트 분할 배치
col_lag, col_lead = st.columns(2)

# =================================================================
# 3-1. 지상역률 (Lagging PF) 일일 사이클 차트
# =================================================================
with col_lag:
    st.markdown("#### 🟢 지상역률(%) 일일 사이클 (추가/감액 기준: 90%)")
    
    fig_lag = go.Figure()

    # KEPCO 규정 시간 배경 (09:00 ~ 22:00)
    # x0, x1을 'time_label' (Category)로 직접 지정
    fig_lag.add_vrect(
        x0="09:00", x1="22:00", 
        fillcolor="yellow", opacity=0.15, layer="below", line_width=0,
        annotation_text="KEPCO 규제 시간 (09시~22시)", 
        annotation_position="top left"
    )

    # **체크박스 조건에 따른 라인 추가**
    if show_on_work:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == '가동']
        fig_lag.add_trace(go.Scatter(
            x=df_plot['time_label'], # ***수정: time_label 사용***
            y=df_plot['avg_lag_pf'],
            mode='lines',
            name='가동',
            line=dict(color='#1f77b4', width=2)
        ))
        
    if show_on_off:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == '휴무']
        fig_lag.add_trace(go.Scatter(
            x=df_plot['time_label'], # ***수정: time_label 사용***
            y=df_plot['avg_lag_pf'],
            mode='lines',
            name='휴무',
            line=dict(color='#ff7f0e', width=2)
        ))

    # 요금제 기준선 (90%)
    fig_lag.add_hline(
        y=90, line_dash="dash", line_color="red", line_width=2, name="요금제 기준선 (90%)"
    )

    # 레이아웃 설정
    fig_lag.update_layout(
        height=500,
        xaxis=dict(
            title="시간 (Hour, 15분 단위)",
            categoryorder='array',
            categoryarray=all_time_labels, # 전체 카테고리 배열 지정
            tickvals=[f"{h:02d}:00" for h in range(24)], # 1시간 간격만 눈금 표시
            ticktext=[f"{h}" for h in range(24)],
            tickangle=0,
            tickfont=dict(color='black')
        ),
        yaxis=dict(title="평균 지상역률(%)", range=[40, 102], tickfont=dict(color='black')),
        legend=dict(title='작업휴무', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50)
    )
    st.plotly_chart(fig_lag, use_container_width=True)


# =================================================================
# 3-2. 진상역률 (Leading PF) 일일 사이클 차트
# =================================================================
with col_lead:
    st.markdown("#### 🔴 진상역률(%) 일일 사이클 (추가 요금 기준: 95%)")
    
    fig_lead = go.Figure()

    # KEPCO 규정 시간 배경 (22시~09시, 야간)
    fig_lead.add_vrect(x0="22:00", x1="23:45", fillcolor="orange", opacity=0.15, layer="below", line_width=0)
    fig_lead.add_vrect(
        x0="00:00", x1="09:00", 
        fillcolor="orange", opacity=0.15, layer="below", line_width=0,
        annotation_text="KEPCO 규제 시간 (22시~09시)", 
        annotation_position="top left"
    )

    # **체크박스 조건에 따른 라인 추가**
    if show_on_work:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == '가동']
        fig_lead.add_trace(go.Scatter(
            x=df_plot['time_label'], # ***수정: time_label 사용***
            y=df_plot['avg_lead_pf'],
            mode='lines',
            name='가동',
            line=dict(color='#1f77b4', width=2)
        ))
        
    if show_on_off:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == '휴무']
        fig_lead.add_trace(go.Scatter(
            x=df_plot['time_label'], # ***수정: time_label 사용***
            y=df_plot['avg_lead_pf'],
            mode='lines',
            name='휴무',
            line=dict(color='#ff7f0e', width=2)
        ))

    # 요금제 기준선 (95%)
    fig_lead.add_hline(
        y=95, line_dash="dash", line_color="red", line_width=2, name="요금제 기준선 (95%)"
    )

    # 레이아웃 설정
    fig_lead.update_layout(
        height=500,
        xaxis=dict(
            title="시간 (Hour, 15분 단위)",
            categoryorder='array',
            categoryarray=all_time_labels, # 전체 카테고리 배열 지정
            tickvals=[f"{h:02d}:00" for h in range(24)], # 1시간 간격만 눈금 표시
            ticktext=[f"{h}" for h in range(24)],
            tickangle=0,
            tickfont=dict(color='black')
        ),
        yaxis=dict(title="평균 진상역률(%)", range=[0, 102], tickfont=dict(color='black')),
        legend=dict(title='작업휴무', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50)
    )
    st.plotly_chart(fig_lead, use_container_width=True)

# 4. 종합 캡션 (분석 결과 요약)
st.markdown("##### 🔍 분석 결과 요약:")
# (분석 결과 캡션은 그대로 유지)
# ...

st.divider()