import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(page_title="전력 데이터 분석", page_icon="📊", layout="wide")


# ----------------- 데이터 로드 및 전처리 -----------------

# @st.cache_data를 사용하여 데이터 로딩 최적화
@st.cache_data
def load_data():
    # 파일 경로를 사용자님의 환경에 맞게 조정해주세요. 주석추가
    df = pd.read_csv(r"C:\Users\USER\Desktop\electric_power_-team\data\train_df.csv")
    # df = pd.read_csv("train_df.csv") # 사용자님이 업로드해주신 파일명을 사용했습니다.

    # 필수 날짜/시간 처리
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['month'] = df['측정일시'].dt.month
    df['year'] = df['측정일시'].dt.year
    df['day'] = df['측정일시'].dt.day
    df['hour'] = df['측정일시'].dt.hour
    df['date'] = df['측정일시'].dt.date
    
    # '단가' 결측치 1개 처리 (혹시 모를 에러 방지)
    df.dropna(subset=['단가'], inplace=True) 

    return df

# 데이터 로드
df = load_data()

# ----------------- 월별 전체 데이터 집계 함수 (메인 그래프용) -----------------
# 기간 필터에 관계없이 항상 1년 추이를 보여주기 위한 함수
@st.cache_data
def get_monthly_all_data(data_df):
    """전체 데이터셋 기준으로 월별 전력사용량 합계와 평균 요금을 계산합니다."""
    monthly = data_df.groupby('month').agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'mean'
    }).reset_index()
    # 11월 데이터까지 있으므로 11월까지만 사용
    monthly = monthly[monthly['month'] <= 11]
    monthly['label'] = monthly['month'].apply(lambda x: f"2024-{x:02d}")
    return monthly

# ----------------- Streamlit UI 시작 -----------------

# ----------------- PDF 파일 로드 함수 (Streamlit Download Button용) -----------------

@st.cache_data
def get_pdf_bytes(file_path):
    """PDF 파일을 바이너리 형태로 읽어 Streamlit download_button에 전달합니다."""
    try:
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        return pdf_bytes
    except FileNotFoundError:
        st.error(f"🚨 파일을 찾을 수 없습니다: {file_path}")
        return None

# PDF 파일명 정의
RATE_FILE_NAME = "C:\\Users\\USER\\Desktop\\electric_power_-team\\2024년도7월1일시행전기요금표(종합)_출력용.pdf"
pdf_data = get_pdf_bytes(RATE_FILE_NAME)

# ----------------- Streamlit UI 시작 -----------------

# 페이지 제목과 다운로드 버튼을 나란히 배치하기 위해 컬럼 사용
# 제목 60%, 버튼 3개 (총 40%) 공간을 차지하도록 분할
title_col, report_col, bill_col, rate_col = st.columns([0.6, 0.13, 0.13, 0.14]) 

with title_col:
    st.title("📊 LS ELECTRIC 청주 공장 전력 사용 현황")
    
# 다운로드 버튼에 사용할 데이터 준비 (고지서와 보고서 모두 월별 집계 데이터 사용)
# monthly_download_data = get_monthly_all_data(df)는 상단에 정의되어 있어야 합니다.
monthly_download_data = get_monthly_all_data(df)
csv_monthly = monthly_download_data.to_csv(index=False, encoding='utf-8-sig')


# ----------------- 2. 다운로드 버튼 배치 -----------------

# 2-1. 보고서 다운로드 버튼 (파란색 워드 파일)
with report_col:
    st.markdown("<br>", unsafe_allow_html=True) # 제목과의 높이 맞추기 위한 공백
    st.download_button(
        label="보고서 다운로드",
        data=csv_monthly,
        file_name="에너지_분석_보고서.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
        key="report_btn", 
        help="분석 보고서 (가상 DOCX)를 다운로드합니다."
    )

# 2-2. 고지서 다운로드 버튼 (초록색 CSV 파일)
with bill_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="고지서 다운로드",
        data=csv_monthly,
        file_name="월별_에너지_고지서_집계.csv",
        mime="text/csv",
        key="bill_btn",
        help="전체 기간의 월별 집계 데이터를 CSV 형태로 다운로드합니다."
    )

# 2-3. 요금표 다운로드 버튼 (보라색 PDF 파일)
with rate_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if pdf_data:
        st.download_button(
            label="요금표 다운로드",
            data=pdf_data,
            file_name=RATE_FILE_NAME,
            mime="application/pdf", 
            key="rate_btn",
            help="2024년 7월 1일 시행 전기요금표 (PDF)를 다운로드합니다."
        )

st.divider()

# ===== 사이드바 필터 (수정) =====
st.sidebar.header("🔍 필터 선택")

# 1. 월별 선택 (기본 분석 단위)
month_options = ["전체"] + sorted(df['month'].unique().tolist()) # 1월~11월이므로
selected_month = st.sidebar.selectbox(
    "1️⃣ 분석할 월을 선택하세요",
    options=month_options,
    format_func=lambda x: "전체(1~11월)" if x == "전체" else f"{x}월"
)

st.sidebar.markdown("---")
st.sidebar.markdown("2️⃣ **세부 기간 선택 (선택 사항)**")

# 2. 세부 기간 선택 (보조 필터)
min_date = df['측정일시'].min().date()
max_date = df['측정일시'].max().date()

# 기본값을 월별 선택에 따라 설정
if selected_month != "전체":
    # 선택된 월의 시작일과 마지막 날짜를 계산
    if selected_month == 11:
        # 데이터가 11월 30일까지 있으므로 max_date를 그대로 사용
        month_start = df[df['month'] == selected_month]['측정일시'].min().date()
        month_end = df[df['month'] == selected_month]['측정일시'].max().date()
    else:
        month_start = df[df['month'] == selected_month]['측정일시'].min().date()
        # 다음 달의 1일 - 1일을 구하여 해당 월의 마지막 날짜를 정확히 설정해야 하지만,
        # 편의상 해당 월의 최대 날짜로 설정
        month_end = df[df['month'] == selected_month]['측정일시'].max().date()
        
    date_input_value = (month_start, month_end)
else:
    date_input_value = (min_date, max_date) # 전체 기간

# 사용자가 직접 세부 기간을 설정할 수 있게 함
date_range = st.sidebar.date_input(
    "기간을 직접 지정하세요",
    value=date_input_value,
    min_value=min_date,
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.markdown("3️⃣ **작업 상태 선택**")

# 3. 작업휴무 체크박스 필터 (기존 코드 유지)
work_status_options = sorted(df['작업휴무'].unique().tolist())
selected_work_status = st.sidebar.multiselect(
    "작업 여부 선택",
    options=work_status_options,
    default=work_status_options
)

# ----------------- 필터링 로직 (수정) -----------------

# 최종 필터링은 date_range와 selected_work_status를 따릅니다.
if len(date_range) == 2:
    start_date, end_date = date_range
    # 1차 필터링: 기간 필터 적용
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # 레이블 설정 (기간이 월별 선택과 일치하는지 확인)
    is_full_month = (start_date == date_input_value[0] and end_date == date_input_value[1])
    if selected_month != "전체" and is_full_month:
        label = f"{selected_month}월"
    elif selected_month == "전체" and start_date == min_date and end_date == max_date:
        label = "전체(1~11월)"
    else:
        label = f"{start_date} ~ {end_date}"
else:
    # 날짜 입력이 완료되지 않았을 경우
    filtered_df = df.copy()
    label = "전체"
    
# 2차 필터링: 작업휴무 필터 적용
if selected_work_status:
    filtered_df = filtered_df[filtered_df['작업휴무'].isin(selected_work_status)].copy()
    
# 필터링 결과 확인
if filtered_df.empty:
    st.error("선택된 필터 조건에 해당하는 데이터가 없습니다. 필터를 조정해주세요.")
    st.stop()

    
# ===== 주요 지표 (KPI) =====
st.markdown(f"## 📅 {label} 주요 지표")
st.markdown(
    f"**데이터 기간**: {filtered_df['측정일시'].min().strftime('%Y-%m-%d')} ~ "
    f"{filtered_df['측정일시'].max().strftime('%Y-%m-%d')}"
)

# KPI 계산
total_power = filtered_df['전력사용량(kWh)'].sum()
total_cost = filtered_df['전기요금(원)'].sum()
total_carbon = (filtered_df['탄소배출량(tCO2)'].sum()) * 1000

# >>> 수정된 Day KPI 계산 <<<
# 1. Day KPI 계산을 위해 작업휴무 필터가 적용되기 전의 데이터(날짜 필터만 적용된 데이터)를 준비합니다.
if len(date_range) == 2:
    start_date, end_date = date_range
    # 원본 df에서 날짜만 필터링한 DataFrame
    df_for_day_count = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
else:
    df_for_day_count = df.copy()

# 2. 작업휴무별 유니크한 날짜(일) 수를 계산합니다.
# '가동일'은 '가동' 상태가 한 번이라도 기록된 날짜의 총 개수입니다.
total_working_days = df_for_day_count[df_for_day_count['작업휴무'] == "가동"]['date'].nunique()
total_holiday_days = df_for_day_count[df_for_day_count['작업휴무'] == "휴무"]['date'].nunique()

# KPI 스타일 (이전 코드 유지)
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
        <div class="kpi-value">{total_carbon:,.0f}</div>
        <div class="kpi-unit">CO2[Kg]</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">총 가동일 수</div>
        <div class="kpi-value">{total_working_days:,}</div> 
        <div class="kpi-unit">일</div>
    </div>
    """, unsafe_allow_html=True) # 제목 및 단위 수정

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">총 휴무일 수</div>
        <div class="kpi-value">{total_holiday_days:,}</div>
        <div class="kpi-unit">일</div>
    </div>
    """, unsafe_allow_html=True) # 제목 및 단위 수정

st.divider()

# ----------------- 2. 월별 분석 (추이 및 비교) -----------------
st.header("1️⃣ 월별 전력 사용 개요")
col_monthly_trend, col_monthly_comp = st.columns(2)

# ===============================================================
# 2-1. 좌측 그래프: 월별 전력사용량 + 월 평균 전기요금 추이 (전체 기간)
# ===============================================================
with col_monthly_trend:
    st.subheader("월별 전력사용량 및 평균 요금 추이")
    
    # 월별 전체 데이터 집계 (전체 df 사용)
    monthly = get_monthly_all_data(df)

    # -------------------------------------------------------------
    # ⭐ 수정된 로직: 선택된 월에 따른 막대 색상 결정
    # -------------------------------------------------------------
    bar_colors = []
    
    # selected_month는 문자열 ("전체" 또는 "1", "2"...) 형태일 수 있습니다.
    if selected_month == "전체":
        # '전체' 선택 시, 모든 막대를 기본 파란색으로 표시
        bar_colors = ['#1f77b4'] * len(monthly)
    else:
        # 특정 월이 선택된 경우, 해당 월만 파란색, 나머지는 회색
        try:
            # 사이드바에서 선택된 selected_month가 문자열일 수 있으므로 int 변환
            selected_month_int = int(selected_month) 
        except ValueError:
            selected_month_int = -1 # 안전값

        for month_num in monthly['month']:
            if month_num == selected_month_int:
                bar_colors.append('#1f77b4') # 선택 월: 파란색
            else:
                bar_colors.append('lightgray') # 나머지 월: 회색


    # 그래프 생성 (이전 코드 재활용)
    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 월별 사용량 (막대)
    fig_monthly.add_trace(go.Bar(
        x=monthly['label'],
        y=monthly['전력사용량(kWh)'],
        name='월별 사용량',
        marker_color=bar_colors, # 동적 색상 리스트 적용
        text=monthly['전력사용량(kWh)'].apply(lambda x: f"{x:,.0f}"),
        textposition='outside'
    ), secondary_y=False)
    
    # 평균 요금 라인
    fig_monthly.add_trace(go.Scatter(
        x=monthly['label'],
        y=monthly['전기요금(원)'],
        name='월 평균 전기요금',
        mode='lines+markers',
        line=dict(color='crimson', width=2),
        marker=dict(size=8)
    ), secondary_y=True)
    
    # ⭐ 수정된 부분: 그리드 제거
    fig_monthly.update_xaxes(showgrid=False)
    
    # ⭐ 수정된 부분: y축 (좌측) 그리드 제거
    fig_monthly.update_yaxes(title_text="전력사용량 (kWh)", secondary_y=False, showgrid=False)
    
    # ⭐ 수정된 부분: y축 (우측) 그리드 제거
    fig_monthly.update_yaxes(title_text="평균 전기요금 (원)", secondary_y=True, showgrid=False)
    
    fig_monthly.update_layout(
        height=450,
        font=dict(color='black'),
        xaxis=dict(tickfont=dict(color='black')),
        yaxis=dict(tickfont=dict(color='black')),
        yaxis2=dict(tickfont=dict(color='black'))
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

# ===============================================================
# 2-2. 우측 그래프: 선택 월 vs. 24년 평균 전력 사용량 비교 (신규)
# ===============================================================
# -------------------------------------------------------------
# 1. 월평균/전월 데이터 계산 블록 (col_monthly_comp 블록 외부, 필터링 로직 아래에 추가)
# -------------------------------------------------------------

# 연간 월별 총 전력사용량 목록 계산 (전체 데이터 기준)
monthly_totals_all = df.groupby('month')['전력사용량(kWh)'].sum()
# 2024년 월평균 전력사용량 (월별 총합의 평균)
annual_monthly_avg_power = monthly_totals_all.mean()

# 현재 선택된 기간의 총 전력사용량
selected_period_total_power = filtered_df['전력사용량(kWh)'].sum()

# -------------------------------------------------------------
# 2. 우측 그래프: 선택 월 vs. 24년 월평균 vs. 전월 총 전력사용량 비교 (기존 코드 유지)
# -------------------------------------------------------------
with col_monthly_comp:
    st.subheader("총 전력사용량 비교")
    
    comp_labels = [label, '2024년 월평균']
    comp_values = [selected_period_total_power, annual_monthly_avg_power]
    comp_colors = {label: '#1f77b4', '2024년 월평균': 'lightgray'}
    comp_title = '총 전력사용량 (kWh)'
    
    # 순서를 정의할 리스트
    category_order = ['2024년 월평균'] 

    # '전체' 월이 아닌 숫자가 선택되었고, 해당 월이 1월이 아닌 경우 (전월 데이터 필요)
    if isinstance(selected_month, int) and selected_month > df['month'].min():
        prev_month = selected_month - 1
        
        # 전월 전체 데이터의 총 전력사용량
        prev_month_total_power = monthly_totals_all.get(prev_month, 0)
        
        # 데이터 목록에 전월 데이터 추가 및 순서 조정
        prev_label = f'{prev_month}월 (전월)'
        comp_labels.append(prev_label)
        comp_values.append(prev_month_total_power)
        comp_colors[prev_label] = '#ffb366' # 옅은 주황색으로 강조도 낮춤
        
        # 순서 리스트에 전월 추가
        category_order.append(prev_label)

    # 순서 리스트의 마지막에 현재 선택된 기간/월 추가
    category_order.append(label)

    # 데이터프레임 구성
    comp_data = pd.DataFrame({
        '구분': comp_labels,
        comp_title: comp_values
    })

    # 막대 그래프
    fig_comp = px.bar(
        comp_data, 
        x='구분', 
        y=comp_title, 
        color='구분',
        color_discrete_map=comp_colors,
        text=comp_title,
        title='선택 기간/월 총 전력사용량 비교'
    )
    
    # ⭐ 수정된 부분: 막대 위에 표시되는 텍스트 색상을 검정색으로 지정 (명시적 지정)
    fig_comp.update_traces(
        texttemplate='%{text:,.0f} kWh', 
        textposition='outside',
        textfont=dict(color='black')
    )
    
    # ⭐ 수정된 부분: x축 카테고리 순서 지정
    fig_comp.update_xaxes(
        categoryorder='array', 
        categoryarray=category_order, # 정의된 순서 리스트를 적용
        tickfont=dict(color='black') # 축 눈금 글씨색
    )
    
    # y축 범위 설정
    max_val = comp_data[comp_title].max() if not comp_data.empty else 1
    
    fig_comp.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="",
        yaxis_title=comp_title,
        yaxis_range=[0, max_val * 1.2],
        font=dict(color='black') # ⭐ 수정된 부분: 그래프 제목, 축 레이블 등 기본 글꼴 색상
    )
    
    fig_comp.update_yaxes(tickfont=dict(color='black')) # y축 눈금 글씨색

    st.plotly_chart(fig_comp, use_container_width=True)
# -------------------------------------------------------------
# 3. 분석 캡션을 위한 일평균 변수 재정의 (오류 해결)
# -------------------------------------------------------------

# 연간 일평균 계산 (전체 데이터)
annual_daily_avg = df.groupby(df['측정일시'].dt.date)['전력사용량(kWh)'].sum().mean()

# 선택 기간의 일평균 계산 (filtered_df 사용)
selected_month_daily_sum = filtered_df.groupby(filtered_df['측정일시'].dt.date)['전력사용량(kWh)'].sum()
selected_month_daily_avg = selected_month_daily_sum.mean() if not selected_month_daily_sum.empty else 0


st.markdown("##### 🔍 월별 분석 결과 요약:")
st.caption(f"월별 추이 그래프는 계절적 요인(예: 여름철 냉방)에 따른 사용량 변화를 보여줍니다. 선택된 **{label}**의 일평균 사용량({selected_month_daily_avg:,.0f} kWh)은 연간 일평균({annual_daily_avg:,.0f} kWh)과 비교하여 현재 사용 수준을 가늠할 수 있습니다.")
st.divider()

# ----------------- 3. 일별 분석 (전력 및 요금) -----------------
st.header("2️⃣ 일별 사용량 및 재무 영향 분석")
col_daily_power, col_daily_cost = st.columns(2)

# ===============================================================
# 3-1. 좌측 그래프: 일별 전력량 분석 (Stacked Bar) (유지)
# ===============================================================
with col_daily_power:
    st.subheader("📊 일별 전력량 분석")
    
    # 작업유형 매핑 (이전 코드 유지)
    load_map = {
        'Light_Load': '경부하',
        'Medium_Load': '중간부하',
        'Maximum_Load': '최대부하'
    }
    
    analysis_df = filtered_df.copy()
    analysis_df['부하타입'] = analysis_df['작업유형'].map(load_map)
    
    # 일별 집계
    analysis_df['날짜'] = analysis_df['측정일시'].dt.date
    daily = analysis_df.groupby(['날짜', '부하타입'])['전력사용량(kWh)'].sum().reset_index()
    
    # 피벗 테이블 생성
    daily_pivot = daily.pivot(index='날짜', columns='부하타입', values='전력사용량(kWh)').fillna(0).reset_index()
    daily_pivot = daily_pivot.sort_values('날짜')
    daily_pivot['날짜_str'] = pd.to_datetime(daily_pivot['날짜']).dt.strftime('%m-%d')
    
    # Stacked Bar 차트 (이전 코드 유지)
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
        xaxis=dict(showgrid=False, tickfont=dict(color='black'), tickangle=-45, type='category'),
        yaxis=dict(showgrid=False, tickfont=dict(color='black')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ===============================================================
# 3-2. 우측 그래프: 일별 전기요금 분석 (탄소 -> 요금으로 변경)
# ===============================================================
with col_daily_cost:
    st.subheader("💰 일별 총 전기요금 추이 (원)")

    # 일별 전기요금 합계 계산
    daily_cost = filtered_df.groupby(filtered_df['측정일시'].dt.date)['전기요금(원)'].sum().reset_index()
    daily_cost.columns = ['날짜', '총 전기요금(원)']
    daily_cost['날짜_str'] = pd.to_datetime(daily_cost['날짜']).dt.strftime('%m-%d')

    fig_cost = px.line(
        daily_cost,
        x='날짜_str',
        y='총 전기요금(원)',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['#28a745'] # 녹색 계열 (비용/재무)
    )
    fig_cost.update_layout(
        height=550,
        xaxis_title='날짜',
        yaxis_title='총 전기요금 (원)',
        font=dict(color='black'),
        xaxis=dict(showgrid=False, tickfont=dict(color='black'), tickangle=-45, type='category'),
        yaxis=dict(showgrid=False, tickfont=dict(color='black'))
    )
    st.plotly_chart(fig_cost, use_container_width=True)

st.markdown("##### 🔍 일별 분석 결과 요약:")
st.caption("일별 전력량 분석은 부하 유형별 사용량 분포를 보여주어 **설비 운영 패턴**을 파악하는 데 유용합니다. 특히 **일별 전기요금 추이**를 통해 사용량은 비슷하더라도 **시간대별 단가(TOU)**에 의해 요금이 급증하는 날을 식별하여 **요금 효율성**을 검토할 수 있습니다.")
st.divider()

# ----------------- 4. 시간대 패턴 분석 -----------------
st.header("3️⃣ 시간대별 패턴 분석")
col_hourly_pattern, col_hourly_load = st.columns(2)

# ===============================================================
# 4-1. 좌측 그래프: 시간대별 전력 사용량 패턴 (기존 좌측 그래프)
# ===============================================================
with col_hourly_pattern:
    st.subheader("📈 시간대별 전력 사용량 패턴 (평균/최소/최대)")
    
    # 시간대별 집계 (선택 기간의 데이터 사용)
    hourly = filtered_df.groupby('hour').agg({
        '전력사용량(kWh)': ['mean', 'min', 'max']
    }).reset_index()
    hourly.columns = ['hour', 'avg', 'min', 'max']
    hourly['label'] = hourly['hour'].apply(lambda x: f"{x:02d}:00")
    
    # 그래프 생성 (이전 코드 재활용)
    fig_hourly = go.Figure()
    
    # 범위 (면적)
    fig_hourly.add_trace(go.Scatter(
        x=hourly['label'], y=hourly['max'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly['label'], y=hourly['min'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', name='최소-최대 범위'
    ))
    
    # 평균 라인
    fig_hourly.add_trace(go.Scatter(
        x=hourly['label'], y=hourly['avg'], mode='lines+markers', name='평균 전력사용량',
        line=dict(color='#1f77b4', width=3), marker=dict(size=6)
    ))
    
    fig_hourly.update_layout(
        height=550, # 높이 조정
        xaxis_title='시간', yaxis_title='전력사용량 (kWh)', font=dict(color='black'),
        xaxis=dict(tickfont=dict(color='black')), yaxis=dict(tickfont=dict(color='black'))
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# ===============================================================
# 4-2. 우측 그래프: 시간대별 부하 발생 빈도 (기존 좌측 그래프)
# ===============================================================
with col_hourly_load:
    st.subheader("📊 시간대별 부하 발생 빈도")

    # 부하 유형 매핑 (기존 코드 유지)
    load_map = { '경부하': 'Light_Load', '중간부하': 'Medium_Load', '최대부하': 'Maximum_Load' }
    polar_colors = {
        '경부하': {'line': '#4CAF50', 'fill': 'rgba(76, 175, 80, 0.3)'},
        '중간부하': {'line': '#FFC107', 'fill': 'rgba(255, 193, 7, 0.3)'},
        '최대부하': {'line': '#EF5350', 'fill': 'rgba(239, 83, 80, 0.3)'}
    }
    
    # 부하 유형 다중 선택 (체크박스)
    st.markdown("##### 부하 유형 선택")
    col_check1, col_check2, col_check3 = st.columns(3)
    selected_loads_ui = []
    if col_check1.checkbox('최대부하', value=True, key="p1"): selected_loads_ui.append('최대부하')
    if col_check2.checkbox('중간부하', value=True, key="p2"): selected_loads_ui.append('중간부하')
    if col_check3.checkbox('경부하', value=True, key="p3"): selected_loads_ui.append('경부하')

    fig_polar = go.Figure()
    all_hour_counts = []
    total_count = 0
    
    if not selected_loads_ui:
        st.warning("⚠️ 최소한 하나의 부하 유형을 선택해야 합니다.")
    else:
        for load_ui_name in selected_loads_ui:
            load_data_name = load_map[load_ui_name]
            load_filtered = filtered_df[filtered_df['작업유형'] == load_data_name]
            hour_counts = load_filtered.groupby('hour').size().reindex(range(24), fill_value=0)
            total_count += len(load_filtered)
            all_hour_counts.extend(hour_counts.values.tolist())
            
            fig_polar.add_trace(go.Scatterpolar(
                r=hour_counts.values, theta=[f"{h:02d}:00" for h in range(24)], fill='toself',
                fillcolor=polar_colors[load_ui_name]['fill'],
                line=dict(color=polar_colors[load_ui_name]['line'], width=2),
                marker=dict(size=8, color=polar_colors[load_ui_name]['line']),
                name=load_ui_name
            ))

        max_val = max(all_hour_counts) if all_hour_counts else 10
        fig_polar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.1])),
            height=550, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_polar, use_container_width=True)
        st.caption(f"📌 선택한 기간 내 **선택 부하 유형** 총 발생 건수: **{total_count:,}건**")

st.markdown("##### 🔍 시간대별 분석 결과 요약:")
st.caption("시간대별 전력 패턴은 하루 중 설비 가동 시간 및 피크 시간대를 파악하는 데 중요합니다. 부하 발생 빈도를 극좌표 차트로 확인하여 전력 품질에 영향을 줄 수 있는 특정 시간대(예: 최대 부하의 집중 시간)를 시각적으로 분석할 수 있습니다.")
st.divider()


# =========================================================
# 역률 그래프
# =========================================================
st.subheader("⚡ 작업휴무별 역률 일일 사이클 분석")

# 부하 유형별 색상 사전 정의 (일관성을 위해)
pf_colors = {
    '가동': '#1f77b4', # 파란색 계열
    '휴무': '#ff7f0e'  # 주황색 계열
}

# 참고: selected_work_status 변수는 상위 코드 (사이드바 로직)에서 정의되어 있습니다.
# selected_work_status = ['가동', '휴무'] 또는 ['가동'] 등

st.markdown(f"##### 📌 분석 대상: {', '.join(selected_work_status)}일 패턴")
if not selected_work_status:
    st.warning("⚠️ 사이드바에서 '작업 상태 선택' 필터를 통해 최소한 '가동' 또는 '휴무'를 선택해야 합니다.")
    # st.stop() # 필터링된 데이터가 없으므로 상위 로직에서 멈추지만, 사용자 편의를 위해 여기에 경고만 표시

# 1. 일일 사이클 집계 및 데이터 준비
cycle_df = filtered_df.copy()

# 15분 단위로 그룹핑
cycle_df['time_15min'] = ((cycle_df['hour'] * 60 + cycle_df['minute']) // 15) * 15 
cycle_df['time_label'] = cycle_df['time_15min'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")

# 작업휴무, 15분 단위 시간별 평균 역률 계산
daily_cycle = cycle_df.groupby(['작업휴무', 'time_15min', 'time_label']).agg(
    avg_lag_pf=('지상역률(%)', 'mean'),
    avg_lead_pf=('진상역률(%)', 'mean')
).reset_index().sort_values('time_15min')

# X축에 사용할 모든 15분 단위 레이블 생성 (00:00 ~ 23:45)
all_time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]

# 2. 차트 분할 배치
col_lag, col_lead = st.columns(2)

# =================================================================
# 2-1. 지상역률 (Lagging PF) 일일 사이클 차트
# =================================================================
with col_lag:
    st.markdown("#### 🟢 지상역률(%) 일일 사이클 (추가/감액 기준: 90%)")
    
    fig_lag = go.Figure()

    # KEPCO 규정 시간 배경 (09:00 ~ 22:00)
    fig_lag.add_vrect(
        x0="09:00", x1="22:00", 
        fillcolor="yellow", opacity=0.15, layer="below", line_width=0,
        annotation_text="KEPCO 규제 시간 (09시~22시)", 
        annotation_position="top left"
    )

    # **사이드바 필터 조건에 따른 라인 추가**
    for status in selected_work_status:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == status]
        fig_lag.add_trace(go.Scatter(
            x=df_plot['time_label'],
            y=df_plot['avg_lag_pf'],
            mode='lines',
            name=f'{status}', # 가동 또는 휴무
            line=dict(color=pf_colors.get(status, 'gray'), width=2)
        ))
        
    # 요금제 기준선 (90%)
    fig_lag.add_hline(
        y=90, line_dash="dash", line_color="red", line_width=2, 
        annotation_text="요금제 기준선 (90%)", 
        annotation_position="bottom right",
        name="요금제 기준선 (90%)"
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
# 2-2. 진상역률 (Leading PF) 일일 사이클 차트
# =================================================================
with col_lead:
    st.markdown("#### 🔴 진상역률(%) 일일 사이클 (추가 요금 기준: 95%)")
    
    fig_lead = go.Figure()

    # KEPCO 규정 시간 배경 (22시~09시, 야간)
    # 22:00 ~ 23:45
    fig_lead.add_vrect(x0="22:00", x1="23:45", fillcolor="orange", opacity=0.15, layer="below", line_width=0)
    # 00:00 ~ 09:00
    fig_lead.add_vrect(
        x0="00:00", x1="09:00", 
        fillcolor="orange", opacity=0.15, layer="below", line_width=0,
        annotation_text="KEPCO 규제 시간 (22시~09시)", 
        annotation_position="top left"
    )

    # **사이드바 필터 조건에 따른 라인 추가**
    for status in selected_work_status:
        df_plot = daily_cycle[daily_cycle['작업휴무'] == status]
        fig_lead.add_trace(go.Scatter(
            x=df_plot['time_label'],
            y=df_plot['avg_lead_pf'],
            mode='lines',
            name=f'{status}', # 가동 또는 휴무
            line=dict(color=pf_colors.get(status, 'gray'), width=2)
        ))
        
    # 요금제 기준선 (95%)
    fig_lead.add_hline(
        y=95, line_dash="dash", line_color="red", line_width=2, 
        annotation_text="요금제 기준선 (95%)", 
        annotation_position="bottom right",
        name="요금제 기준선 (95%)"
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

# 3. 종합 캡션 (분석 결과 요약)
st.markdown("##### 🔍 분석 결과 요약:")
st.caption("이 차트는 선택된 기간과 작업휴무 조건에 따른 평균 역률 패턴을 보여줍니다. 지상역률은 90% 미만, 진상역률은 95% 초과 시 요금에 영향을 줄 수 있습니다.")

st.divider()