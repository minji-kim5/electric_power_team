import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from report import generate_dynamic_report # report 파일이 없으면 주석 처리

# 페이지 설정
st.set_page_config(page_title="전력 데이터 분석", page_icon="📊", layout="wide")

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
/* ... (Tooltip CSS 및 기타 KPI 관련 스타일도 모두 여기에 배치) ... */
</style>
""", unsafe_allow_html=True)

# ----------------- 데이터 로드 및 전처리 -----------------

@st.cache_data
def load_data():
    # 파일 경로를 사용자님의 환경에 맞게 조정해주세요.
    df = pd.read_csv("data_dash\\train_dash_df.csv") 

    # 필수 날짜/시간 처리
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['month'] = df['측정일시'].dt.month
    df['year'] = df['측정일시'].dt.year
    df['day'] = df['측정일시'].dt.day
    df['hour'] = df['측정일시'].dt.hour
    df['date'] = df['측정일시'].dt.date.astype(str) # KPI 계산을 위해 str로 변환
    
    # '단가' 결측치 1개 처리 (혹시 모를 에러 방지)
    df.dropna(subset=['단가'], inplace=True) 

    return df

# 데이터 로드
df = load_data()

# ----------------- 월별 전체 데이터 집계 함수 (메인 그래프용) -----------------
@st.cache_data
def get_monthly_all_data(data_df):
    """전체 데이터셋 기준으로 월별 전력사용량 합계와 평균 요금을 계산합니다."""
    monthly = data_df.groupby('month').agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'mean'
    }).reset_index()
    monthly = monthly[monthly['month'] <= 11]
    monthly['label'] = monthly['month'].apply(lambda x: f"2024-{x:02d}")
    return monthly


# >>> 수정된 부분: 역률 패널티 데이터 로드 <<<
# 이전에 계산된 월별 요약 데이터 (기본요금, 역률 조정금액 포함)를 로드합니다.
try:
    # 이전에 생성된 파일명인 monthly_power_billing_summary.csv를 사용합니다.
    monthly_summary_df = pd.read_csv('data_dash\\월별 역률 패널티 계산.csv') 
    monthly_summary_df['year'] = monthly_summary_df['year'].astype(int)
    monthly_summary_df['month'] = monthly_summary_df['month'].astype(int)
except FileNotFoundError:
    # 파일이 없는 경우, 빈 DataFrame을 만들어 오류를 피하고 0으로 표시합니다.
    st.error("🚨 오류: 'monthly_power_billing_summary.csv' 파일을 찾을 수 없습니다. 역률 지표를 표시할 수 없습니다.")
    monthly_summary_df = pd.DataFrame(columns=['year', 'month', '역률_조정금액(원)'])
    # st.stop() # 실제 환경에서 파일을 반드시 요구한다면 주석 해제


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
RATE_FILE_NAME = "data_dash\\2024년도7월1일시행전기요금표(종합)_출력용.pdf"
pdf_data = get_pdf_bytes(RATE_FILE_NAME)

# ----------------- Streamlit UI 시작 -----------------

title_col, report_col, bill_col, rate_col = st.columns([0.6, 0.13, 0.13, 0.14]) 

with title_col:
    st.title("📊 LS ELECTRIC 청주 공장 전력 사용 현황")
    
monthly_download_data = get_monthly_all_data(df)
csv_monthly = monthly_download_data.to_csv(index=False, encoding='utf-8-sig')


# ----------------- 2. 다운로드 버튼 배치 (생략 없이 원본 유지) -----------------

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


# ==============================================================================
# ===== 사이드바 필터 (수정) ===== 필터 관련
# ==============================================================================
st.sidebar.header("🔍 필터 선택")

# 1. 월별 선택
month_options = ["전체"] + sorted(df['month'].unique().tolist())
selected_month = st.sidebar.selectbox(
    " 분석할 월을 선택하세요",
    options=month_options,
    format_func=lambda x: "전체(1~11월)" if x == "전체" else f"{x}월"
)

# -------------------------------------------------------------
# ⭐ NEW: 제거된 세부 기간 필터 로직 대체 및 변수 정의
# (Downstream filtering logic이 사용하는 date_range 변수를 정의합니다.)
# -------------------------------------------------------------
min_date = df['측정일시'].min().date()
max_date = df['측정일시'].max().date()

if selected_month != "전체":
    # 선택된 월의 시작일과 마지막 날짜를 계산
    month_start = df[df['month'] == selected_month]['측정일시'].min().date()
    month_end = df[df['month'] == selected_month]['측정일시'].max().date()
    date_range = (month_start, month_end)
else:
    # 전체 기간을 date_range로 설정
    date_range = (min_date, max_date)

# date_input_value는 이제 사용되지 않지만, downstream code의 호환성을 위해 정의합니다.
date_input_value = date_range 
# -------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown(" **작업 상태 선택**")

# 3. 작업휴무 체크박스 필터
work_status_options = sorted(df['작업휴무'].unique().tolist())
selected_work_status = st.sidebar.multiselect(
    "작업 여부 선택",
    options=work_status_options,
    default=work_status_options
)

# ----------------- 필터링 로직 (원본 유지) -----------------

if len(date_range) == 2:
    start_date, end_date = date_range
    # 날짜 입력 위젯은 date 객체를 반환하므로, 비교를 위해 df['date']도 date 객체로 변환하거나 비교합니다.
    # df['date']가 string이므로 start_date, end_date를 string으로 변환하여 비교합니다.
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # 1차 필터링: 기간 필터 적용
    filtered_df = df[(df['date'] >= start_date_str) & (df['date'] <= end_date_str)].copy()
    
    # 레이블 설정
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



# ... (상위 코드 및 탭 생성 코드 유지) ...
# 탭 생성 
# ===================================================

tab1, tab2, tab3 = st.tabs([

    "월별 시각화",

    "일별 시각화",

    "역률 관리 및 비생산 전력 사용"

])

# ============================================================================
# ----------------- 탭 1: 월별 시각화 -----------------
# ============================================================================

with tab1:
    st.markdown("")
    st.caption("")
    st.header("월별 전력 사용 개요")
    
    # -------------------------------------------------------------
    # ⭐⭐ 1. KPI 계산 (KPI 로직은 이 탭에서 사용하므로 여기에 유지) ⭐⭐
    # -------------------------------------------------------------
    
    # ===== 주요 지표 (KPI) 계산 =====
    total_power = filtered_df['전력사용량(kWh)'].sum()
    total_cost = filtered_df['전기요금(원)'].sum()
    total_carbon = (filtered_df['탄소배출량(tCO2)'].sum()) * 1000 # tCO2를 kgCO2로 변환
    
    # >>> Day KPI 계산 (원본 유지) <<<
    # ... (df_for_day_count, total_working_days, total_holiday_days 계산 로직 유지) ...
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        df_for_day_count = df[(df['date'] >= start_date_str) & (df['date'] <= end_date_str)].copy()
    else:
        df_for_day_count = df.copy()
    
    total_working_days = df_for_day_count[df_for_day_count['작업휴무'] == "가동"]['date'].nunique()
    total_holiday_days = df_for_day_count[df_for_day_count['작업휴무'] == "휴무"]['date'].nunique()
    
    # >>> 역률 조정 금액 KPI 계산 (월별 데이터 기반) <<<
    # ... (filtered_months, monthly_summary_filtered, total_pf_adjustment 계산 로직 유지) ...
    filtered_months = filtered_df[['year', 'month']].drop_duplicates()
    monthly_summary_filtered = monthly_summary_df.merge(
        filtered_months, on=['year', 'month'], how='inner'
    )
    if not monthly_summary_filtered.empty:
        total_pf_adjustment = monthly_summary_filtered['역률_조정금액(원)'].sum().round(0).astype(int)
    else:
        total_pf_adjustment = 0 
    
    
    # -------------------------------------------------------------
    # ⭐⭐ 2. KPI 표시 ⭐⭐
    # -------------------------------------------------------------

    st.markdown(f"## 📅 {label} 주요 지표")
    st.markdown(
        f"**데이터 기간**: {filtered_df['측정일시'].min().strftime('%Y-%m-%d')} ~ "
        f"{filtered_df['측정일시'].max().strftime('%Y-%m-%d')}"
    )
    
    # ... (KPI Style CSS 블록 유지) ...
    
    # KPI 카드 (컬럼 5개 유지)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. 총 전력사용량
    # 1. 총 전력사용량

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">총 전력사용량</div>
            <div class="kpi-value">{total_power:,.0f}</div>
            <div class="kpi-unit">kWh</div>
        </div>
        """, unsafe_allow_html=True)

   

    # 2. 총 전기요금
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">총 전기요금</div>
            <div class="kpi-value">{total_cost:,.0f}</div>
            <div class="kpi-unit">원</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. 총 탄소배출량
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">총 탄소배출량</div>
            <div class="kpi-value">{total_carbon:,.0f}</div>
            <div class="kpi-unit">CO2[Kg]</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. 가동일 / 휴무일 (통합)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">가동일 / 휴무일</div>
            <div class="kpi-value">{total_working_days:,} / {total_holiday_days:,}</div>
            <div class="kpi-unit">일</div>
        </div>
        """, unsafe_allow_html=True)


    # 5. 역률 조정 금액 (새로운 KPI)
    # 델타 및 색상 로직 설정
    if total_pf_adjustment < 0:
        pf_title = "역률 감액 (절감)"
        pf_value = f"{abs(total_pf_adjustment):,.0f}"
        pf_unit = "원 (절감)"
        pf_color_style = "border-left: 5px solid #00b050;" # 감액(절감)은 녹색
    elif total_pf_adjustment > 0:
        pf_title = "역률 패널티 (추가)"
        pf_value = f"{total_pf_adjustment:,.0f}"
        pf_unit = "원 (추가)"
        pf_color_style = "border-left: 5px solid #ff7f0e;" # 패널티(추가)는 주황색
    else:
        pf_title = "역률 조정금액"
        pf_value = "0"
        pf_unit = "원"
        pf_color_style = "border-left: 5px solid #1f77b4;" # 기본 색상

   

    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="{pf_color_style}">
            <div class="kpi-title">{pf_title}</div>
            <div class="kpi-value">{pf_value}</div>
            <div class="kpi-unit">{pf_unit}</div>
        </div>
        """, unsafe_allow_html=True)

   

    st.divider()

    # -------------------------------------------------------------
    # ⭐⭐ 3. 그래프 표시 (기존 코드 유지) ⭐⭐
    # -------------------------------------------------------------
    
    col_monthly_trend, col_monthly_comp = st.columns(2)

    # ===============================================================
    # 2-1. 좌측 그래프: 월별 전력사용량 + 월 평균 전기요금 추이 (전체 기간)
    # ===============================================================
    with col_monthly_trend:
        st.subheader("월별 전력사용량 및 평균 요금 추이")
        # ... (그래프 생성 및 표시 로직 유지) ...
        
        # 월별 전체 데이터 집계 (전체 df 사용)
        monthly = get_monthly_all_data(df)

        # -------------------------------------------------------------
        # ⭐ 수정된 로직: 선택된 월에 따른 막대 색상 결정
        # -------------------------------------------------------------
        bar_colors = []
        
        if selected_month == "전체":
            bar_colors = ['#1f77b4'] * len(monthly)
        else:
            try:
                selected_month_int = int(selected_month) 
            except ValueError:
                selected_month_int = -1 

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
        
        fig_monthly.update_xaxes(showgrid=False)
        fig_monthly.update_yaxes(title_text="전력사용량 (kWh)", secondary_y=False, showgrid=False)
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
    
    # ... (월평균/전월 데이터 계산 로직 유지) ...
    monthly_totals_all = df.groupby('month')['전력사용량(kWh)'].sum()
    annual_monthly_avg_power = monthly_totals_all.mean()
    selected_period_total_power = filtered_df['전력사용량(kWh)'].sum()
    
    with col_monthly_comp:
        st.subheader("총 전력사용량 비교")
        
        comp_labels = [label, '2024년 월평균']
        comp_values = [selected_period_total_power, annual_monthly_avg_power]
        comp_colors = {label: '#1f77b4', '2024년 월평균': 'lightgray'}
        comp_title = '총 전력사용량 (kWh)'
        category_order = ['2024년 월평균'] 

        if isinstance(selected_month, int) and selected_month > df['month'].min():
            prev_month = selected_month - 1
            prev_month_total_power = monthly_totals_all.get(prev_month, 0)
            prev_label = f'{prev_month}월 (전월)'
            comp_labels.append(prev_label)
            comp_values.append(prev_month_total_power)
            comp_colors[prev_label] = '#ffb366' 
            category_order.append(prev_label)

        category_order.append(label)

        comp_data = pd.DataFrame({'구분': comp_labels, comp_title: comp_values})

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
        
        fig_comp.update_traces(
            texttemplate='%{text:,.0f} kWh', 
            textposition='outside',
            textfont=dict(color='black')
        )
        
        fig_comp.update_xaxes(
            categoryorder='array', 
            categoryarray=category_order, 
            tickfont=dict(color='black') 
        )
        
        max_val = comp_data[comp_title].max() if not comp_data.empty else 1
        
        fig_comp.update_layout(
            height=450,
            showlegend=False,
            xaxis_title="",
            yaxis_title=comp_title,
            yaxis_range=[0, max_val * 1.2],
            font=dict(color='black') 
        )
        
        fig_comp.update_yaxes(tickfont=dict(color='black')) 

        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

# ----------------- 2. 월별 분석 (추이 및 비교) -----------------




# ----------------- 탭 2: 역률 관리도 -----------------
with tab2:
    st.markdown("")
    st.caption("")

    

    # # 연간 일평균 계산 (전체 데이터)
    # annual_daily_avg = df.groupby(df['측정일시'].dt.date)['전력사용량(kWh)'].sum().mean()

    # # 선택 기간의 일평균 계산 (filtered_df 사용)
    # selected_month_daily_sum = filtered_df.groupby(filtered_df['측정일시'].dt.date)['전력사용량(kWh)'].sum()
    # selected_month_daily_avg = selected_month_daily_sum.mean() if not selected_month_daily_sum.empty else 0
    
    
    # st.markdown("##### 🔍 월별 분석 결과 요약:")
    # st.caption(f"월별 추이 그래프는 계절적 요인(예: 여름철 냉방)에 따른 사용량 변화를 보여줍니다. 선택된 **{label}**의 일평균 사용량({selected_month_daily_avg:,.0f} kWh)은 연간 일평균({annual_daily_avg:,.0f} kWh)과 비교하여 현재 사용 수준을 가늠할 수 있습니다.")
    # st.divider()
    
    # ----------------- 3. 일별 분석 (전력 및 요금) -----------------
    st.header("일별 사용량 및 일별 전기 요금 분석")
    col_daily_power, col_daily_cost = st.columns(2)
    
    # ===============================================================
    # 3-1. 좌측 그래프: 일별 전력량 분석 (Stacked Bar) (유지)
    # ===============================================================
    with col_daily_power:
        st.subheader("일별 전력량 분석")
        
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
        st.subheader("일별 총 전기요금 추이 (원)")
    
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
    st.header("시간대별 패턴 분석")
    col_hourly_pattern, col_hourly_load = st.columns(2)
    
    # ===============================================================
    # 4-1. 좌측 그래프: 시간대별 전력 사용량 패턴 (기존 좌측 그래프)
    # ===============================================================
    
    with col_hourly_pattern:
        st.subheader("시간대별 전력 사용량 패턴 (평균/최소/최대)")
        
        # 시간대별 집계
        hourly = filtered_df.groupby('hour').agg({
            '전력사용량(kWh)': ['mean', 'min', 'max']
        }).reset_index()
        hourly.columns = ['hour', 'avg', 'min', 'max']
        
        # 시간 구간 정의
        time_zones = [
            {'name': '야간', 'start': 0, 'end': 8.25, 'color': 'rgba(150, 150, 180, 0.1)'},
            {'name': '가동준비', 'start': 8.25, 'end': 9, 'color': 'rgba(255, 200, 100, 0.15)'},
            {'name': '오전생산', 'start': 9, 'end': 12, 'color': 'rgba(100, 200, 150, 0.15)'},
            {'name': '점심시간', 'start': 12, 'end': 13, 'color': 'rgba(255, 180, 150, 0.15)'},
            {'name': '오후생산', 'start': 13, 'end': 17.25, 'color': 'rgba(100, 200, 150, 0.15)'},
            {'name': '퇴근시간', 'start': 17.25, 'end': 18.5, 'color': 'rgba(255, 200, 100, 0.15)'},
            {'name': '야간초입', 'start': 18.5, 'end': 21, 'color': 'rgba(180, 180, 200, 0.1)'},
            {'name': '야간', 'start': 21, 'end': 24, 'color': 'rgba(150, 150, 180, 0.1)'}
        ]
        
        # 그래프 생성
        fig_hourly = go.Figure()
        
        # 구간별 배경 및 라벨 표시
        max_y = hourly['avg'].max() * 1.1  # 라벨 위치를 위한 최대값 계산
        
        for zone in time_zones:
            # 배경색
            fig_hourly.add_vrect(
                x0=zone['start'], x1=zone['end'],
                fillcolor=zone['color'],
                layer="below", line_width=0
            )
            
            # 구간 라벨 (상단에 표시)
            mid_point = (zone['start'] + zone['end']) / 2
            fig_hourly.add_annotation(
                x=mid_point,
                y=max_y,
                text=zone['name'],
                showarrow=False,
                font=dict(size=11, color='gray'),
                yshift=10
            )
        
        # 평균 라인 표시
        fig_hourly.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['avg'], 
            mode='lines+markers', 
            name='평균 전력사용량',
            line=dict(color='#1f77b4', width=3), 
            marker=dict(size=7, color='#1f77b4'),
            customdata=list(zip(hourly['min'], hourly['max'])),
            hovertemplate='<b>%{x}:00시</b><br>' +
                          '평균: %{y:.1f} kWh<br>' +
                          '최소: %{customdata[0]:.1f} kWh<br>' +
                          '최대: %{customdata[1]:.1f} kWh<extra></extra>'
        ))
        
        fig_hourly.update_layout(
            height=550,
            xaxis_title='시간', 
            yaxis_title='전력사용량 (kWh)', 
            font=dict(color='black'),
            xaxis=dict(
                tickfont=dict(color='black'),
                tickmode='array',
                tickvals=list(range(0, 25, 2)),
                ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)],
                range=[-0.5, 24]
            ),
            yaxis=dict(
                tickfont=dict(color='black'),
                range=[0, max_y * 1.15]  # 라벨 공간 확보
            ),
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # ===============================================================
    # 4-2. 우측 그래프: 시간대별 부하 발생 빈도 (기존 좌측 그래프)
    # ===============================================================
    with col_hourly_load:
        
        # 툴팁 내용 정의 - HTML 이스케이프 처리
        tooltip_content = """
    [공장 부하 패턴 정의]
    
    1. 🏖️ 휴무일 (가동 쉬는 날): 전체 시간대 경부하
    
    2. 🏭 가동일 (운영 시간대)
      • 봄/여름/가을 (3월-10월) 최대부하: 10:00-12:00, 13:00-17:00
      • 겨울철 (11월-2월) 최대부하: 10:00-12:00, 17:00-20:00, 22:00-23:00
      • 경부하 구간: 23:00 - 09:00
        """
        
        # HTML 특수문자 이스케이프
        tooltip_content_escaped = tooltip_content.replace('<', '&lt;').replace('>', '&gt;')
        
        # CSS 스타일 정의
        st.markdown("""
        <style>
        .tooltip-container {
            position: relative;
            display: inline-block;
        }
        .tooltip-icon {
            cursor: help;
            color: #1f77b4;
            font-size: 20px;
            margin-left: 8px;
            vertical-align: middle;
        }
        .tooltip-container .tooltip-text {
            visibility: hidden;
            width: 400px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 15px;
            position: absolute;
            z-index: 1000;
            top: 100%;
            left: 50%;
            margin-left: -200px;
            margin-top: 10px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-line;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        .title-with-tooltip {
            display: flex;
            align-items: center;
            margin-top: 0px;
            margin-bottom: 1rem;
        }
        .title-with-tooltip h3 {
            margin: 0;
            display: inline;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 제목과 툴팁 아이콘을 같은 줄에 배치
        st.markdown(f"""
        <div class="title-with-tooltip">
            <h3>시간대별 부하 발생 빈도</h3>
            <div class="tooltip-container">
                <span class="tooltip-icon">ⓘ</span>
                <span class="tooltip-text">{tooltip_content_escaped}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # -------------------------------------------------------------
        # 그래프 로직 시작
        # -------------------------------------------------------------
    
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
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, max_val * 1.1],
                        tickfont=dict(color='black')
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='black'),
                        direction='clockwise', 
                        rotation=90,           
                        dtick=3                
                    )
                ),
                height=550, 
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(color='black')
            )
            st.plotly_chart(fig_polar, use_container_width=True)
            st.caption(f"📌 선택한 기간 내 **선택 부하 유형** 총 발생 건수: **{total_count:,}건**")
    
    st.markdown("##### 🔍 시간대별 분석 결과 요약:")
    st.caption("시간대별 전력 패턴은 하루 중 설비 가동 시간 및 피크 시간대를 파악하는 데 중요합니다. 부하 발생 빈도를 극좌표 차트로 확인하여 전력 품질에 영향을 줄 수 있는 특정 시간대(예: 최대 부하의 집중 시간)를 시각적으로 분석할 수 있습니다.")
    st.divider()




# ----------------- 탭 3: 역률 관리 및 비생산 전력 낭비-----------------
# ... (상위 코드 및 탭 1, 탭 2 블록 유지) ...

# ============================================================================
# ----------------- 탭 3: 역률 관리 및 비생산 전력 낭비 -----------------
# ============================================================================
with tab3:
    st.markdown("### 역률 관리 및 비생산 전력 낭비 분석")
    st.caption("선택된 기간에 대해 역률 일일 사이클을 분석하고, 휴무일 기준선 대비 비생산 시간대 전력 낭비 규모를 탐지합니다.")
    st.markdown("---")

    # ⭐ NEW: 탭 3 전용 상세 기간 필터 ⭐
    # 사이드바의 min/max 날짜를 사용합니다.
    min_date_tab3 = df['측정일시'].min().date()
    max_date_tab3 = df['측정일시'].max().date()
    
    # 기본값은 전체 기간으로 설정
    tab3_date_range = st.date_input(
        "분석할 상세 기간을 지정하세요 (탭 3 전용)",
        value=(min_date_tab3, max_date_tab3),
        min_value=min_date_tab3,
        max_value=max_date_tab3,
        key='tab3_date_filter'
    )
    
    # --- 탭 3 데이터 준비: 필터 적용 ---
    
    # 1. 탭 3 기간 필터 적용
    if len(tab3_date_range) == 2:
        start_date_tab3 = tab3_date_range[0].strftime('%Y-%m-%d')
        end_date_tab3 = tab3_date_range[1].strftime('%Y-%m-%d')
        
        # filtered_df (사이드바의 월, 작업휴무 필터 적용됨)를 복사한 후, 
        # 탭 3의 기간 필터를 추가로 적용합니다.
        analysis_df = filtered_df[
            (filtered_df['date'] >= start_date_tab3) & 
            (filtered_df['date'] <= end_date_tab3)
        ].copy()
    else:
        st.warning("⚠️ 유효한 기간이 선택되지 않았습니다.")
        st.stop()


    # -------------------------------------------------------------
    # ⭐ 역률 관리 섹션 ⭐
    # -------------------------------------------------------------
    st.subheader("역률 일일 사이클 분석")

    pf_colors = { '가동': '#1f77b4', '휴무': '#ff7f0e' }
    
    # 1. 일일 사이클 집계 및 데이터 준비 (analysis_df 사용)
    cycle_df = analysis_df.copy()

    if not selected_work_status:
        st.warning("⚠️ 사이드바에서 '작업 상태 선택' 필터를 통해 최소한 '가동' 또는 '휴무'를 선택해야 합니다.")
        st.stop() 
        
    cycle_df['time_15min'] = ((cycle_df['hour'] * 60 + cycle_df['minute']) // 15) * 15 
    cycle_df['time_label'] = cycle_df['time_15min'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")

    daily_cycle = cycle_df.groupby(['작업휴무', 'time_15min', 'time_label']).agg(
        avg_lag_pf=('지상역률(%)', 'mean'),
        avg_lead_pf=('진상역률(%)', 'mean')
    ).reset_index().sort_values('time_15min')

    all_time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    col_lag, col_lead = st.columns(2)

    # 2-1. 지상역률 (Lagging PF) 일일 사이클 차트 (fig_lag)
    with col_lag:
        st.markdown("#### 🟢 지상역률(%) 일일 사이클 (추가/감액 기준: 90%)")
        fig_lag = go.Figure()
        
        # KEPCO 규정 시간 배경 (09:00 ~ 22:00)
        fig_lag.add_vrect(x0="09:00", x1="22:00", fillcolor="yellow", opacity=0.15, layer="below", line_width=0,
                         annotation_text="KEPCO 규제 시간 (09시~22시)", annotation_position="top left")
        
        for status in selected_work_status:
            df_plot = daily_cycle[daily_cycle['작업휴무'] == status]
            fig_lag.add_trace(go.Scatter(x=df_plot['time_label'], y=df_plot['avg_lag_pf'], mode='lines', name=f'{status}', line=dict(color=pf_colors.get(status, 'gray'), width=2)))

        fig_lag.add_hline(y=90, line_dash="dash", line_color="red", line_width=2, annotation_text="요금제 기준선 (90%)", annotation_position="bottom right", name="요금제 기준선 (90%)")
        
        fig_lag.update_layout(height=500, xaxis=dict(title="시간 (Hour, 15분 단위)", categoryorder='array', categoryarray=all_time_labels, tickvals=[f"{h:02d}:00" for h in range(24)], ticktext=[f"{h}" for h in range(24)], tickangle=0, tickfont=dict(color='black')),
                              yaxis=dict(title="평균 지상역률(%)", range=[40, 102], tickfont=dict(color='black')), legend=dict(title='작업휴무', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50))
        st.plotly_chart(fig_lag, use_container_width=True)


    # 2-2. 진상역률 (Leading PF) 일일 사이클 차트 (fig_lead)
    with col_lead:
        st.markdown("#### 🔴 진상역률(%) 일일 사이클 (추가 요금 기준: 95%)")
        fig_lead = go.Figure()

        # KEPCO 규정 시간 배경 (22시~09시, 야간)
        fig_lead.add_vrect(x0="22:00", x1="23:45", fillcolor="orange", opacity=0.15, layer="below", line_width=0)
        fig_lead.add_vrect(x0="00:00", x1="09:00", fillcolor="orange", opacity=0.15, layer="below", line_width=0,
                           annotation_text="KEPCO 규제 시간 (22시~09시)", annotation_position="top left")

        for status in selected_work_status:
            df_plot = daily_cycle[daily_cycle['작업휴무'] == status]
            fig_lead.add_trace(go.Scatter(x=df_plot['time_label'], y=df_plot['avg_lead_pf'], mode='lines', name=f'{status}', line=dict(color=pf_colors.get(status, 'gray'), width=2)))

        fig_lead.add_hline(y=95, line_dash="dash", line_color="red", line_width=2, annotation_text="요금제 기준선 (95%)", annotation_position="bottom right", name="요금제 기준선 (95%)")
        
        fig_lead.update_layout(height=500, xaxis=dict(title="시간 (Hour, 15분 단위)", categoryorder='array', categoryarray=all_time_labels, tickvals=[f"{h:02d}:00" for h in range(24)], ticktext=[f"{h}" for h in range(24)], tickangle=0, tickfont=dict(color='black')),
                              yaxis=dict(title="평균 진상역률(%)", range=[0, 102], tickfont=dict(color='black')), legend=dict(title='작업휴무', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50))
        st.plotly_chart(fig_lead, use_container_width=True)

    st.markdown("##### 분석 결과 요약:")
    st.caption("이 차트는 선택된 기간과 작업휴무 조건에 따른 평균 역률 패턴을 보여줍니다. 지상역률은 90% 미만, 진상역률은 95% 초과 시 요금에 영향을 줄 수 있습니다.")
    st.divider()


    # -------------------------------------------------------------
    # ⭐ 비생산 전력 낭비 분석 섹션 ⭐
    # -------------------------------------------------------------

    st.header("공장 운영 패턴 분석: 비생산 시간대 비효율 탐지")
    st.caption("LS 공장 운영 스케줄에 따라 **비생산 시간대(야간초입, 야간)** 의 전력 사용 패턴을 분석하여 **낭비가 발생하는 날/주**를 식별합니다.")

    # 1. 시간대 분류 및 비효율 지표 계산 (analysis_df_tab3 사용)
    # analysis_df_tab3는 이미 탭 3 필터 기간이 적용된 analysis_df입니다.
    
    def classify_time_zone(hour, minute):
        """시간을 LS 공장 운영 패턴에 따라 분류"""
        time_decimal = hour + minute / 60

        if (8.25 <= time_decimal < 9):
            return '가동준비'
        elif (9 <= time_decimal < 12):
            return '오전생산'
        elif (12 <= time_decimal < 13):
            return '점심시간'
        elif (13 <= time_decimal < 17.25):
            return '오후생산'
        elif (17.25 <= time_decimal < 18.5):
            return '퇴근시간'
        elif (18.5 <= time_decimal < 21):
            return '야간초입'
        else:
            return '야간'

    non_production_zones = ['야간초입', '야간']
    analysis_df_tab3['시간대구분'] = analysis_df_tab3.apply(
        lambda row: classify_time_zone(row['hour'], row['minute']), axis=1
    )
    analysis_df_tab3['생산구분'] = analysis_df_tab3['시간대구분'].apply(
        lambda x: '비생산시간' if x in non_production_zones else '생산시간'
    )

    daily_analysis = analysis_df_tab3.groupby([analysis_df_tab3['측정일시'].dt.date, '작업휴무', '생산구분']).agg({'전력사용량(kWh)': 'sum'}).reset_index()
    daily_analysis.columns = ['날짜', '작업휴무', '생산구분', '전력사용량']

    daily_pivot = daily_analysis.pivot_table(index=['날짜', '작업휴무'], columns='생산구분', values='전력사용량', fill_value=0).reset_index()

    if not daily_pivot[daily_pivot['작업휴무'] == '휴무'].empty:
        holiday_baseline = daily_pivot[daily_pivot['작업휴무'] == '휴무']['비생산시간'].mean()
    else:
        holiday_baseline = daily_pivot['비생산시간'].mean() * 0.5 if not daily_pivot.empty else 1 

    working_days = daily_pivot[daily_pivot['작업휴무'] == '가동'].copy()
    working_days['비효율지수'] = working_days['비생산시간'] - holiday_baseline
    working_days['비효율율(%)'] = (working_days['비효율지수'] / holiday_baseline * 100).round(1)
    working_days['날짜_str'] = pd.to_datetime(working_days['날짜']).dt.strftime('%m-%d')
    working_days['주차'] = pd.to_datetime(working_days['날짜']).dt.isocalendar().week
    
    # KPI 계산을 위해 diff_avg 재계산
    avg_working_non_prod = working_days['비생산시간'].mean()
    diff_avg = avg_working_non_prod - holiday_baseline
    
    
    # -----------------------------------------------------------------
    # === KPI 및 그래프 표시 (기존 코드 유지) ===
    # -----------------------------------------------------------------
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    # KPI 1: 휴무일 기준선
    with col_kpi1:
        st.metric(
            label="휴무일 비생산시간 기준선",
            value=f"{holiday_baseline:,.0f} kWh",
            delta="최소 유지 부하"
        )
    
    # KPI 2: 가동일 평균
    if not working_days.empty:
        with col_kpi2:
            st.metric(
                label="가동일 비생산시간 평균",
                value=f"{avg_working_non_prod:,.0f} kWh",
                delta=f"+{diff_avg:,.0f} kWh ({(diff_avg/holiday_baseline*100):.1f}%)",
                delta_color="inverse"
            )
        
        # KPI 3: 총 낭비 전력
        total_waste = working_days['비효율지수'].sum()
        with col_kpi3:
            st.markdown(
                f"""
                <div style='background-color:#ffebeb; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
                    <div style='font-size: 16px; color: #581515;'>🚨 총 낭비 전력 (잠재적)</div>
                    <div style='font-size: 32px; font-weight: bold; color: #FF4B4B;'>{total_waste:,.0f}</div>
                    <div style='font-size: 14px; color: #FF4B4B;'>kWh</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("선택 기간에 가동일 데이터가 없어 비효율 KPI를 계산할 수 없습니다.", icon="ℹ️")
        st.stop() 
    
    st.markdown("---")
    
    # 메인 그래프: 일별 비효율 막대 (Top 15)
    st.subheader("일별 비생산시간 비효율 지수")

    display_days = working_days.nlargest(15, '비효율지수')
    
    def get_color(value, baseline):
        if value > baseline * 0.5:
            return '#d32f2f' 
        elif value > baseline * 0.3:
            return '#ff6f00' 
        elif value > baseline * 0.1:
            return '#ffa726' 
        else:
            return '#ffcc80' 

    colors = [get_color(x, holiday_baseline) for x in display_days['비효율지수']]

    fig_main = go.Figure()
    fig_main.add_trace(go.Bar(
        x=display_days['날짜_str'],
        y=display_days['비효율지수'],
        marker_color=colors,
        text=display_days['비효율율(%)'].apply(lambda x: f"+{x}%" if x > 0 else f"{x}%"),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' + '비효율: %{y:,.0f} kWh<br>' + '비율: %{text}<extra></extra>'
    ))

    fig_main.add_hline(y=holiday_baseline * 0.5, line_dash="dash", line_color="red", annotation_text="⚠️ 경고 기준 (+50%)", annotation_position="right")
    fig_main.add_hline(y=holiday_baseline * 0.3, line_dash="dot", line_color="orange", annotation_text="주의 (+30%)", annotation_position="right")

    fig_main.update_layout(height=450, xaxis_title='날짜 (비효율 상위 15일)', yaxis_title='비효율 지수 (kWh)', xaxis=dict(tickangle=-45, type='category', tickfont=dict(size=11)), showlegend=False, plot_bgcolor='white')

    st.plotly_chart(fig_main, use_container_width=True)

    # 하단: 2개 컬럼 (주별 평균 + Top 5 테이블)
    col_weekly, col_table = st.columns([1, 1])

    with col_weekly:
        st.markdown("#### 주별 평균 비효율")
        weekly_waste = working_days.groupby('주차').agg({'비효율지수': 'mean', '비생산시간': 'mean'}).reset_index()

        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Bar(x=weekly_waste['주차'].astype(str) + '주차', y=weekly_waste['비효율지수'], marker_color='#ff7f0e', text=weekly_waste['비효율지수'].apply(lambda x: f"{x:,.0f}"), textposition='outside'))

        fig_weekly.update_layout(height=350, xaxis_title='', yaxis_title='평균 비효율 (kWh)', showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig_weekly, use_container_width=True)

    with col_table:
        st.markdown("#### 비효율 Top 5")
        top5 = working_days.nlargest(5, '비효율지수')[['날짜_str', '비생산시간', '비효율지수', '비효율율(%)']]
        top5.columns = ['날짜', '비생산 사용량 (kWh)', '비효율 (kWh)', '비율 (%)']

        st.dataframe(top5.style.format({'비생산 사용량 (kWh)': '{:,.0f}', '비효율 (kWh)': '{:,.0f}', '비율 (%)': '{:+.1f}%'}).background_gradient(subset=['비효율 (kWh)'], cmap='Reds'), use_container_width=True, hide_index=True, height=280)

    # 분석 요약
    st.markdown("---")
    st.markdown("##### 🔍 분석 결과 요약:")

    high_waste_days = len(working_days[working_days['비효율지수'] > holiday_baseline * 0.5])
    total_days = len(working_days)
    waste_percentage = (high_waste_days / total_days * 100) if total_days > 0 else 0

    st.caption(
        f"**핵심 발견사항:**<br>"
        f"• 전체 {total_days}일 중 **{high_waste_days}일({waste_percentage:.1f}%)**이 경고 기준(+50%) 초과<br>"
        f"• 평균적으로 가동일 비생산시간은 휴무일 대비 **{diff_avg:,.0f} kWh ({(diff_avg/holiday_baseline*100):.1f}%) 더 사용**<br>"
        f"• **개선 방향:** 상위 비효율 날짜를 중심으로 야간(21:00~08:00) 설비 대기전력 점검 필요",
        unsafe_allow_html=True
    )
    st.divider()