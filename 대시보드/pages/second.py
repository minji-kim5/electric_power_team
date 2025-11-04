import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from report import generate_report_from_template

# ============================================================================
# App config
# ============================================================================
st.set_page_config(page_title="전력 데이터 분석", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .kpi-card { background-color:#f0f2f6; padding:20px; border-radius:10px; border-left:5px solid #1f77b4; height:140px; display:flex; flex-direction:column; justify-content:center; }
    .kpi-title { font-size:16px; color:#666; margin-bottom:10px; }
    .kpi-value { font-size:32px; font-weight:bold; color:#1f77b4; margin-bottom:5px; }
    .kpi-unit { font-size:14px; color:#888; }
    </style>
    """,
    unsafe_allow_html=True,
)


# (Streamlit 파일 상단 <style> 블록에 추가)
st.markdown("""
<style>
/* Insight Panel Styles */
.insights-panel-container {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-top: 20px;
}
.insight-item {
    padding: 15px;
    margin-bottom: 15px;
    border-left: 4px solid #667eea; /* 메인 색상 */
    background: #f8f9fa;
    border-radius: 6px;
}
.insight-item:last-child {
    margin-bottom: 0;
}
.insight-title {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 8px;
    font-size: 16px;
}
.insight-text {
    color: #444;
    line-height: 1.6;
    font-size: 14px;
}
.insight-header {
    font-size: 24px;
    font-weight: 600;
    color: #667eea;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Paths
# ============================================================================
DATA_DIR = Path("data_dash")
TRAIN_PATH = DATA_DIR / "train_dash_df.csv"
MONTHLY_PF_PATH = DATA_DIR / "월별 역률 패널티 계산.csv"  # << 고정된 파일명 사용 >>
RATE_PDF = DATA_DIR / "2024년도7월1일시행전기요금표(종합)_출력용.pdf"
TEMPLATE_PATH = Path(r"C:\Users\USER\Desktop\electric_power_-team\대시보드\data_dash\고지서_템플릿.docx")






# ============================================================================
# Data loaders & helpers
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["측정일시"], errors="coerce")
    df = df.assign(
        측정일시=dt,
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
        date=dt.dt.date.astype(str),
    )
    if "단가" in df.columns:
        df = df.dropna(subset=["단가"])  # 안전장치
    return df

@st.cache_data(show_spinner=False)
def get_monthly_all_data(data_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        data_df.groupby("month").agg({"전력사용량(kWh)": "sum", "전기요금(원)": "mean"}).reset_index()
    )
    monthly = monthly[monthly["month"] <= 11]
    monthly["label"] = monthly["month"].apply(lambda x: f"2024-{x:02d}")
    return monthly

@st.cache_data(show_spinner=False)
def load_monthly_pf(path: Path) -> pd.DataFrame:
    try:
        pf = pd.read_csv(path)
        pf["year"] = pf["year"].astype(int)
        pf["month"] = pf["month"].astype(int)
        return pf
    except FileNotFoundError:
        st.error(f"🚨 오류: '{path.name}' 파일을 찾을 수 없습니다. 역률 지표가 0으로 표시됩니다.")
        return pd.DataFrame(columns=["year", "month", "역률_조정금액(원)"])

@st.cache_data(show_spinner=False)
def get_pdf_bytes(path: Path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        st.error(f"🚨 파일을 찾을 수 없습니다: {path}")
        return None

# ============================================================================
# Load data
# ============================================================================
df = load_data(TRAIN_PATH)
monthly_summary_df = load_monthly_pf(MONTHLY_PF_PATH)
pdf_data = get_pdf_bytes(RATE_PDF)

# Precomputed anchors for comparisons
monthly_totals_all = df.groupby("month")["전력사용량(kWh)"].sum()
annual_monthly_avg_power = monthly_totals_all.mean()

# UI tweaks
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size:20px; font-weight:600; }
    .stTabs [data-baseweb="tab-list"] button { padding-top:10px; padding-bottom:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# ===== 사이드바 필터 (개선된 로직) =====
# ==============================================================================
# 1단계: 필터 단위 선택
st.sidebar.markdown(" **분석 단위 선택**")
filter_unit = st.sidebar.radio(
    "분석 단위를 선택하세요",
    ('월별', '일별'),
    index=0 # 기본값: 월별
)

# 2단계: 세부 기간 선택 (조건부)
st.sidebar.markdown("---")
st.sidebar.markdown(" **세부 기간 선택**")

min_date = df['측정일시'].min().date()
max_date = df['측정일시'].max().date()
start_date_str, end_date_str, label = "", "", ""

if filter_unit == '월별':
    # 월별 선택: 전체, 1월, 2월, ... 드롭다운
    sorted_months = sorted(df['month'].unique().tolist())
    month_options = ["전체 기간"] + [f"{m}월" for m in sorted_months]
    selected_month_label = st.sidebar.selectbox(
        "분석 월을 선택하세요",
        options=month_options,
        index=0 # 기본값: 전체 기간
    )
    
    if selected_month_label == "전체 기간":
        # 전체 기간 선택
        filtered_df = df.copy()
        label = "전체 기간"
    else:
        # 특정 월 선택 (예: '1월' -> 1)
        selected_month = int(selected_month_label.replace('월', ''))
        filtered_df = df[df['month'] == selected_month].copy()
        label = f"2024년 {selected_month}월"

elif filter_unit == '일별':
    # 일별 선택: Date Range Picker
    date_range = st.sidebar.date_input(
        "날짜 범위를 지정하세요",
        value=(min_date, max_date), # 기본값: 전체 기간
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # 'date' 컬럼을 기반으로 필터링
        filtered_df = df[(df['date'] >= start_date_str) & (df['date'] <= end_date_str)].copy()
        
        if start_date == min_date and end_date == max_date:
            label = "전체 기간"
        else:
            label = f"{start_date} ~ {end_date}"
    else:
        # 날짜 범위가 완전히 선택되지 않았을 경우 (단일 날짜만 선택된 경우)
        filtered_df = df.copy()
        label = "전체 기간"
        
st.sidebar.markdown("---")
st.sidebar.markdown(" **작업 상태 선택**") # 제목 볼드체

# 3단계: 작업휴무 체크박스 필터 (Multiselect를 태그처럼 사용)
work_status_options = sorted(df['작업휴무'].unique().tolist())

# work_status_options가 비어있지 않은 경우에만 필터 생성
if work_status_options:
    st.sidebar.markdown("작업 여부 선택") # 🌟 "작업 여부 선택" 레이블을 별도로 Markdown으로 표시
    
    selected_work_status = st.sidebar.multiselect(
        "작업 여부 선택 (숨겨진 레이블)", # 실제 레이블은 숨김
        options=work_status_options,
        default=work_status_options,
        label_visibility="collapsed" # 🌟 CSS와 함께 태그만 보이게 함
    )
    
    # 최종 필터링 적용 (작업휴무)
    if selected_work_status:
        # 필터링된 데이터에 한 번 더 작업휴무 필터를 적용
        filtered_df = filtered_df[filtered_df['작업휴무'].isin(selected_work_status)].copy()
        
    if filtered_df.empty:
        st.error("선택된 필터 조건에 해당하는 데이터가 없습니다. 필터를 조정해주세요.")
        st.stop()
    
# 데이터 후처리 (필수)
filtered_df['측정일시'] = pd.to_datetime(filtered_df['측정일시'], errors='coerce')
if 'month' not in filtered_df.columns:
    filtered_df['month'] = filtered_df['측정일시'].dt.month

# 숫자열 정리
for c in ["전력사용량(kWh)", "전기요금(원)"]:
    if c in filtered_df.columns:
        filtered_df[c] = pd.to_numeric(filtered_df[c], errors="coerce").fillna(0)

# 고지서 생성
word_file_data = generate_report_from_template(filtered_df, str(TEMPLATE_PATH))

# ============================================================================
# Header & downloads
# ============================================================================
head_title, _, _, _ = st.columns([0.6, 0.13, 0.13, 0.14])
with head_title:
    st.title("📊 LS ELECTRIC 청주 공장 전력 사용 현황")

monthly_download_data = get_monthly_all_data(df)
csv_monthly = monthly_download_data.to_csv(index=False, encoding="utf-8-sig")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⬇️ 파일 다운로드")

if word_file_data:
    try:
        mm = int(filtered_df["month"].iloc[0])
    except Exception:
        mm = 0
    st.sidebar.download_button(
        label="📄 고지서 다운로드",
        data=word_file_data,
        file_name=f"LS일렉트릭_전기요금_고지서_{mm:02d}월.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key="bill_sidebar_docx",
        use_container_width=True,
        help="선택 기간의 데이터가 반영된 워드 고지서입니다.",
    )
else:
    st.sidebar.warning("⚠️ 고지서 파일 생성 준비 중...")

if pdf_data:
    st.sidebar.download_button(
        label="📑 요금표 다운로드 (PDF)",
        data=pdf_data,
        file_name="2024년_전기요금표.pdf",
        mime="application/pdf",
        key="rate_sidebar",
        use_container_width=True,
    )

# ============================================================================
# Tabs
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["월별 시각화", "일별 시각화", "역률 관리", "공회전 에너지 분석"])
monthly_totals_all_series = df.groupby('month')['전력사용량(kWh)'].sum()
annual_monthly_avg_power = monthly_totals_all_series.mean()

# 1. 사이드바 필터 결과 분석
unique_months = filtered_df['month'].unique()
if len(unique_months) == 1 and filter_unit == '월별':
    # 사이드바에서 특정 단일 월이 '월별' 단위로 선택됨
    current_month_num = unique_months[0]
else:
    # 전체 기간, 일별 범위, 또는 다수 월이 선택됨
    current_month_num = None
    
# 현재 기간의 라벨 (사이드바에서 설정된 'label' 변수 사용)
current_label_for_comp = label 

# 전월 데이터가 필요한 경우에 대비해 현재 기간의 월을 저장 (단일 월이 아닌 경우 None)
current_month_int = current_month_num

# ============================================================================
# Tab 1. 월별 시각화  (월 선택 필터를 '총 전력사용량 비교' 옆으로 이동)
# ============================================================================
with tab1:
    # ── 헤더만 표시
    # ▼ KPI (filtered_df 기준)
    total_power = filtered_df["전력사용량(kWh)"].sum()
    # ... (나머지 KPI 계산 로직 유지) ...
    total_cost = filtered_df["전기요금(원)"].sum()
    total_carbon = (filtered_df.get("탄소배출량(tCO2)", pd.Series(dtype=float)).sum()) * 1000
    total_working_days = filtered_df[filtered_df["작업휴무"] == "가동"]["date"].nunique()
    total_holiday_days = filtered_df[filtered_df["작업휴무"] == "휴무"]["date"].nunique()

    filtered_months = filtered_df[["year", "month"]].drop_duplicates()
    monthly_summary_filtered = monthly_summary_df.merge(filtered_months, on=["year", "month"], how="inner")
    total_pf_adjustment = (
        monthly_summary_filtered["역률_조정금액(원)"].sum().round(0).astype(int)
        if not monthly_summary_filtered.empty else 0
    )

    st.markdown(f"## 📅 {label} 주요 지표")
    st.markdown(
        f"**데이터 기간**: {filtered_df['측정일시'].min().strftime('%Y-%m-%d')} ~ "
        f"{filtered_df['측정일시'].max().strftime('%Y-%m-%d')}"
    )

    # ... (KPI 카드 표시 로직 유지) ...
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card"><div class="kpi-title">총 전력사용량</div>
        <div class="kpi-value">{total_power:,.0f}</div><div class="kpi-unit">kWh</div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card"><div class="kpi-title">총 전기요금</div>
        <div class="kpi-value">{total_cost:,.0f}</div><div class="kpi-unit">원</div></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card"><div class="kpi-title">총 탄소배출량</div>
        <div class="kpi-value">{total_carbon:,.0f}</div><div class="kpi-unit">CO2[Kg]</div></div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card"><div class="kpi-title">가동일 / 휴무일</div>
        <div class="kpi-value">{total_working_days:,} / {total_holiday_days:,}</div><div class="kpi-unit">일</div></div>
        """, unsafe_allow_html=True)

    if total_pf_adjustment < 0:
        pf_title, pf_value, pf_unit, pf_style = "역률 감액 (절감)", f"{abs(total_pf_adjustment):,.0f}", "원 (절감)", "border-left: 5px solid #00b050;"
    elif total_pf_adjustment > 0:
        pf_title, pf_value, pf_unit, pf_style = "역률 패널티 (추가)", f"{total_pf_adjustment:,.0f}", "원 (추가)", "border-left: 5px solid #ff7f0e;"
    else:
        pf_title, pf_value, pf_unit, pf_style = "역률 조정금액", "0", "원", "border-left: 5px solid #1f77b4;"

    with c5:
        st.markdown(f"""
        <div class="kpi-card" style="{pf_style}">
          <div class="kpi-title">{pf_title}</div>
          <div class="kpi-value">{pf_value}</div>
          <div class="kpi-unit">{pf_unit}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── 두 그래프 영역
    col_monthly_trend, col_monthly_comp = st.columns(2)

# 1) 왼쪽 그래프: 월별 추이 (선택 월 강조)
with col_monthly_trend:
    st.subheader("월별 전력사용량 및 평균 요금 추이")
    
    # get_monthly_all_data(df)는 전체 월별 데이터를 반환한다고 가정
    monthly = get_monthly_all_data(df) 
    
    # 🏆 X축 라벨 한글화 (월 번호 사용)
    # monthly['month'] 컬럼이 월 번호를 가지고 있다고 가정
    x_labels_kr = [f"{m}월" for m in monthly["month"]]

    # 🏆 사이드바에서 선택된 월을 강조
    sel = current_month_int
    bar_colors = [
        "#1f77b4" if (sel is not None and m == sel) else "lightgray"
        for m in monthly["month"]
    ]

    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. 막대 그래프: 전력사용량 (월별 합계)
    fig_monthly.add_trace(
        go.Bar(
            # 🏆 X축에 한글 라벨 적용
            x=x_labels_kr, 
            y=monthly["전력사용량(kWh)"], 
            name="월별 사용량",
            marker_color=bar_colors,
            # 🏆 막대 위에 숫자 추가
            text=monthly["전력사용량(kWh)"].apply(lambda x: f"{x:,.0f} kWh"),
            textposition='outside', # 막대 위에 표시
            textfont=dict(color='black', size=30), # 숫자 글씨색 검정으로 지정
        ),
        secondary_y=False,
    )
    
    # 2. 꺾은선 그래프: 평균 전기요금
    fig_monthly.add_trace(
        go.Scatter(
            # 🏆 X축에 한글 라벨 적용 (막대와 동일한 X축을 공유해야 함)
            x=x_labels_kr, 
            y=monthly["전기요금(원)"], 
            name="월 평균 전기요금",
            mode="lines+markers", 
            line=dict(color="#d62728", width=3), # 🏆 선 굵기 및 색상 유지/강조
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    
    # 🏆 X축/Y축 텍스트 및 Title 색상/크기 확대 적용
    axis_font_size = 18 # 축 라벨 크기
    title_font_size = axis_font_size + 2 # 축 Title 크기

    # X축 설정: 한글 라벨, 검정색, 그리드 제거
    fig_monthly.update_xaxes(
        showgrid=False, 
        tickfont=dict(color='black', size=axis_font_size),
        title_font=dict(color='black', size=title_font_size) # X축 Title 색상
    )
    
    # Y1축 설정: 전력사용량 (좌측)
    fig_monthly.update_yaxes(
        title_text="전력사용량 (kWh)", 
        secondary_y=False, 
        showgrid=False, 
        tickfont=dict(color='black', size=axis_font_size), 
        title_font=dict(color='black', size=title_font_size) # 🏆 Y축 Title 색상 검정색
    )
    
    # Y2축 설정: 평균 전기요금 (우측)
    fig_monthly.update_yaxes(
        title_text="평균 전기요금 (원)", 
        secondary_y=True, 
        showgrid=False, 
        tickfont=dict(color='black', size=axis_font_size), 
        title_font=dict(color='black', size=title_font_size) # 🏆 Y축 Title 색상 검정색
    )

    fig_monthly.update_layout(
        height=450, 
        # font=dict(color="black"), # 전체 기본 폰트 색상
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

# 2) 오른쪽 그래프: 총 전력사용량 비교 (현재 기간, 전월, 월평균)
with col_monthly_comp:
    st.subheader("총 전력사용량 비교")
    
    current_total_power = total_power
    current_label = current_label_for_comp

    comp_labels = [current_label, "2024년 월평균"]
    comp_values = [current_total_power, annual_monthly_avg_power]
    comp_colors = {current_label: "#1f77b4", "2024년 월평균": "lightgray"}
    category_order = ["2024년 월평균"]

    # ── 전월 값 계산 (단일 월이 선택되었을 때만 비교)
    if current_month_int is not None:
        prev_month_num = current_month_int - 1
        # 전월이 데이터 범위 내에 있는지 확인
        if prev_month_num in monthly_totals_all_series.index: 
            prev_val = monthly_totals_all_series.get(prev_month_num, 0)
            prev_label = f"{prev_month_num}월 (전월)"
            comp_labels.append(prev_label)
            comp_values.append(prev_val)
            comp_colors[prev_label] = "#ff7f0e"
            category_order.append(prev_label)
    
    category_order.append(current_label)

    comp_df = pd.DataFrame({"구분": comp_labels, "총 전력사용량 (kWh)": comp_values})
    fig_comp = px.bar(
        comp_df, x="구분", y="총 전력사용량 (kWh)",
        color="구분", color_discrete_map=comp_colors, text="총 전력사용량 (kWh)",
        title="선택 기간/월 총 전력사용량 비교",
    )
    fig_comp.update_traces(texttemplate="%{text:,.0f} kWh", textposition="outside", textfont_color="black", textfont_size=20)
    max_val = comp_df["총 전력사용량 (kWh)"].max() or 1

    # 🏆 X축/Y축 텍스트 크기 확대 적용
    axis_font_size = 18
    title_font_size = axis_font_size + 2 # Title 폰트 크기 변수 재사용

    fig_comp.update_layout(
        height=450, showlegend=False, xaxis_title="", yaxis_title="총 전력사용량 (kWh)",
        yaxis_range=[0, max_val * 1.2], 
        font_color="black", # 전체 폰트 색상을 'black'으로 유지 (필요 시)
        # 🏆 X축 설정
        xaxis=dict(showgrid=False, categoryorder="array", categoryarray=category_order, 
                    tickfont=dict(color='black', size=axis_font_size), 
                    title_font=dict(color='black', size=title_font_size)), # X축 Title 색상 추가
        # 🏆 Y축 설정 (여기에 Title 색상 추가!)
        yaxis=dict(showgrid=False, 
                    tickfont=dict(color='black', size=axis_font_size), 
                    title_font=dict(color='black', size=title_font_size)), # 🌟 Y축 Title 색상을 검정색으로 수정
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")


axis_style = dict(
    tickfont=dict(color='black', size=16),  # 눈금 라벨 (예: 1월, 2월, 100kWh)
    title_font=dict(color='black', size=16) # 축 제목 (예: 전력사용량 (kWh))
)

# --- 폰트 크기 변수 설정 ---
# 축 라벨, 제목 폰트 크기
AXIS_FONT_SIZE = 16 
# 막대 위 숫자 폰트 크기
BAR_VALUE_FONT_SIZE = 22 

# 1. 일반 X/Y축 스타일 (Title 포함)
AXIS_STYLE = dict(
    tickfont=dict(color='black', size=AXIS_FONT_SIZE),
    title_font=dict(color='black', size=AXIS_FONT_SIZE)
)

# 2. 🌟 극좌표계 (Polar) 전용 스타일 (Title 관련 속성 제외) 🌟
POLAR_AXIS_STYLE = dict(
    tickfont=dict(color='black', size=AXIS_FONT_SIZE)
)

# ============================================================================
# Tab 2. 일별 시각화
# ============================================================================
with tab2:
    st.header("일별 사용량 및 일별 전기 요금 분석")
    col_daily_power, col_daily_cost = st.columns(2)

    # Left: 일별 전력량 (Stacked by load type)
    with col_daily_power:
        st.subheader("일별 전력량 분석")
        load_map = {"Light_Load": "경부하", "Medium_Load": "중간부하", "Maximum_Load": "최대부하"}
        analysis_df = filtered_df.copy()
        analysis_df["부하타입"] = analysis_df["작업유형"].map(load_map)
        analysis_df["날짜"] = analysis_df["측정일시"].dt.date

        daily = analysis_df.groupby(["날짜", "부하타입"])['전력사용량(kWh)'].sum().reset_index()
        daily_pivot = (
            daily.pivot(index="날짜", columns="부하타입", values="전력사용량(kWh)").fillna(0).reset_index()
        )
        daily_pivot = daily_pivot.sort_values("날짜")
        daily_pivot["날짜_str"] = pd.to_datetime(daily_pivot["날짜"]).dt.strftime("%m-%d")

        colors = {"경부하": "#4CAF50", "중간부하": "#FFC107", "최대부하": "#EF5350"}
        fig_daily = go.Figure()
        for lt in ["경부하", "중간부하", "최대부하"]:
            if lt in daily_pivot.columns:
                fig_daily.add_trace(
                    go.Bar(
                        name=lt,
                        x=daily_pivot["날짜_str"],
                        y=daily_pivot[lt],
                        marker_color=colors[lt],
                        hovertemplate='날짜: %{x}<br>' + lt + ': %{y:,.0f} kWh<extra></extra>',
                    )
                )
        fig_daily.update_layout(
            barmode="stack",
            height=550,
            xaxis_title="날짜",
            yaxis_title="전력사용량 (kWh)",
            font_color="black",
            # 🌟 X축 스타일 적용
            xaxis=dict(showgrid=False, tickangle=-45, type="category", **axis_style),
            # 🌟 Y축 스타일 적용
            yaxis=dict(showgrid=False, **axis_style),
            # 전체 Layout 폰트 크기도 16으로 명시적 설정
            font=dict(color="black", size=16),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16)),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # Right: 일별 전기요금 합계
    with col_daily_cost:
        st.subheader("일별 총 전기요금 추이 (원)")
        daily_cost = (
            filtered_df.groupby(filtered_df["측정일시"].dt.date)["전기요금(원)"].sum().reset_index()
        )
        daily_cost.columns = ["날짜", "총 전기요금(원)"]
        daily_cost["날짜_str"] = pd.to_datetime(daily_cost["날짜"]).dt.strftime("%m-%d")
        fig_cost = px.line(
            daily_cost,
            x="날짜_str",
            y="총 전기요금(원)",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#28a745"],
        )
        fig_cost.update_layout(
            height=550,
            xaxis_title="날짜",
            yaxis_title="총 전기요금 (원)",
            font_color="black",
            # 🌟 X축 스타일 적용
            xaxis=dict(showgrid=False, tickangle=-45, type="category", **axis_style),
            # 🌟 Y축 스타일 적용
            yaxis=dict(showgrid=False, **axis_style),
            # 전체 Layout 폰트 크기도 16으로 명시적 설정
            font=dict(color="black", size=16),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    st.caption(
        "일별 전력량은 부하 유형별 분포를, 전기요금 추이는 TOU 영향으로 비용 급증일을 식별하는 데 유용합니다.")
    st.divider()

    # 시간대 패턴
    st.header("시간대별 패턴 분석")
    col_hourly_pattern, col_hourly_load = st.columns(2)

    # Left: 시간대별 전력 사용량 (평균/최소/최대)
    with col_hourly_pattern:
        st.subheader("시간대별 전력 사용량 패턴 (평균/최소/최대)")
        hourly = (
            filtered_df.groupby("hour").agg({"전력사용량(kWh)": ["mean", "min", "max"]}).reset_index()
        )
        hourly.columns = ["hour", "avg", "min", "max"]

        time_zones = [
            {"name": "야간", "start": 0, "end": 8.25, "color": "rgba(150,150,180,0.1)"},
            {"name": "가동준비", "start": 8.25, "end": 9, "color": "rgba(255,200,100,0.15)"},
            {"name": "오전생산", "start": 9, "end": 12, "color": "rgba(100,200,150,0.15)"},
            {"name": "점심시간", "start": 12, "end": 13, "color": "rgba(255,180,150,0.15)"},
            {"name": "오후생산", "start": 13, "end": 17.25, "color": "rgba(100,200,150,0.15)"},
            {"name": "퇴근시간", "start": 17.25, "end": 18.5, "color": "rgba(255,200,100,0.15)"},
            {"name": "야간초입", "start": 18.5, "end": 21, "color": "rgba(180,180,200,0.1)"},
            {"name": "야간", "start": 21, "end": 24, "color": "rgba(150,150,180,0.1)"},
        ]

        fig_hourly = go.Figure()
        max_y = hourly["avg"].max() * 1.1
        for z in time_zones:
            fig_hourly.add_vrect(x0=z["start"], x1=z["end"], fillcolor=z["color"], layer="below", line_width=0)
            mid = (z["start"] + z["end"]) / 2
            # 🌟 Annotation 폰트 크기 수정
            fig_hourly.add_annotation(x=mid, y=max_y, text=z["name"], showarrow=False, font=dict(size=16, color="gray"), yshift=10)

        fig_hourly.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["avg"],
                mode="lines+markers",
                name="평균 전력사용량",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=7, color="#1f77b4"),
                customdata=list(zip(hourly["min"], hourly["max"])),
                hovertemplate="<b>%{x}:00시</b><br>평균: %{y:.1f} kWh<br>최소: %{customdata[0]:.1f} kWh<br>최대: %{customdata[1]:.1f} kWh<extra></extra>",
            )
        )
        fig_hourly.update_layout(
            height=550,
            xaxis_title="시간",
            yaxis_title="전력사용량 (kWh)",
            font_color="black",
            # 🌟 X축 스타일 적용
            xaxis=dict(tickmode="array", tickvals=list(range(0, 25, 2)), ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)], range=[-0.5, 24], **axis_style),
            # 🌟 Y축 스타일 적용
            yaxis=dict(range=[0, max_y * 1.15], **axis_style),
            font=dict(color="black", size=16),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Right: 시간대별 부하 발생 빈도 (극좌표)
    with col_hourly_load:
        st.markdown(
            """
            <style>
            .tooltip-container{position:relative;display:inline-block}.tooltip-icon{cursor:help;color:#1f77b4;font-size:20px;margin-left:8px;vertical-align:middle}
            .tooltip-container .tooltip-text{visibility:hidden;width:400px;background:#333;color:#fff;text-align:left;border-radius:6px;padding:15px;position:absolute;z-index:1000;top:100%;left:50%;margin-left:-200px;margin-top:10px;opacity:0;transition:opacity .3s;font-size:13px;line-height:1.6;white-space:pre-line;box-shadow:0 4px 6px rgba(0,0,0,.3)}
            .tooltip-container:hover .tooltip-text{visibility:visible;opacity:1}
            .title-with-tooltip{display:flex;align-items:center;margin-bottom:1rem}
            .title-with-tooltip h3{margin:0;display:inline}
            </style>
            """,
            unsafe_allow_html=True,
        )
        tooltip = (
            "[공장 부하 패턴 정의]\n"
            "1. 휴무일: 전체 시간대 경부하\n"
            "2. 가동일\n • 봄/여름/가을 최대부하: 10-12, 13-17\n • 겨울철 최대부하: 10-12, 17-20, 22-23\n • 경부하: 23-09"
        )
        st.markdown(
            f"""
            <div class="title-with-tooltip">
                <h3>시간대별 부하 발생 빈도</h3>
                <div class="tooltip-container"><span class="tooltip-icon">ⓘ</span><span class="tooltip-text">{tooltip}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        load_map2 = {"경부하": "Light_Load", "중간부하": "Medium_Load", "최대부하": "Maximum_Load"}
        polar_colors = {"경부하": {"line": "#4CAF50", "fill": "rgba(76,175,80,.3)"}, "중간부하": {"line": "#FFC107", "fill": "rgba(255,193,7,.3)"}, "최대부하": {"line": "#EF5350", "fill": "rgba(239,83,80,.3)"}}

        st.markdown("##### 부하 유형 선택")
        s1, s2, s3 = st.columns(3)
        selected = []
        if s1.checkbox("최대부하", value=True, key="p1"): selected.append("최대부하")
        if s2.checkbox("중간부하", value=True, key="p2"): selected.append("중간부하")
        if s3.checkbox("경부하", value=True, key="p3"): selected.append("경부하")

        fig_polar = go.Figure()
        all_counts, total_count = [], 0
        if not selected:
            st.warning("⚠️ 최소한 하나의 부하 유형을 선택해야 합니다.")
        else:
            for ui_name in selected:
                data_name = load_map2[ui_name]
                sub = filtered_df[filtered_df["작업유형"] == data_name]
                hour_counts = sub.groupby("hour").size().reindex(range(24), fill_value=0)
                total_count += len(sub)
                all_counts.extend(hour_counts.values.tolist())
                fig_polar.add_trace(
                    go.Scatterpolar(
                        r=hour_counts.values,
                        theta=[f"{h:02d}:00" for h in range(24)],
                        fill="toself",
                        fillcolor=polar_colors[ui_name]["fill"],
                        line=dict(color=polar_colors[ui_name]["line"], width=2),
                        marker=dict(size=8, color=polar_colors[ui_name]["line"]),
                        name=ui_name,
                    )
                )
            max_val = max(all_counts) if all_counts else 10
            fig_polar.update_layout(
            polar=dict(
                # 🌟 RadialAxis (반지름 축) 스타일 적용 - title_font는 필요 없으나, AXIS_STYLE에서 제거했으므로 가능
                radialaxis=dict(visible=True, range=[0, max_val * 1.1], **POLAR_AXIS_STYLE), 
                # 🌟 AngularAxis (각도 축) 스타일 적용 - title_font 속성 제거
                angularaxis=dict(direction="clockwise", rotation=90, dtick=3, **POLAR_AXIS_STYLE), 
            ),
            height=550,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=AXIS_FONT_SIZE)),
            font=dict(color="black", size=AXIS_FONT_SIZE),
        )
            st.plotly_chart(fig_polar, use_container_width=True)
            st.caption(f"📌 선택한 기간 내 선택 부하 유형 총 발생 건수: **{total_count:,}건**")



# ============================================================================
# Tab 3. 역률 관리

# ============================================================================
# ⚙️ 역률 규정 및 시간 필터링 설정
# ============================================================================

# 한전 역률 규정 기준
LAG_PF_THRESHOLD_PENALTY = 90  # 지상역률 가산 기준: 90% 미달 시
LAG_PF_THRESHOLD_INCENTIVE = 95 # 지상역률 감액 기준: 90% 초과 ~ 95%까지 (감액 혜택 최대치)
LEAD_PF_THRESHOLD_PENALTY = 95 # 진상역률 가산 기준: 95% 미달 시 (진상으로 95% 미달 = 95% 초과)

def calculate_time_based_metrics(df):
    """주간/야간 시간 기준에 따라 평균 역률 및 무효전력량을 계산합니다."""
    
    # 1. 지상역률 적용 시간: 주간 09:00 부터 22:00 까지
    lag_time_df = df[(df["hour"] >= 9) & (df["hour"] < 22)].copy()
    
    # 2. 진상역률 적용 시간: 야간 22:00 부터 다음 날 09:00 까지
    lead_time_df = df[(df["hour"] >= 22) | (df["hour"] < 9)].copy()
    
    # 평균 지상 역률 (주간 데이터만 사용)
    valid_lag_pf = lag_time_df[lag_time_df["지상역률(%)"] > 0]["지상역률(%)"]
    avg_lag_pf_actual = valid_lag_pf.mean() if not valid_lag_pf.empty else 0
    
    # 평균 진상 역률 (야간 데이터만 사용)
    valid_lead_pf = lead_time_df[lead_time_df["진상역률(%)"] > 0]["진상역률(%)"]
    avg_lead_pf_actual = valid_lead_pf.mean() if not valid_lead_pf.empty else 0
    
    return avg_lag_pf_actual, avg_lead_pf_actual

# ============================================================================
# 탭 3 시작
# ============================================================================
with tab3:
    if not selected_work_status:
        st.warning("⚠️ 사이드바에서 '작업 상태 선택'을 지정하세요.")
        st.stop()

    # 1. 💡 KPI 지표 계산 및 시간 기준 반영
    # 전력량 합계는 전체 시간을 기준으로 계산
    total_power_usage = filtered_df["전력사용량(kWh)"].sum()
    total_lag_kvarh = filtered_df["지상무효전력량(kVarh)"].sum()
    total_lead_kvarh = filtered_df["진상무효전력량(kVarh)"].sum()

    # 규정에 맞는 시간대별 평균 역률 계산
    avg_lag_pf_actual, avg_lead_pf_actual = calculate_time_based_metrics(filtered_df)

    # 델타 값 계산 및 색상 로직 준비
    delta_lag = (avg_lag_pf_actual - LAG_PF_THRESHOLD_PENALTY)
    delta_lead = (avg_lead_pf_actual - LEAD_PF_THRESHOLD_PENALTY)
    
    # 지상 역률 델타 (90% 기준. 낮으면 위험/빨강)
    delta_lag_text = f"{(delta_lag):.2f}% vs {LAG_PF_THRESHOLD_PENALTY}%"
    delta_lag_color = "#dc3545" if delta_lag < 0 else "#28a745"

    # 진상 역률 델타 (95% 기준. 높으면 위험/빨강)
    delta_lead_text = f"{(delta_lead):.2f}% vs {LEAD_PF_THRESHOLD_PENALTY}%"
    delta_lead_color = "#dc3545" if delta_lead > 0 else "#28a745"

    # ----------------------------------------------------
    # 2. 🌟 KPI 박스 시각화 (CSS 통일) 🌟
    # ----------------------------------------------------
    
    # 탭 1 CSS를 다시 정의합니다. (최상단에서 한 번만 정의하는 것이 좋지만, 여기서는 통합을 위해 삽입)
    st.markdown(
        """
        <style>
        .kpi-card { background-color:#f0f2f6; padding:20px; border-radius:10px; border-left:5px solid #1f77b4; height:140px; display:flex; flex-direction:column; justify-content:center; margin-bottom: 15px; }
        .kpi-title { font-size:16px; color:#666; margin-bottom:5px; }
        .kpi-value { font-size:32px; font-weight:bold; color:#1f77b4; margin-bottom:5px; }
        .kpi-unit { font-size:14px; color:#888; margin-top: -5px; }
        .kpi-delta { font-size:14px; font-weight:bold; margin-top:5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 기간별 역률 관리 핵심 지표")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. 총 전력사용량
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">총 전력사용량</div><div class="kpi-value">{total_power_usage:,.0f}</div><div class="kpi-unit">kWh</div></div>', unsafe_allow_html=True)
    
    # 2. 총 지상 무효 전력량
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">총 지상 무효전력량</div><div class="kpi-value">{total_lag_kvarh:,.0f}</div><div class="kpi-unit">kVarh</div></div>', unsafe_allow_html=True)

    # 3. 총 진상 무효 전력량
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">총 진상 무효전력량</div><div class="kpi-value">{total_lead_kvarh:,.0f}</div><div class="kpi-unit">kVarh</div></div>', unsafe_allow_html=True)

    # 4. 평균 지상 역률 (90% 기준, 주간)
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left:5px solid {delta_lag_color};">
                <div class="kpi-title">평균 지상 역률 (주간)</div>
                <div class="kpi-value">{avg_lag_pf_actual:.2f} %</div>
                <div class="kpi-delta" style="color:{delta_lag_color};">{delta_lag_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # 5. 평균 진상 역률 (95% 기준, 야간)
    with col5:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left:5px solid {delta_lead_color};">
                <div class="kpi-title">평균 진상 역률 (야간)</div>
                <div class="kpi-value">{avg_lead_pf_actual:.2f} %</div>
                <div class="kpi-delta" style="color:{delta_lead_color};">{delta_lead_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # ----------------------------------------------------
    # 3. 그래프 및 캡션 수정
    # ----------------------------------------------------
    
    st.subheader("역률 일일 사이클 분석")
    pf_colors = {"가동": "#1f77b4", "휴무": "#ff7f0e"}

    cycle_df = filtered_df.copy()
    cycle_df["time_15min"] = ((cycle_df["hour"] * 60 + cycle_df["minute"]) // 15) * 15
    cycle_df["time_label"] = cycle_df["time_15min"].apply(lambda x: f"{x//60:02d}:{x%60:02d}")

    # 일일 사이클 평균 계산 (여기는 시간 필터링 없이 전체 패턴을 보여주는 것이 일반적입니다.)
    daily_cycle = (
        cycle_df.groupby(["작업휴무", "time_15min", "time_label"]).agg(
            avg_lag_pf=("지상역률(%)", "mean"), avg_lead_pf=("진상역률(%)", "mean")
        )
    ).reset_index().sort_values("time_15min")

    all_time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    col_lag, col_lead = st.columns(2)

    with col_lag:
        # 🌟 지상역률 그래프 수정: 90% 기준선, 09:00~22:00 강조
        st.markdown("#### 🟢 지상역률(%) 일일 사이클 (추가/감액 기준: 90%)")
        fig_lag = go.Figure()
        fig_lag.add_vrect(x0="09:00", x1="22:00", fillcolor="yellow", opacity=0.15, layer="below", line_width=0)
        for status in selected_work_status:
            sub = daily_cycle[daily_cycle["작업휴무"] == status]
            fig_lag.add_trace(
                go.Scatter(x=sub["time_label"], y=sub["avg_lag_pf"], mode="lines", name=status, line=dict(color=pf_colors.get(status, "gray"), width=2))
            )
        fig_lag.add_hline(y=LAG_PF_THRESHOLD_PENALTY, line_dash="dash", line_color="red", line_width=2) # 90% 기준선
        fig_lag.add_hline(y=LAG_PF_THRESHOLD_INCENTIVE, line_dash="dash", line_color="#28a745", line_width=1) # 95% 감액 최대선
        fig_lag.update_layout(
            height=500,
            xaxis=dict(title="시간 (15분)", categoryorder="array", categoryarray=all_time_labels, tickvals=[f"{h:02d}:00" for h in range(24)], ticktext=[f"{h}" for h in range(24)]),
            yaxis=dict(title="평균 지상역률(%)", range=[40, 102]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_lag, use_container_width=True)

    with col_lead:
        # 🌟 진상역률 그래프 수정: 95% 기준선, 22:00~09:00 강조
        st.markdown("#### 🔴 진상역률(%) 일일 사이클 (추가 요금 기준: 95%)")
        fig_lead = go.Figure()
        fig_lead.add_vrect(x0="22:00", x1="23:45", fillcolor="orange", opacity=0.15, layer="below", line_width=0)
        fig_lead.add_vrect(x0="00:00", x1="09:00", fillcolor="orange", opacity=0.15, layer="below", line_width=0)
        for status in selected_work_status:
            sub = daily_cycle[daily_cycle["작업휴무"] == status]
            fig_lead.add_trace(
                go.Scatter(x=sub["time_label"], y=sub["avg_lead_pf"], mode="lines", name=status, line=dict(color=pf_colors.get(status, "gray"), width=2))
            )
        fig_lead.add_hline(y=LEAD_PF_THRESHOLD_PENALTY, line_dash="dash", line_color="red", line_width=2) # 95% 기준선
        fig_lead.update_layout(
            height=500,
            xaxis=dict(title="시간 (15분)", categoryorder="array", categoryarray=all_time_labels, tickvals=[f"{h:02d}:00" for h in range(24)], ticktext=[f"{h}" for h in range(24)]),
            yaxis=dict(title="평균 진상역률(%)", range=[0, 102]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_lead, use_container_width=True)

    # ----------------------------------------------------
    # 4. 동적 캡션 생성 로직 수정 (90% 기준으로 변경)
    # ----------------------------------------------------
    
    analysis_results = []
    
    # 1. 지상 역률 위험 구간 진단 (90% 기준)
    lag_risk_data = daily_cycle[daily_cycle["avg_lag_pf"] < LAG_PF_THRESHOLD_PENALTY]
    
    if not lag_risk_data.empty:
        worst_lag = lag_risk_data["avg_lag_pf"].min()
        worst_row = lag_risk_data[lag_risk_data["avg_lag_pf"] == worst_lag].iloc[0]
        status_kr = "가동일" if worst_row["작업휴무"] == "가동" else "휴무일"
        
        analysis_results.append(
            f"① **지상역률 위험:** **{status_kr}**의 **{worst_row['time_label']}** 구간에서 평균 역률이 **{worst_lag:.2f}%**로 **90% 미달**을 기록했습니다. 이 구간의 설비 부하 패턴을 즉시 점검하여 요금 가산을 방지하세요."
        )
    else:
        analysis_results.append(
            f"① **지상역률 양호:** 주간 시간(09시~22시) 동안 지상역률이 **90%** 이상으로 잘 유지되었습니다. **95%** 초과 구간을 목표로 관리하여 감액 혜택을 극대화하세요."
        )

    # 2. 진상 역률 위험 구간 진단 (95% 기준)
    # 진상 역률은 95% 미만이어야 양호 (95% 초과 = 진상도가 95% 미만인 상태)
    # 규정: "진상역률에 대해서 95%에 미달하는 경우에는 ... 추가한다." -> PF 95% 초과 시 진상 가산 없음.
    # 즉, 진상으로 95% 미달 = 리스크, 진상 95% 초과 = 안전. (이전 해석과 다름)
    # Plotly 그래프의 Y축이 0~102이므로, PF 95% 초과는 '진상' 관점에서는 문제가 없어야 합니다.
    # (일반적으로 진상역률이 100%에 가까울수록 진상 무효 전력이 0에 가까워집니다. 하지만 콘덴서 과투입 위험은 진상 무효전력량 자체가 높을 때 발생합니다.)
    # 여기서는 '진상역률(%)' 컬럼이 낮을수록 진상 무효 전력이 많아짐을 가정하고, 95% 미만을 위험으로 판단합니다.
    
    # 안전하게 95% 미만을 위험 구간으로 설정 (지상과 동일한 방식으로)
    lead_risk_data = daily_cycle[daily_cycle["avg_lead_pf"] < LEAD_PF_THRESHOLD_PENALTY]
    
    if not lead_risk_data.empty:
        worst_lead = lead_risk_data["avg_lead_pf"].min()
        worst_row = lead_risk_data[lead_risk_data["avg_lead_pf"] == worst_lead].iloc[0]
        status_kr = "가동일" if worst_row["작업휴무"] == "가동" else "휴무일"
        
        analysis_results.append(
            f"② **진상역률 위험:** **{status_kr}**의 **{worst_row['time_label']}** 구간에서 진상역률이 **{worst_lead:.2f}%**로 **95% 미달**을 기록했습니다. 이는 야간 시간대(22시~09시) 콘덴서 **과투입/설비 리스크**를 시사하며, 요금 가산 리스크가 있습니다."
        )
    else:
        analysis_results.append(
            "② **진상역률 양호:** 야간 시간(22시~09시) 동안 진상역률이 **95%** 이상으로 잘 유지되었습니다. 콘덴서 제어가 잘 작동 중입니다."
        )

    # 3. 휴무일 특이 사항 진단 (90% 미만 / 95% 초과를 일반적인 '이상'으로 진단)
    if "휴무" in selected_work_status:
        rest_day_data = daily_cycle[daily_cycle["작업휴무"] == "휴무"]
        rest_day_lag_risk = rest_day_data[rest_day_data["avg_lag_pf"] < 90]
        rest_day_lead_risk = rest_day_data[rest_day_data["avg_lead_pf"] < 95] # 95% 미만으로 가정
        
        if not rest_day_lag_risk.empty or not rest_day_lead_risk.empty:
            analysis_results.append(
                "③ **휴무일 특이사항:** 휴무일에도 **비정상적인 역률 변동** (90% 미만 또는 95% 미만)이 관찰되었습니다. 이는 상시 가동되는 주요 설비의 비효율적인 콘덴서 제어 또는 누설 전류로 인한 것일 수 있습니다. **설비 점검**이 필요합니다."
            )
        else:
            analysis_results.append(
                "③ **휴무일 특이사항:** 휴무일에는 역률이 안정적으로 유지되어 특별한 위험이 발견되지 않았습니다."
            )
            
    # 최종 캡션 출력
    final_caption = "\n\n".join(analysis_results)
    st.caption(final_caption)
# ============================================================================
# Tab 4. 공회전 에너지 분석
# ============================================================================

def get_idle_data(df: pd.DataFrame):
    if df.empty:
        return None, None, None
    df_work = df[df["작업휴무"] == "가동"].copy()
    df_rest = df[df["작업휴무"] == "휴무"].copy()

    work_night = df_work[(df_work["hour"] >= 22) | (df_work["hour"] < 8)].copy()
    work_baseline_val = work_night['전력사용량(kWh)'].quantile(0.3) if not work_night.empty else 0
    rest_baseline_val = df_rest['전력사용량(kWh)'].quantile(0.3) if not df_rest.empty else 0

    df_work.loc[:, 'baseline'] = work_baseline_val
    df_work.loc[:, 'is_idle_hour'] = (df_work['hour'] >= 22) | (df_work['hour'] < 8)
    df_work.loc[:, 'idle_power'] = 0.0
    cond_work = (df_work['is_idle_hour']) & (df_work['전력사용량(kWh)'] > df_work['baseline'])
    df_work.loc[cond_work, 'idle_power'] = df_work['전력사용량(kWh)'] - df_work['baseline']

    df_rest.loc[:, 'baseline'] = rest_baseline_val
    df_rest.loc[:, 'is_idle_hour'] = True
    df_rest.loc[:, 'idle_power'] = 0.0
    cond_rest = (df_rest['전력사용량(kWh)'] > df_rest['baseline'])
    df_rest.loc[cond_rest, 'idle_power'] = df_rest['전력사용량(kWh)'] - df_rest['baseline']

    combined = pd.concat([df_work, df_rest], ignore_index=True)
    combined.loc[:, 'idle_cost'] = 0.0
    valid = combined['전력사용량(kWh)'] != 0
    combined.loc[valid, 'idle_cost'] = combined['전기요금(원)'] * (combined['idle_power'] / combined['전력사용량(kWh)'])

    daily_idle = (
        combined.groupby(['date', '작업휴무']).agg(loss=('idle_power', 'sum'), cost=('idle_cost', 'sum')).reset_index()
    )
    daily_idle = daily_idle.rename(columns={'작업휴무': 'type'})
    daily_idle['cumulative_loss'] = daily_idle['loss'].cumsum().round(1)

    kpis = {
        '가동일 야간 베이스라인': {'value': work_baseline_val, 'unit': 'kWh'},
        '휴무일 베이스라인': {'value': rest_baseline_val, 'unit': 'kWh'},
        '공회전 에너지 손실': {
            'value': daily_idle['loss'].sum().round(0),
            'unit': 'kWh',
            'details': [daily_idle[daily_idle['type'] == '가동']['loss'].sum().round(0), daily_idle[daily_idle['type'] == '휴무']['loss'].sum().round(0)],
        },
        '공회전 비용 손실': {'value': daily_idle['cost'].sum().round(0), 'unit': '₩', 'details': []},
    }
    return daily_idle, kpis, combined

with tab4:
    current_min_date = filtered_df['date'].min()
    current_max_date = filtered_df['date'].max()

    daily_idle_summary, kpis_idle, _ = get_idle_data(filtered_df)

    if daily_idle_summary is None or daily_idle_summary.empty:
        st.warning("⚠️ 선택된 기간에 데이터가 없어 공회전 분석을 진행할 수 없습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class=\"kpi-card\" style=\"border-left-color:#667eea; background-color:#f0f7ff;\"> 
                    <div class=\"kpi-title\">가동일 - 야간 베이스라인</div>
                    <div class=\"kpi-value\">{kpis_idle['가동일 야간 베이스라인']['value']:,.1f} {kpis_idle['가동일 야간 베이스라인']['unit']}</div>
                    <div class=\"kpi-unit\">평균 전력 (하위 30%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class=\"kpi-card\" style=\"border-left-color:#f5576c; background-color:#fff0f2;\">
                    <div class=\"kpi-title\">휴무일 베이스라인</div>
                    <div class=\"kpi-value\">{kpis_idle['휴무일 베이스라인']['value']:,.1f} {kpis_idle['휴무일 베이스라인']['unit']}</div>
                    <div class=\"kpi-unit\">평균 전력 (하위 30%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class=\"kpi-card\" style=\"border-left-color:#ffa751; background-color:#fffaf0;\">
                    <div class=\"kpi-title\">공회전 에너지 손실</div>
                    <div class=\"kpi-value\">{kpis_idle['공회전 에너지 손실']['value']:,.0f} {kpis_idle['공회전 에너지 손실']['unit']}</div>
                    <div class=\"kpi-unit\">가동-야간: {kpis_idle['공회전 에너지 손실']['details'][0]:,.0f} kWh | 휴무일: {kpis_idle['공회전 에너지 손실']['details'][1]:,.0f} kWh</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class=\"kpi-card\" style=\"border-left-color:#43e97b; background-color:#f0fff7;\">
                    <div class=\"kpi-title\">공회전 비용 손실</div>
                    <div class=\"kpi-value\">₩{kpis_idle['공회전 비용 손실']['value']:,.0f}</div>
                    <div class=\"kpi-unit\">계산된 누적 요금</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.divider()


        # ----------------------------------------------------
        # 🏆 일별 공회전 손실 TOP 10  (선택 기간(filtered_df) 기준)
        # ----------------------------------------------------
        
        # 축 스타일 변수 (글자색: 검정, 크기: 18)
        AXIS_FONT_SIZE = 18 
        AXIS_STYLE = dict(
            tickfont=dict(color='black', size=AXIS_FONT_SIZE),
            title_font=dict(color='black', size=AXIS_FONT_SIZE)
        )
        
        st.subheader("🏆 일별 공회전 손실 TOP 10")
        
        pivot = (
            daily_idle_summary
            .pivot(index="date", columns="type", values="loss")
            .fillna(0)
        )
        pivot["total_loss"] = pivot.sum(axis=1)
        
        # 휴무 손실이 크면 #f5576c (빨강 계열), 가동 손실이 크면 #667eea (파랑 계열) 유지
        pivot["major"] = np.where(pivot.get("휴무", 0) >= pivot.get("가동", 0), "휴무", "가동")
        
        # ✅ 손실 내림차순 정렬
        top10 = (
            pivot.sort_values("total_loss", ascending=False)
                 .head(10)
                 .reset_index()
        )
        top10["label"] = pd.to_datetime(top10["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        
        # 🌟 막대 색상 로직: 요청하신 bar_colors와 다른 로직이지만, 공회전 손실의 의미에 맞게 휴무/가동 로직 유지
        # 휴무가 더 크면 빨강(위험), 가동이 더 크면 파랑(일반)으로 설정
        top10["color"] = np.where(top10["major"].eq("휴무"), "#f5576c", "#667eea")
        
        fig_top = go.Figure(
            go.Bar(
                x=top10["total_loss"],
                y=top10["label"].astype(str),
                orientation="h",
                marker_color=top10["color"],
                text=top10["total_loss"].round(1),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>손실: %{x:.1f} kWh<extra></extra>",
                # 🌟 막대 위 텍스트 크기 및 색상 설정
                textfont=dict(color='black', size=AXIS_FONT_SIZE) 
            )
        )
        
        # 🔒 y축을 카테고리로 고정 + 우리가 준 순서를 '위에서 아래'로 사용
        fig_top.update_layout(
            height=420,
            xaxis_title="손실 (kWh)",
            yaxis_title="날짜",
            font=dict(color="black", size=AXIS_FONT_SIZE), # 🌟 전체 폰트 크기 및 색상 적용
            
            # 🌟 X축 스타일 적용
            xaxis=dict(showgrid=False, **AXIS_STYLE), 
            
            # 🌟 Y축 스타일 적용
            yaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=top10["label"].tolist(), 
                autorange="reversed",
                **AXIS_STYLE # 🌟 Y축 스타일 적용
            ),
            margin=dict(l=80, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.divider()


        # ==============================
        # 🕘 시간대별 손실 패턴 & 베이스라인
        #   - 범위: 22:00 ~ 08:00
        #   - 실제 전력(선) + 베이스라인(점선) + 공회전 손실(막대)
        #   - '가동일' / '휴무일' 토글
        # ==============================

        st.subheader("📈 시간대별 손실 패턴 & 베이스라인")

        # --- 베이스라인 값 가져오기(안전 추출) ---
        work_baseline = float(kpis_idle.get("가동일 야간 베이스라인", {}).get("value", 0) or 0.0)
        rest_baseline = float(kpis_idle.get("휴무일 베이스라인", {}).get("value", 0) or 0.0)

        # --- 토글(세그먼트) ---
        mode = st.radio("보기", ["가동일", "휴무일"], horizontal=True, index=0)
        sel_flag = "가동" if mode == "가동일" else "휴무"
        baseline = work_baseline if mode == "가동일" else rest_baseline

        # --- 선택 데이터(해당 기간 + 상태) ---
        df_sel = filtered_df.loc[filtered_df["작업휴무"].eq(sel_flag)].copy()
        df_sel["dt"] = pd.to_datetime(df_sel["측정일시"], errors="coerce")
        df_sel["hour"] = df_sel["dt"].dt.hour

        # 밤 10시(22) ~ 아침 8시(08) 구간만 추림
        df_night = df_sel[(df_sel["hour"] >= 22) | (df_sel["hour"] < 8)].copy()

        # --- 시간축(연속형 숫자)과 레이블 준비 ---
        import numpy as np
        vals = np.arange(22, 32)  # 22,23,24(=00),...31(=07)
        labels = [f"{(h if h < 24 else h-24):02d}:00" for h in vals]

        # 시간별 평균 전력(kWh)
        df_night["xnum"] = df_night["hour"].apply(lambda h: h if h >= 22 else h + 24)
        hourly = (
            df_night.groupby("xnum")["전력사용량(kWh)"]
            .mean()
            .reindex(vals, fill_value=0.0)
            .reset_index()
            .rename(columns={"전력사용량(kWh)": "power"})
        )
        hourly["loss"] = (hourly["power"] - baseline).clip(lower=0)

        # --- Plotly ---
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig_hour = make_subplots(specs=[[{"secondary_y": False}]])

        # (1) 공회전 손실 막대
        fig_hour.add_trace(
            go.Bar(
                x=hourly["xnum"],
                y=hourly["loss"],
                name="공회전 손실",
                marker=dict(color="rgba(255,193,7,0.45)", line=dict(color="rgba(255,193,7,1.0)", width=1.8)),
                hovertemplate="<b>%{x}</b><br>손실: %{y:.1f} kWh<extra></extra>",
            )
        )

        # (2) 실제 전력 선
        fig_hour.add_trace(
            go.Scatter(
                x=hourly["xnum"],
                y=hourly["power"],
                name="실제 전력 (kWh)",
                mode="lines+markers",
                line=dict(width=3, color="#5B7BFA"),
                marker=dict(size=7, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>전력: %{y:.1f} kWh<extra></extra>",
            )
        )

        # (3) 베이스라인
        fig_hour.add_hline(
            y=baseline,
            line_dash="dot",
            line_color="crimson",
            line_width=2,
            annotation_text="베이스라인",
            annotation_position="top right",
        )

        # (4) 야간 영역(22~08) 하이라이트
        fig_hour.add_vrect(x0=22, x1=31, fillcolor="rgba(91,123,250,0.10)", line_width=0, layer="below")

        # 축/레이아웃
        fig_hour.update_xaxes(
            tickmode="array",
            tickvals=vals,
            ticktext=labels,
            title_text="야간 시간대 (22:00~08:00)",
            showgrid=False,
            range=[21.5, 31.5],
        )
        fig_hour.update_yaxes(title_text="전력 (kWh)", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        fig_hour.update_layout(
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=10, b=40),
            font_color="black",
        )

        st.plotly_chart(fig_hour, use_container_width=True)
        st.caption(f"기준: {mode} 베이스라인 {baseline:,.1f} kWh")
        st.divider()

        # ==============================
        # 📊 공회전 에너지 누적 (일별 추이)
        #   - 왼쪽 Y: 일별 공회전(kWh) 막대
        #   - 오른쪽 Y: 누적 공회전(kWh) 선
        #   - 최근 7일 하이라이트
        # ==============================

        st.subheader("📊 공회전 에너지 누적 (일별 추이)")

        # 1) 날짜 정렬 & 컬럼 준비
        cum_df = daily_idle_summary.copy()
        cum_df["dt"] = pd.to_datetime(cum_df["date"], errors="coerce")
        cum_df = cum_df.sort_values("dt")
        # (이미 daily_idle_summary에 cumulative_loss가 있다면 그대로 사용)
        if "cumulative_loss" not in cum_df.columns:
            cum_df["cumulative_loss"] = cum_df["loss"].cumsum()

        # 2) 최근 7일 하이라이트 구간 계산
        if not cum_df.empty:
            end_dt = cum_df["dt"].max()
            start_dt = end_dt - pd.Timedelta(days=6)  # 최근 7일 (끝 포함해서 7일)
        else:
            end_dt = pd.Timestamp.today()
            start_dt = end_dt - pd.Timedelta(days=6)

        # 3) 그래프
        fig_cumul = make_subplots(specs=[[{"secondary_y": True}]])

        # (A) 일별 공회전 막대 (반투명 + 테두리)
        fig_cumul.add_trace(
            go.Bar(
                x=cum_df["dt"],
                y=cum_df["loss"],
                name="일별 공회전 (kWh)",
                marker=dict(
                    color="rgba(102,126,234,0.30)",  # 연보라 반투명
                    line=dict(color="rgba(102,126,234,1.0)", width=2),  # 테두리
                ),
                hovertemplate="<b>%{x|%m-%d}</b><br>일별: %{y:.1f} kWh<extra></extra>",
            ),
            secondary_y=False,
        )

        # (B) 누적 공회전 선 (마커 포함)
        fig_cumul.add_trace(
            go.Scatter(
                x=cum_df["dt"],
                y=cum_df["cumulative_loss"],
                name="누적 공회전 (kWh)",
                mode="lines+markers",
                line=dict(color="#f5576c", width=3),
                marker=dict(size=7, line=dict(width=0)),
                hovertemplate="<b>%{x|%m-%d}</b><br>누적: %{y:,.0f} kWh<extra></extra>",
            ),
            secondary_y=True,
        )

        # (C) 최근 7일 영역 강조
        fig_cumul.add_vrect(
            x0=start_dt, x1=end_dt,
            fillcolor="rgba(245,87,108,0.10)",  # 연분홍 하이라이트
            layer="below", line_width=0,
        )

        # 4) 축/레이아웃
        fig_cumul.update_xaxes(
            title_text="날짜",
            showgrid=False,
            tickformat="%m-%d",
        )
        fig_cumul.update_yaxes(
            title_text="일별 (kWh)",
            secondary_y=False,
            showgrid=False,
            rangemode="tozero",
        )
        fig_cumul.update_yaxes(
            title_text="누적 (kWh)",
            secondary_y=True,
            showgrid=False,
            rangemode="tozero",
        )

        fig_cumul.update_layout(
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=10, b=40),
            font_color="black",
        )

        st.plotly_chart(fig_cumul, use_container_width=True)
        st.divider()


    # ==============================================
    # 분석 인사이트 패널 (Streamlit용)
    # ==============================================
    def render_insights_panel(kpis_idle: dict, filtered_df: pd.DataFrame):
        # --- 안전한 값 추출 ---
        total_loss = float(kpis_idle.get('공회전 에너지 손실', {}).get('value', 0) or 0)
        details = kpis_idle.get('공회전 에너지 손실', {}).get('details', [0, 0]) or [0, 0]
        loss_work = float(details[0] if len(details) > 0 else 0)
        loss_rest = float(details[1] if len(details) > 1 else 0)
        work_baseline_val = float(kpis_idle.get('가동일 야간 베이스라인', {}).get('value', 0) or 0)
        total_idle_cost = float(kpis_idle.get('공회전 비용 손실', {}).get('value', 0) or 0)

        # --- 파생 지표 ---
        rest_percentage = (loss_rest / total_loss * 100) if total_loss > 0 else 0.0
        num_rest_days = int(filtered_df.loc[filtered_df['작업휴무'].eq('휴무'), 'date'].nunique())
        avg_daily_rest_loss = (loss_rest / num_rest_days) if num_rest_days > 0 else 0.0

        # --- 스타일 + 마크업 ---
        st.markdown("""
        <style>
          .insights-panel-container { 
            border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 22px; 
            background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,.05);
          }
          .insight-header { font-weight: 800; font-size: 18px; margin-bottom: 12px; }
          .insight-item { border-top: 1px dashed #e5e7eb; padding: 16px 0; }
          .insight-item:first-of-type { border-top: none; }
          .insight-title { font-weight: 700; margin-bottom: 6px; }
          .insight-text { line-height: 1.65; color: #333; }
          .insight-text strong { color: #111; }
          .insight-text b { color: #111; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insights-panel-container">
          <div class="insight-header">💡 분석 인사이트 & 개선 제안</div>

          <div class="insight-item">
            <div class="insight-title">1. 휴무일 공회전 비중이 높습니다 ({rest_percentage:,.1f}%)</div>
            <div class="insight-text">
              선택 기간 내 전체 공회전 손실 중 <strong>{rest_percentage:,.1f}%</strong>가 휴무일에 발생했습니다.
              휴무일 일평균 불필요 소비는 <strong>{avg_daily_rest_loss:,.1f} kWh</strong>입니다.
              <br>비중이 높다면 <b>자동 차단 시스템</b> 도입을 검토하세요.
            </div>
          </div>

          <div class="insight-item">
            <div class="insight-title">2. 가동일 야간 베이스라인 개선 필요</div>
            <div class="insight-text">
              가동일 야간(22:00–08:00) 베이스라인은 <strong>{work_baseline_val:,.1f} kWh</strong>입니다.
              해당 수준을 초과해 <b>idle_power</b>가 발생한 설비(압축기/HVAC/조명 등)의
              <b>야간 가동 스케줄</b>을 재점검하세요.
            </div>
          </div>

          <div class="insight-item">
            <div class="insight-title">3. 공회전 손실 TOP Day 집중 관리</div>
            <div class="insight-text">
              TOP 10 손실일을 확인하여 휴무 전날 <b>설비 차단 체크리스트</b> 및
              <b>관리자 알림</b> 자동화를 적용하십시오.
            </div>
          </div>

          <div class="insight-item">
            <div class="insight-title">4. 단기 액션 플랜 & 예상 절감 효과</div>
            <div class="insight-text">
              공회전 비용 손실(선택 기간): <strong>₩{total_idle_cost:,.0f}</strong><br><br>
              • <b>즉시(비용 0)</b>: 휴무일 설비 수동 차단 체크리스트 → 초기 절감 효과 파악<br>
              • <b>1개월(₩500,000)</b>: 타이머/스케줄러 기반 자동 차단 시스템 구축<br>
              • <b>3개월(₩2,500,000)</b>: 스마트 EMS 알림/모니터링 시스템 구축<br><br>
              현재 공회전 손실의 50%만 개선해도 <b>약 ₩{total_idle_cost * 0.5:,.0f}</b> 절감이 가능합니다.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # --- 사용 위치 예시 (탭4 맨 끝쯤 차트 아래) ---
    st.markdown("---")
    render_insights_panel(kpis_idle, filtered_df)
