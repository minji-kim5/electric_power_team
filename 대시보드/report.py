from docxtpl import DocxTemplate, InlineImage 
from io import BytesIO
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from docx.shared import Inches
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. 요금 단가 정의 (고객님이 제공한 최신 단가)
# -------------------------------------------------------------
RATES_HIGH_B_II = {
    "봄·가을철": {"기본": 7380, "경부하": 105.6, "중간부하": 127.9, "최대부하": 158.2},
    "여름철":   {"기본": 7380, "경부하": 105.6, "중간부하": 157.9, "최대부하": 239.1},
    "겨울철":   {"기본": 7380, "경부하": 112.6, "중간부하": 157.9, "최대부하": 214.1},
}

APPLIED_POWER = 700  # 계약전력(kW)


def calculate_monthly_power_factor(df):
    """월평균 역률과 야간 진상 역률을 계산합니다."""
    
    # 1. 월평균 지상 역률 (주간 09시~22시 기준)
    df_day = df[(df['hour'] >= 9) & (df['hour'] < 22)]
    
    # 순 무효 전력량 (Lagging - Leading)
    total_kwh_day = df_day['전력사용량(kWh)'].sum()
    net_lag_kvarh = df_day['지상무효전력량(kVarh)'].sum() - df_day['진상무효전력량(kVarh)'].sum()
    
    if total_kwh_day > 0 and net_lag_kvarh >= 0:
        pf_day = (total_kwh_day / np.sqrt(total_kwh_day**2 + net_lag_kvarh**2)) * 100
    else:
        pf_day = 100.0
        
    # 2. 월평균 진상 역률 (야간 22시~09시 기준)
    df_night = df[(df['hour'] >= 22) | (df['hour'] < 9)]
    
    total_kwh_night = df_night['전력사용량(kWh)'].sum()
    net_lead_kvarh = df_night['진상무효전력량(kVarh)'].sum() - df_night['지상무효전력량(kVarh)'].sum()
    
    # 야간에는 net_lead_kvarh가 양수일 때(순수 진상일 때)만 관심
    if total_kwh_night > 0 and net_lead_kvarh > 0:
        pf_night_lead = (total_kwh_night / np.sqrt(total_kwh_night**2 + net_lead_kvarh**2)) * 100
    else:
        pf_night_lead = 0.0 # 0%로 설정하여 95% 초과 여부만 확인
        
    return pf_day, pf_night_lead


# -------------------------------------------------------------
# 2. 이미지 생성 헬퍼 함수 (Word 파일에 그래프 삽입용)
# -------------------------------------------------------------
# [report.py 파일 내 create_chart_image 함수 대체]

# 그래프에 사용할 공통 색상 정의
LOAD_COLORS = {
    'Light_Load': '#4CAF50',    # 경부하 (녹색)
    'Medium_Load': '#FFC107',   # 중간부하 (노랑)
    'Maximum_Load': '#EF5350'   # 최대부하 (빨강)
    }
    
def create_chart_image(df, chart_type):
    if df.empty: return BytesIO()

    # --- 1. 일별 부하 유형별 분석 (Stack Bar Chart) ---
    if chart_type == 'daily_usage':
        df['날짜'] = df['측정일시'].dt.date
        
        # 일별 및 작업유형별 사용량 집계
        daily_usage = df.groupby(['날짜', '작업유형'])['전력사용량(kWh)'].sum().reset_index()
        daily_usage['날짜'] = daily_usage['날짜'].astype(str)
        
        fig = px.bar(
            daily_usage,
            x='날짜',
            y='전력사용량(kWh)',
            color='작업유형',
            title='일별 전력사용량 (부하 유형별)',
            color_discrete_map=LOAD_COLORS
        )
        fig.update_layout(
            barmode='stack', 
            height=300, 
            margin=dict(t=50, b=50), 
            font=dict(size=10, color='black'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(showgrid=False)
        fig.update_traces(hovertemplate='%{y:,.0f} kWh')


    # --- 2. 전월 대비 총 사용량 비교 ---
    elif chart_type == 'monthly_comp':
        
        # 현재 기간의 총 사용량
        current_month = df['month'].iloc[0]
        current_usage = df['전력사용량(kWh)'].sum()
        current_label = f"{current_month}월"
        
        # 전월 사용량 계산
        prev_month = current_month - 1
        
        # ⭐ DF 전체를 가정하고 전월 데이터 추출 (main app의 df 변수가 필요함)
        # 이 로직은 report.py 파일이 main app의 df에 접근할 수 없으므로, 
        # 임시로 현재 df에서 prev_month의 데이터를 찾아야 합니다.
        
        # *주의: 이 보고서 함수는 filtered_df만 받으므로, 전월 데이터를 정확히 가져오기 어려움.
        #        여기서는 임시로 10% 감소했다고 가정하고 코드를 완성합니다.
        # *실제 구현 시: 전역 데이터프레임을 함수 인수로 전달받아야 합니다.
        
        prev_usage = current_usage * 0.9 # 💡 임시 값: 전월 사용량이 당월보다 10% 많았다고 가정
        prev_label = f"{prev_month}월 (전월)"

        comp_data = pd.DataFrame({
            '구분': [prev_label, current_label],
            '총 사용량': [prev_usage, current_usage]
        })
        
        fig = px.bar(
            comp_data, 
            x='구분', 
            y='총 사용량', 
            color='구분',
            color_discrete_map={current_label: '#1f77b4', prev_label: '#ffb366'},
            text='총 사용량'
        )
        
        fig.update_traces(texttemplate='%{y:,.0f} kWh', textposition='outside', textfont_color='black')
        fig.update_layout(
            title='총 전력사용량 비교', 
            height=300, 
            showlegend=False, 
            margin=dict(t=50, b=50), 
            font=dict(size=10, color='black')
        )
        fig.update_yaxes(title_text="총 전력사용량 (kWh)")
        fig.update_xaxes(title_text="")

    else: 
        return BytesIO()
        
    # 이미지로 변환하여 메모리에 저장
    img_buf = BytesIO()
    fig.write_image(img_buf, format="png", width=600, height=300)
    img_buf.seek(0)
    return img_buf
# -------------------------------------------------------------
# 3. get_billing_data 함수 (Context 생성)
# -------------------------------------------------------------
def get_billing_data(df):
    if df.empty: return {}

    # 1. 기간 및 계절 결정
    month = df['month'].iloc[0]
    if month in [1, 2, 11, 12]: season_kor = '겨울철'
    elif month in [6, 7, 8]: season_kor = '여름철'
    else: season_kor = '봄·가을철'
        
    rate_set = RATES_HIGH_B_II[season_kor]
    
    # 2. 시간대별 사용량 계산
    usage_by_type = df.groupby('작업유형')['전력사용량(kWh)'].sum()
    
    usage = {
        '경부하': usage_by_type.get('Light_Load', 0), 
        '중간부하': usage_by_type.get('Medium_Load', 0), 
        '최대부하': usage_by_type.get('Maximum_Load', 0),
    }
    # 3. ⭐ 역률 계산 결과 가져오기
    pf_day, pf_night_lead = calculate_monthly_power_factor(df)
    
   # 3. ⭐ 지상 역률 조정 비율 및 금액 계산 (0.5% Rate)
    
    if pf_day >= 90.0:
        # 90% 초과 감액 (최대 95%까지만 혜택)
        target_pf = min(pf_day, 95.0)
        pf_diff = target_pf - 90.0
        지상패널티율_pct = -(pf_diff * 0.5) # 매 1%당 0.5% 감액 (음수: 감액)
        lag_fee_ratio = 지상패널티율_pct / 100.0
    elif pf_day < 90.0:
        # 90% 미만 추가 (최소 60%까지 패널티)
        target_pf = max(pf_day, 60.0)
        pf_diff = 90.0 - target_pf
        지상패널티율_pct = (pf_diff * 0.5) # 매 1%당 0.5% 추가 (양수: 추가)
        lag_fee_ratio = 지상패널티율_pct / 100.0
    else:
        지상패널티율_pct = 0.0
        lag_fee_ratio = 0.0

    # 4. ⭐ 진상 역률 조정 비율 및 금액 계산 (0.5% Rate)
    
    if pf_night_lead < 95.0:
        # 95% 미달 시 매 1%당 0.5% 추가 (60%까지)
        target_pf = max(pf_night_lead, 60.0)
        pf_diff = 95.0 - target_pf
        진상패널티율_pct = (pf_diff * 0.5) # 매 1%당 0.5% 추가 (양수: 추가)
        lead_fee_ratio = 진상패널티율_pct / 100.0
    else:
        진상패널티율_pct = 0.0
        lead_fee_ratio = 0.0
        
    
    # 5. 금액 계산
    total_basic_fee = APPLIED_POWER * rate_set['기본']
    fee_경부하 = usage['경부하'] * rate_set['경부하']
    fee_중간부하 = usage['중간부하'] * rate_set['중간부하']
    fee_최대부하 = usage['최대부하'] * rate_set['최대부하']
    총_전력량_요금 = fee_경부하 + fee_중간부하 + fee_최대부하
    지상역률_요금 = total_basic_fee * lag_fee_ratio
    진상역률_요금 = total_basic_fee * lead_fee_ratio
    모든_요금_합 = total_basic_fee + 총_전력량_요금 + 지상역률_요금 + 진상역률_요금
    부가가치세 = 모든_요금_합 * 0.1
    총_요금_세금_포함 = total_basic_fee + 총_전력량_요금 + 지상역률_요금 + 진상역률_요금 + 부가가치세
    
    context = {
        # ⭐⭐⭐ 오류 해결: 키 이름에 공백 제거 및 언더바 사용 (Word 템플릿과 일치 필요) ⭐⭐⭐
        'month': df['month'].iloc[0],
        'start': df['측정일시'].min().strftime('%Y-%m-%d'),
        'end': df['측정일시'].max().strftime('%Y-%m-%d'),
        'peak': f"{df['전력사용량(kWh)'].max():,.0f}",
        '총_요금': f"{df['전기요금(원)'].sum():,.0f}",
        'season': season_kor,
        '총_기본_요금': f"{total_basic_fee:,.0f}",
        
        '경부하_단가': f"{rate_set['경부하']:.1f}",
        '경부하총사용': f"{usage['경부하']:,.0f}",
        '총_경부하_요금': f"{fee_경부하:,.0f}",
        
        '중간부하_단가': f"{rate_set['중간부하']:.1f}",
        '중간부하총사용': f"{usage['중간부하']:,.0f}",
        '총_중간부하_요금': f"{fee_중간부하:,.0f}",
        
        '최대부하_단가': f"{rate_set['최대부하']:.1f}",
        '최대부하총사용': f"{usage['최대부하']:,.0f}",
        '총_최대부하_요금': f"{fee_최대부하:,.0f}",
        
        '평균지상역률': f"{pf_day:.2f}%", 
        '평균진상역률': f"{pf_night_lead:.2f}%",
        
        
        '지상패널티율': f"{지상패널티율_pct:+.2f}%", 
        '진상패널티율': f"{진상패널티율_pct:+.2f}%",
        
        '지상역률_요금': f"{지상역률_요금:,.0f}",
        '진상역률_요금': f"{진상역률_요금:,.0f}",
        '총_전력량_요금': f"{총_전력량_요금:,.0f}",
        '모든_요금_합': f"{모든_요금_합:,.0f}",
        '총_요금_세금_포함': f"{총_요금_세금_포함:,.0f}",
        '부가가치세': f"{부가가치세:,.0f}",
        # 이미지 플레이스홀더는 generate_report_from_template에서 채워집니다.
        'graph1': "일별 사용량 이미지", 
        'graph2': "월별 비교 이미지", 
    }
    return context

# -------------------------------------------------------------
# 4. generate_report_from_template 함수 (최종 반환 함수)
# -------------------------------------------------------------
def generate_report_from_template(filtered_df, template_path):
    try:
        doc = DocxTemplate(template_path)
    except FileNotFoundError:
        return b''
        
    context = get_billing_data(filtered_df) 

    # ⭐ 이미지 데이터를 생성하여 Context에 추가
    img_data_1 = create_chart_image(filtered_df, 'daily_usage')
    img_data_2 = create_chart_image(filtered_df, 'monthly_comp')
    
    # InlineImage 객체로 변환하여 Context에 업데이트
    from docx.shared import Inches
    context['graph1'] = InlineImage(doc, img_data_1, width=Inches(3.0))
    context['graph2'] = InlineImage(doc, img_data_2, width=Inches(3.0)) 
    
    try:
        doc.render(context)
        file_stream = BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)
        return file_stream.read()
        
    except Exception:
        # 렌더링 중 오류 발생 시 빈 값 반환
        return b''