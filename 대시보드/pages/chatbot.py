import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="🤖 AI 챗봇", page_icon="🤖", layout="wide")

# ---- Gemini API 설정 ----
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
API_KEY = "AIzaSyAJbO4gJXKf8HetBy6TKwD5fEqAllgX-nc"

try:
    if API_KEY == "YOUR-API-KEY":
        raise KeyError("API 키가 설정되지 않았습니다.")
    genai.configure(api_key=API_KEY)
    API_CONFIGURED = True
except:
    API_CONFIGURED = False


def extract_code_from_response(response_text: str) -> tuple[str, str]:
    """응답에서 코드와 텍스트 분리"""
    code_pattern = r'```python\n(.*?)\n```'
    match = re.search(code_pattern, response_text, re.DOTALL)
    
    if match:
        code = match.group(1)
        text = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
        return text, code
    
    return response_text, None


def call_gemini_api(user_query: str, context: str) -> tuple[str, str]:
    """Gemini API를 호출하여 AI 응답 생성"""
    if not API_CONFIGURED:
        return "❌ API 키가 설정되지 않았습니다.", None
    
    prompt = f"""
당신은 LS ELECTRIC 청주 공장의 전력 관리 AI 어시스턴트입니다.
시설 관리팀을 지원하여 에너지 효율과 비용 절감을 도와줍니다.

[현재 대시보드 데이터]
{context}

[답변 가이드]
1. 질문의 핵심을 파악하세요
2. 위 데이터를 바탕으로 정확하고 구체적으로 답변하세요
3. 수치에는 단위를 명시하세요 (kWh, 원, %, 등)
4. 중요한 정보는 **굵게** 표시하세요
5. 친절하고 전문적인 톤을 유지하세요

[그래프 생성 요청 시]
사용자가 데이터 시각화를 요청하면:
- 분석 내용을 먼저 설명하고 그래프를 생성하세요
- 코드를 요청했을 때 다음 형식으로 Python 코드를 생성하세요:
```python
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure(...)
st.plotly_chart(fig, use_container_width=True)
```

사용자 질문: "{user_query}"
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt, request_options={"timeout": 30})
        text_response, code = extract_code_from_response(response.text.strip())
        return text_response, code
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "401" in error_msg:
            return "❌ API 키가 설정되지 않았습니다.", None
        elif "timeout" in error_msg.lower():
            return "❌ 응답 시간 초과. 잠시 후 다시 시도해주세요.", None
        else:
            return f"❌ 오류: {error_msg}", None


# ---- 데이터 로드 ----
@st.cache_data
def load_data():
    df = pd.read_csv("data_dash\\train_dash_df.csv")
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['month'] = df['측정일시'].dt.month
    df['hour'] = df['측정일시'].dt.hour
    df['minute'] = df['측정일시'].dt.minute
    df['date'] = df['측정일시'].dt.date
    return df


def load_december_data():
    """12월 실시간 스트리밍 데이터 로드"""
    try:
        df_dec = pd.read_csv('data_dash\\december_streaming.csv')
        df_dec['측정일시'] = pd.to_datetime(df_dec['측정일시'])
        df_dec['month'] = 12
        df_dec['hour'] = df_dec['측정일시'].dt.hour
        df_dec['minute'] = df_dec['측정일시'].dt.minute
        df_dec['date'] = df_dec['측정일시'].dt.date
        return df_dec
    except:
        return None


def generate_context(df):
    """대시보드 데이터로 컨텍스트 생성"""
    filtered_df = df.copy()
    
    total_power = filtered_df['전력사용량(kWh)'].sum()
    total_cost = filtered_df['전기요금(원)'].sum()
    total_carbon = filtered_df['탄소배출량(tCO2)'].sum()
    total_lag = filtered_df['지상무효전력량(kVarh)'].sum()
    total_lead = filtered_df['진상무효전력량(kVarh)'].sum()
    
    context = f"""
[대시보드 정보]
데이터 기간: {filtered_df['측정일시'].min().strftime('%Y-%m-%d')} ~ {filtered_df['측정일시'].max().strftime('%Y-%m-%d')}

[기본 KPI 정보]
- 총 전력사용량: {total_power:,.0f} kWh
- 총 전기요금: {total_cost:,.0f} 원
- 총 탄소배출량: {total_carbon:,.2f} tCO2
- 지상무효전력량: {total_lag:,.1f} kVarh
- 진상무효전력량: {total_lead:,.1f} kVarh
"""

    # ========== 추가: 12월 실시간 예측 데이터 ==========
    df_december = load_december_data()
    if df_december is not None and len(df_december) > 0:
        dec_total_power = df_december['전력사용량_예측'].sum()
        dec_total_cost = df_december['전기요금_예측'].sum()
        dec_total_carbon = df_december['탄소배출량_예측'].sum() * 1000
        dec_latest = df_december.iloc[-1]
        dec_step = len(df_december)
        dec_total_rows = len(load_data())
        dec_progress = (dec_step / dec_total_rows * 100) if dec_total_rows > 0 else 0
    
        context += f"""

[12월 실시간 스트리밍 데이터]
수집 현황: {dec_step}/{dec_total_rows} 행 ({dec_progress:.1f}% 완료)
데이터 범위: {df_december['측정일시'].min().strftime('%Y-%m-%d %H:%M')} ~ {df_december['측정일시'].max().strftime('%Y-%m-%d %H:%M')}

[12월 누적 KPI]
- 누적 전력사용량: {dec_total_power:,.2f} kWh
- 누적 전기요금: {dec_total_cost:,.0f} 원
- 누적 탄소배출량: {dec_total_carbon:,.2f} kgCO2

[현재 상태 (최신 데이터)]
- 측정시각: {dec_latest['측정일시'].strftime('%Y-%m-%d %H:%M')}
- 운영상태: {'🟢 가동' if dec_latest['작업휴무'] == '가동' else '🔴 휴무'}
- 작업유형: {dec_latest['작업유형'].replace('_', ' ')}
- 지상역률: {dec_latest['지상역률(%)']:.2f}%
- 진상역률: {dec_latest['진상역률(%)']:.2f}%
"""
    else:
        context += """

[12월 실시간 스트리밍 데이터]
⚠️ 아직 재생 중이 아니거나 데이터가 없습니다.
"""

    # ========== 추가: 역률 데이터 ==========
    try:
        monthly_summary_df = pd.read_csv('data_dash\\월별 역률 패널티 계산.csv')
        total_pf_adjustment = monthly_summary_df['역률_조정금액(원)'].sum()
        context += f"""
[역률 관련 KPI - KEPCO 기준]
- 역률 조정금액: {total_pf_adjustment:,.0f} 원
- 지상역률 기준: 90% (기준 이하 시 감액)
- 진상역률 기준: 95% (기준 초과 시 추가요금)
"""
    except:
        context += """
[역률 관련 KPI]
- 지상역률 기준: 90% (기준 이하 시 감액)
- 진상역률 기준: 95% (기준 초과 시 추가요금)
"""

    # ========== 추가: 가동일/휴무일 통계 ==========
    total_working_days = filtered_df[filtered_df['작업휴무'] == "가동"]['date'].nunique()
    total_holiday_days = filtered_df[filtered_df['작업휴무'] == "휴무"]['date'].nunique()
    context += f"""
[운영 현황]
- 가동일: {total_working_days}일
- 휴무일: {total_holiday_days}일
"""

    # ========== 월별 분석 ==========
    monthly = df.groupby('month').agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'mean'
    }).reset_index()
    monthly = monthly[monthly['month'] <= 11]
    
    context += """

[월별 분석]
"""
    for _, row in monthly.iterrows():
        context += f"\n  * {int(row['month'])}월: 사용량 {row['전력사용량(kWh)']:,.0f} kWh, 평균요금 {row['전기요금(원)']:,.0f} 원"
    
    # ========== 시간대별 분석 (1~11월 데이터만) ==========
    hourly = filtered_df.groupby('hour').agg({
        '전력사용량(kWh)': ['mean', 'min', 'max', 'sum'],
        '전기요금(원)': ['mean', 'sum']
    }).reset_index()
    hourly.columns = ['hour', 'power_avg', 'power_min', 'power_max', 'power_sum', 'cost_avg', 'cost_sum']

    peak_hour = hourly.loc[hourly['power_avg'].idxmax(), 'hour']
    low_hour = hourly.loc[hourly['power_avg'].idxmin(), 'hour']
    peak_value = hourly.loc[hourly['power_avg'].idxmax(), 'power_avg']
    low_value = hourly.loc[hourly['power_avg'].idxmin(), 'power_avg']
    avg_24h = hourly['power_avg'].mean()

    context += f"""

[시간대별 분석 (1~11월 데이터)]
- 최대 부하 시간: {int(peak_hour):02d}:00 (평균 {peak_value:,.0f} kWh)
- 최소 부하 시간: {int(low_hour):02d}:00 (평균 {low_value:,.0f} kWh)
- 24시간 평균: {avg_24h:,.0f} kWh
- 피크/저부하 비율: {peak_value/low_value:.2f}배

[시간대별 상세 데이터 (1~11월)]
시간,평균전력(kWh),최소(kWh),최대(kWh),평균요금(원),누적요금(원)
"""

    # 시간별 데이터 추가
    for _, row in hourly.iterrows():
        hour_str = f"{int(row['hour']):02d}:00"
        context += f"\n{hour_str},{row['power_avg']:.2f},{row['power_min']:.2f},{row['power_max']:.2f},{row['cost_avg']:.0f},{row['cost_sum']:.0f}"

    # ========== 부하 유형별 분석 ==========
    load_map = {
        'Light_Load': '경부하',
        'Medium_Load': '중간부하',
        'Maximum_Load': '최대부하'
    }
    load_analysis = filtered_df.groupby('작업유형').agg({
        '전력사용량(kWh)': ['sum', 'count', 'mean'],
        '전기요금(원)': 'sum'
    }).reset_index()
    load_analysis.columns = ['작업유형', '전력_합', '작업유형_건수', '전력_평균', '요금_합']
    load_analysis['작업유형명'] = load_analysis['작업유형'].map(load_map)
    
    context += """

[작업유형별 분석 - 부하 패턴]
"""
    for _, row in load_analysis.iterrows():
        context += f"\n  * {row['작업유형명']}: 총 {row['전력_합']:,.0f} kWh ({int(row['작업유형_건수'])}건), 평균 {row['전력_평균']:,.0f} kWh, 요금 {row['요금_합']:,.0f}원"

    # ========== 비생산시간 낭비 분석 ==========
    def classify_time_zone(hour):
        """LS 공장 운영 시간대 분류"""
        if 9 <= hour < 12 or 13 <= hour < 17.25:
            return '생산시간'
        elif 18.5 <= hour < 21 or 21 <= hour < 24:
            return '비생산시간'
        else:
            return '기타'
    
    analysis_df = filtered_df.copy()
    analysis_df['시간대구분'] = analysis_df['hour'].apply(classify_time_zone)
    
    non_prod_power = analysis_df[analysis_df['시간대구분'] == '비생산시간']['전력사용량(kWh)'].sum()
    prod_power = analysis_df[analysis_df['시간대구분'] == '생산시간']['전력사용량(kWh)'].sum()
    
    # 휴무일 기준선 계산
    daily_non_prod = analysis_df.groupby([analysis_df['date'], '작업휴무'])['전력사용량(kWh)'].sum().reset_index()
    if not daily_non_prod.empty:
        holiday_avg = daily_non_prod[daily_non_prod['작업휴무'] == '휴무']['전력사용량(kWh)'].mean() if '휴무' in daily_non_prod['작업휴무'].values else 0
        working_avg = daily_non_prod[daily_non_prod['작업휴무'] == '가동']['전력사용량(kWh)'].mean() if '가동' in daily_non_prod['작업휴무'].values else 0
    else:
        holiday_avg = 0
        working_avg = 0
    
    waste_potential = max(0, (working_avg - holiday_avg) * total_working_days) if total_working_days > 0 else 0
    
    context += f"""

[비생산시간 분석 - 낭비 탐지]
- 생산시간 전력: {prod_power:,.0f} kWh
- 비생산시간 전력: {non_prod_power:,.0f} kWh
- 휴무일 기준선: {holiday_avg:,.0f} kWh/일
- 가동일 평균: {working_avg:,.0f} kWh/일
- 일일 초과량: {max(0, working_avg - holiday_avg):,.0f} kWh
- 총 낭비 가능성: {waste_potential:,.0f} kWh
- 개선 포인트: 야간(21:00~08:00) 대기전력 절감
"""

    # ========== 일별 정보 ==========
    analysis_df_daily = filtered_df.copy()
    analysis_df_daily['날짜'] = analysis_df_daily['측정일시'].dt.date
    daily = analysis_df_daily.groupby('날짜')['전력사용량(kWh)'].sum().reset_index()
    
    context += f"""

[일별 통계]
- 분석 대상 일수: {len(daily)}일
- 일평균 전력사용량: {daily['전력사용량(kWh)'].mean():,.0f} kWh
- 최고 사용일: {daily.loc[daily['전력사용량(kWh)'].idxmax(), '날짜']} ({daily['전력사용량(kWh)'].max():,.0f} kWh)
- 최저 사용일: {daily.loc[daily['전력사용량(kWh)'].idxmin(), '날짜']} ({daily['전력사용량(kWh)'].min():,.0f} kWh)
"""
    
    return context


# ---- 세션 상태 초기화 ----
ss = st.session_state
ss.setdefault("chat_history", [])
ss.setdefault("graph_code", None)
ss.setdefault("df", None)

# ---- 제목 ----
st.title("🤖 AI 챗봇")
st.markdown("LS ELECTRIC 청주 공장 전력 관리 AI 어시스턴트")
st.divider()

# ---- CSS 스타일 ----
st.markdown("""
<style>
.user-message-content {
    background: #1f77b4;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    word-wrap: break-word;
    display: inline-block;
}

.bot-message-content {
    background: #e8f4f8;
    color: #333;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 70%;
    word-wrap: break-word;
    display: inline-block;
}

.welcome-message {
    text-align: center;
    color: #999;
    padding: 40px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---- 채팅 메시지 표시 ----
col_new_chat, col_empty = st.columns([1, 10])
with col_new_chat:
    if st.button("➕ 새 채팅", use_container_width=True, help="새 채팅"):
        ss["chat_history"] = []
        ss["graph_code"] = None
        st.rerun()

chat_container = st.container(height=550, border=True)
with chat_container:
    if not ss["chat_history"]:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 안녕하세요!</h3>
            <p>무엇을 도와드릴까요?</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in ss["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div style="text-align: right;"><span class="user-message-content">{msg["content"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: left; background: #e8f4f8; color: #333; padding: 12px 16px; border-radius: 12px; max-width: 70%; word-wrap: break-word; display: inline-block;">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # 마지막 메시지가 사용자 메시지면 로딩 중 표시
        if ss["chat_history"][-1]["role"] == "user":
            with st.spinner("⏳ 답변을 생각하는 중..."):
                # 데이터 로드 및 컨텍스트 생성
                df = load_data()
                ss["df"] = df
                context_data = generate_context(df)
                
                # 마지막 사용자 질문 가져오기
                user_query = ss["chat_history"][-1]["content"]
                
                ss["show_code_only"] = "코드" in user_query and "그래프" not in user_query
                ss["show_graph_only"] = "그래프" in user_query and "코드" not in user_query

                # AI 응답 생성
                ai_response, code = call_gemini_api(user_query, context_data)
            
            # 응답을 chat_history에 추가
            ss["chat_history"].append({"role": "assistant", "content": ai_response})
            ss["graph_code"] = code
            st.rerun()

# ---- 입력 영역 ----
st.divider()

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([20, 1])
    
    with col_input:
        user_input = st.text_input(
            "",
            placeholder="질문을 입력하고 엔터를 누르세요...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col_send:
        submit_button = st.form_submit_button("⬆️", use_container_width=True, help="전송")
    
    if submit_button and user_input and user_input.strip():
        ss["chat_history"].append({"role": "user", "content": user_input})
        ss["graph_code"] = None
        st.rerun()

# ---- 그래프 카드 ----
if ss.get("graph_code") is not None:
    st.divider()
    with st.container(border=True):
        st.subheader("📊 데이터 시각화")
 
        # ✅ 코드만 보여주기 요청 시
        if ss.get("show_code_only"):
            st.code(ss["graph_code"], language="python")

        # ✅ 그래프만 보여주기 요청 시
        elif ss.get("show_graph_only"):
            try:
                exec_globals = {
                    'st': st,
                    'go': go,
                    'pd': pd,
                    'df': ss["df"],
                    'plotly': __import__('plotly')
                }
                exec(ss["graph_code"], exec_globals)
            except Exception as e:
                st.error(f"❌ 그래프 생성 오류: {str(e)}")

        # ✅ 둘 다 요청하거나 일반 요청 시
        else:
            st.code(ss["graph_code"], language="python")
            try:
                exec_globals = {
                    'st': st,
                    'go': go,
                    'pd': pd,
                    'df': ss["df"],
                    'plotly': __import__('plotly')
                }
                exec(ss["graph_code"], exec_globals)
            except Exception as e:
                st.error(f"❌ 그래프 생성 오류: {str(e)}")