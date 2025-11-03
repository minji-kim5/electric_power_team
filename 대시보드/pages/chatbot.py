import streamlit as st
import google.generativeai as genai
import pandas as pd
import time

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


def call_gemini_api(user_query: str, context: str) -> str:
    """Gemini API를 호출하여 AI 응답 생성"""
    if not API_CONFIGURED:
        return "❌ API 키가 설정되지 않았습니다."
    
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
6. 한국어로만 답변하세요

사용자 질문: "{user_query}"
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt, request_options={"timeout": 30})
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "401" in error_msg:
            return "❌ Gemini API 키가 유효하지 않습니다."
        elif "timeout" in error_msg.lower():
            return "❌ 응답 시간 초과. 잠시 후 다시 시도해주세요."
        else:
            return f"❌ 오류: {error_msg}"


# ---- 데이터 로드 ----
@st.cache_data
def load_data():
    df = pd.read_csv("data_dash\\train_dash_df.csv")
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['month'] = df['측정일시'].dt.month
    df['year'] = df['측정일시'].dt.year
    df['day'] = df['측정일시'].dt.day
    df['hour'] = df['측정일시'].dt.hour
    df['minute'] = df['측정일시'].dt.minute
    df['date'] = df['측정일시'].dt.date
    return df


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

[KPI 정보]
- 총 전력사용량: {total_power:,.0f} kWh
- 총 전기요금: {total_cost:,.0f} 원
- 총 탄소배출량: {total_carbon:,.2f} tCO2
- 지상무효전력량: {total_lag:,.1f} kVarh
- 진상무효전력량: {total_lead:,.1f} kVarh

[월별 분석]
"""
    monthly = df.groupby('month').agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'mean'
    }).reset_index()
    monthly = monthly[monthly['month'] <= 11]
    
    for _, row in monthly.iterrows():
        context += f"\n  * {int(row['month'])}월: 사용량 {row['전력사용량(kWh)']:,.0f} kWh, 평균요금 {row['전기요금(원)']:,.0f} 원"
    
    hourly = filtered_df.groupby('hour').agg({
        '전력사용량(kWh)': ['mean', 'min', 'max']
    }).reset_index()
    hourly.columns = ['hour', 'avg', 'min', 'max']
    
    peak_hour = hourly.loc[hourly['avg'].idxmax(), 'hour']
    low_hour = hourly.loc[hourly['avg'].idxmin(), 'hour']
    
    context += f"""

[시간대별 분석]
- 최대 부하 시간: {int(peak_hour):02d}:00 (평균 {hourly.loc[hourly['avg'].idxmax(), 'avg']:,.0f} kWh)
- 최소 부하 시간: {int(low_hour):02d}:00 (평균 {hourly.loc[hourly['avg'].idxmin(), 'avg']:,.0f} kWh)
- 24시간 평균: {hourly['avg'].mean():,.0f} kWh

[작업유형별 분석]
"""
    load_map = {'Light_Load': '경부하', 'Medium_Load': '중간부하', 'Maximum_Load': '최대부하'}
    load_analysis = filtered_df.groupby('작업유형')['전력사용량(kWh)'].agg(['sum', 'count', 'mean']).reset_index()
    load_analysis['작업유형명'] = load_analysis['작업유형'].map(load_map)
    
    for _, row in load_analysis.iterrows():
        context += f"\n  * {row['작업유형명']}: 총 {row['sum']:,.0f} kWh ({int(row['count'])}건), 평균 {row['mean']:,.0f} kWh"
    
    cycle_df = filtered_df.copy()
    cycle_df['time_15min'] = ((cycle_df['hour'] * 60 + cycle_df['minute']) // 15) * 15
    
    daily_cycle = cycle_df.groupby(['작업휴무', 'time_15min']).agg(
        avg_lag_pf=('지상역률(%)', 'mean'),
        avg_lead_pf=('진상역률(%)', 'mean')
    ).reset_index()
    
    on_work = daily_cycle[daily_cycle['작업휴무'] == '가동']
    off_work = daily_cycle[daily_cycle['작업휴무'] == '휴무']
    
    context += f"""

[역률 정보 (KEPCO 기준)]
지상역률 기준: 90% (기준 이하 시 감액)
진상역률 기준: 95% (기준 초과 시 추가요금)

가동일:
  * 지상역률 범위: {on_work['avg_lag_pf'].min():.1f}% ~ {on_work['avg_lag_pf'].max():.1f}%
  * 진상역률 범위: {on_work['avg_lead_pf'].min():.1f}% ~ {on_work['avg_lead_pf'].max():.1f}%

휴무일:
  * 지상역률 범위: {off_work['avg_lag_pf'].min():.1f}% ~ {off_work['avg_lag_pf'].max():.1f}%
  * 진상역률 범위: {off_work['avg_lead_pf'].min():.1f}% ~ {off_work['avg_lead_pf'].max():.1f}%

[일별 정보]
"""
    analysis_df = filtered_df.copy()
    analysis_df['날짜'] = analysis_df['측정일시'].dt.date
    daily = analysis_df.groupby('날짜')['전력사용량(kWh)'].sum().reset_index()
    
    context += f"- 분석 대상 일수: {len(daily)}일"
    context += f"\n- 일평균 전력사용량: {daily['전력사용량(kWh)'].mean():,.0f} kWh"
    context += f"\n- 최고 사용일: {daily.loc[daily['전력사용량(kWh)'].idxmax(), '날짜']} ({daily['전력사용량(kWh)'].max():,.0f} kWh)"
    context += f"\n- 최저 사용일: {daily.loc[daily['전력사용량(kWh)'].idxmin(), '날짜']} ({daily['전력사용량(kWh)'].min():,.0f} kWh)"
    
    return context


# ---- 세션 상태 초기화 ----
ss = st.session_state
ss.setdefault("chat_history", [])

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

# ---- 채팅 메시지 표시 (상단에 새 채팅 버튼) ----
col_new_chat, col_empty = st.columns([1, 10])
with col_new_chat:
    if st.button("➕ 새 채팅", use_container_width=True, help="새 채팅"):
        ss["chat_history"] = []
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
                st.markdown(f'<div style="text-align: left;"><span class="bot-message-content">{msg["content"]}</span></div>', unsafe_allow_html=True)
        
        # 마지막 메시지가 사용자 메시지면 로딩 중 표시
        if ss["chat_history"][-1]["role"] == "user":
            with st.spinner("⏳ 답변을 생각하는 중..."):
                # 데이터 로드 및 컨텍스트 생성
                df = load_data()
                context_data = generate_context(df)
                
                # 마지막 사용자 질문 가져오기
                user_query = ss["chat_history"][-1]["content"]
                
                # AI 응답 생성
                ai_response = call_gemini_api(user_query, context_data)
            
            # 응답 추가
            ss["chat_history"].append({"role": "assistant", "content": ai_response})
            st.rerun()

# ---- 입력 영역 (폼 사용) ----
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([20, 1])
    
    with col_input:
        user_input = st.text_input(
            "",
            placeholder="질문을 입력하고 엔터를 누르세요...",
            label_visibility="collapsed"
        )
    
    with col_send:
        submit_button = st.form_submit_button("⬆️", use_container_width=True, help="전송")
    
    # 폼 제출 시에만 실행
    if submit_button and user_input and user_input.strip():
        # 사용자 메시지 추가
        ss["chat_history"].append({"role": "user", "content": user_input})
        st.rerun()

# 로딩 및 응답 처리
if ss["chat_history"] and ss["chat_history"][-1]["role"] == "user":
    # 마지막 메시지가 사용자 메시지면 AI 응답 생성
    with st.spinner("⏳ 답변을 생각하는 중..."):
        # 데이터 로드 및 컨텍스트 생성
        df = load_data()
        context_data = generate_context(df)
        
        # 마지막 사용자 질문 가져오기
        user_query = ss["chat_history"][-1]["content"]
        
        # AI 응답 생성
        ai_response = call_gemini_api(user_query, context_data)
    
    # 응답 추가
    ss["chat_history"].append({"role": "assistant", "content": ai_response})
    st.rerun()