# pages/3_📚_부록.py  (또는 appendix.py)
import streamlit as st

st.set_page_config(page_title="부록", layout="wide")
st.title("📚 부록 ")




# ── 1) EDA ──
with st.expander("① EDA", expanded=False):
    st.markdown("## 📊 데이터 사용 패턴 및 주요 기준")

    st.markdown("---")

    # 1. 공장 작업 부하 시간대 구분
    st.markdown("### 1. 공장 작업 부하 시간대 구분 (Load Profile)")
    
    st.markdown("""
    이 데이터는 **계절 및 요일에 따라** 전력 부하 시간대가 명확히 구분됩니다.
    
    * **공장 비가동일 (휴일/주말):** 가동 여부와 관계없이 **모든 시간대**는 **경부하**로 분류됩니다.
    """)
    
    # 시간대별 부하 구분 테이블 (가독성 향상)
    load_profile_data = {
        '구분': ['봄/여름/가을 (3월 ~ 10월)', '겨울철 (11월 ~ 2월)'],
        '중간부하': ['09:00\\~10:00, 12:00\\~13:00, 17:00\\~23:00',
                    '09:00\\~10:00, 12:00\\~17:00, 20:00\\~22:00'],
        '최대부하': ['10:00\\~12:00, 13:00\\~17:00',
                    '10:00\\~12:00, 17:00\\~20:00, 22:00\\~23:00'],
        '경부하': ['23:00 \\~ 09:00 (공통)', '23:00 \\~ 09:00 (공통)']
    }
    
    st.table(load_profile_data)
    
    # 시간대별 부하 구분 이미지 (시각화 자료)
    st.image("data_dash\\15min_avg_lagging_pf_cycle.png", caption="")
    st.markdown("_*(9시~22시 기준) 휴무일 때 지상역률 100% , 가동일 때 지상역률 관리 미흡 → 휴무 시에는 역률이 매우 안정적이며, 요금 문제가 발생하지 않음._")

    st.image("data_dash\\15min_avg_leading_pf_cycle.png", caption="")
    st.markdown("_*(22시~09시 기준) 진상역률이 규제 기준(95%)을 수준을 유지 → 시스템이 진상 무효 전력 발생에 대해서는 완벽하게 관리하고 있으며, 이로 인한 요금 추가 페널티는 발생하지 않음._")
    st.markdown("---")


    # 2. 시간대별 패턴
    st.markdown("### 2. 시간대별 전력사용량 패턴 분석")

    st.markdown("#### 시간대별 운영 구간")

    # 테이블 데이터
    table_data = {
        '구간': ['준비', '오전 생산', '점심 시간', '오후 생산', '퇴근 시간', '야간 초입', '야간'],
        '시간': ['08:15~09:00', '09:15~12:00', '12:15~13:00', '13:15~17:15', '17:30~18:30', '18:45~21:00', '21:15~08:00'],
        '활동': ['공장 가동 준비', '오전 생산', '점심 시간', '오후 생산', '퇴근 시간', '야간 초입(잔여)', '야간'],
        '특징': ['낮은 전력사용량', '높은 전력사용량', '급격한 감소', '높은 전력사용량', '중간 수준', '점진적 감소', '매우 낮은 수준']
    }

    st.dataframe(table_data, use_container_width=True)

    st.image("data_dash\\hourly_pattern.png", use_container_width=True)

 
    # 3. 데이터 학습 제외 사유 및 예외 패턴
    st.markdown("### 3. 학습 데이터 제외 사유 (1~2월 야근 패턴)")
    
    st.warning("⚠️ **주의:** 1월 및 2월 데이터는 **특정 패턴**으로 인해 주요 학습 데이터에서 **제외**되었습니다.")
    st.markdown("""
    * **사유:** 1, 2월은 **야근 작업**이 많아 평소 공장의 정상적인 전력 사용 사이클과 다른 **특이 패턴**을 보입니다.
    * **특이점:** 특히 **19:00 ~ 00:00** 사이의 월별 전력 사용량 그래프에서 다른 달과 명확한 차이를 보입니다.
    """)
    
    # 19:00-00:00 월별 전력사용량 그래프 이미지 (패턴 비교)
    st.image("data_dash\\exclude_month.png", caption="")
    st.markdown("---")


    # 4. 한전 역률 규정 및 요금 가감 제도
    st.markdown("### 4. 한전 역률(Power Factor) 규정 및 요금 영향")
    st.markdown("이 데이터 분석은 **한전의 역률 기준**을 준수하는지 파악하여 **전기 요금 가감액**을 계산하는 데 활용됩니다.")

    st.markdown("**💰 요금 가감 기준 (평균 역률 기준)**")
    
    st.markdown("""
    * **09시 ~ 22시 (주간 시간대 - 지상역률 기준):**
        * **미달 시 (90% 미만):** 평균 역률이 90% 미달 시, **60%**까지 매 1%당 기본요금의 **0.2%** 추가.
        * **초과 시 (90% 초과):** 평균 역률이 90% 초과 시, **95%**까지 매 1%당 기본요금의 **0.2%** 감액.
    * **22시 ~ 09시 (야간 시간대 - 진상역률 기준):**
        * **미달 시 (95% 미만):** 평균 역률이 95% 미달 시, **60%**까지 매 1%당 기본요금의 **0.2%** 추가.
    """)
    
    # # 진상역률 및 지상역률 관련 설명 이미지
    # st.image("이미지_파일_경로_역률규정.png", caption="") # 사용자가 준비한 이미지 파일 경로를 넣어주세요.
    st.markdown("---")
    

    # 5. 단가
    st.markdown("### 5. 단가 공식")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data_dash/단가1.png", use_container_width=True)

    with col2:
        st.image("data_dash/단가2.png", use_container_width=True)


# ── 2) 12월 데이터 정보 ──
with st.expander("② 12월 데이터 정보", expanded=True):
    st.write("여기에 12월 데이터 요약을 표시할 예정입니다.")


# ── 3) 파생변수 정의(사전) ──
with st.expander("③ 파생변수 정의(사전)", expanded=False):
    st.write("예) hour, dow, is_daytime, rolling_kwh_7d, pf_est ...")
    st.caption("실제 컬럼 확정 후 여기 표 형태로 정의서를 넣으면 됩니다.")

# ── 4) 모델 설명 (선정 이유·평가 계획) ──
with st.expander("④ 모델 설명 (선정 이유·평가 계획)", expanded=False):
    st.write("예) 알고리즘 후보, 평가 지표, 검증 전략, 리스크/가정 등")
    algo   = st.selectbox("알고리즘", ["베이스라인(평균)", "SARIMA", "Prophet", "XGBoost(회귀)", "LSTM"])
    metric = st.multiselect("지표", ["MAE", "RMSE", "MAPE", "SMAPE"], default=["MAE", "RMSE"])
    st.write(f"선택: {algo}, 지표: {', '.join(metric)}")
