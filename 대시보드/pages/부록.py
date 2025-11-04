import streamlit as st
import pandas as pd

st.set_page_config(page_title="부록", layout="wide")
st.title("부록 (Appendix) ")




# ── 1) EDA ──
with st.expander("① EDA", expanded=False):
    st.markdown("## 데이터 사용 패턴 및 주요 기준")

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

    table_data = {
        '구간': ['준비', '오전 생산', '점심 시간', '오후 생산', '퇴근 시간', '야간 초입', '야간'],
        '시간': ['08:15~09:00', '09:15~12:00', '12:15~13:00', '13:15~17:15', '17:30~18:30', '18:45~21:00', '21:15~08:00'],
        '활동': ['공장 가동 준비', '오전 생산', '점심 시간', '오후 생산', '퇴근 시간', '야간 초입(잔여)', '야간'],
        '특징': ['낮은 전력사용량', '높은 전력사용량', '급격한 감소', '높은 전력사용량', '중간 수준', '점진적 감소', '매우 낮은 수준']
    }

    st.dataframe(table_data, use_container_width=True)

    st.image("data_dash\\hourly_pattern.png", use_container_width=True)
    st.markdown("---")
 
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
    
    st.markdown("---")

    # 5. 단가
    st.markdown("### 5. 단가 공식")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data_dash/단가1.png", use_container_width=True)

    with col2:
        st.image("data_dash/단가2.png", use_container_width=True)


# ── 2) 12월 데이터 정보 ──
with st.expander("② 12월 데이터 정보", expanded=False):
    st.markdown("## 12월 운영 스케줄 및 특이사항")
    
    # 12월 휴무일 정보
    st.markdown("### 12월 휴무일")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 전체 휴무일 리스트")
        holiday_data = {
            '날짜': ['12월 2일 (월)', '12월 9일 (월)', '12월 15일 (일)', 
                    '12월 22일 (일)', '12월 23일 (월)', '12월 25일 (수)', 
                    '12월 29일 (일)', '12월 30일 (월)', '12월 31일 (화)'],
            '비고': ['', '', '', '', '', '크리스마스', '', '', '']
        }
        
        st.dataframe(holiday_data, use_container_width=True)
    
    with col2:
        st.markdown("#### 기간별 휴무일 분포")
        
        period_data = {
            '기간': ['연말 전 (1~20일)', '연말 (21~31일)'],
            '휴무일': ['2일, 9일, 15일', '22일, 23일, 25일, 29일, 30일, 31일'],
            '총 일수': ['3일', '6일']
        }
        
        st.dataframe(period_data, use_container_width=True)
    
    st.markdown("---")
    
    # 12월 조기종료일 정보
    st.markdown("### 12월 조기종료일 (17시 공장 셧다운)")
    
    st.info("""
    **조기종료 대상일:**
    - **12월 1일, 8일**: '가동'인 일요일 조기종료
    - **12월 21일, 28일**: 연속 휴무 직전 조기종료
    
    ⚠️ **영향:** 해당일 17시부터 공장 가동 중단으로 전력사용량 급격히 감소
    """)
    
    early_close_data = {
        '날짜': ['12월 1일 (일)', '12월 8일 (일)', '12월 21일 (토)', '12월 28일 (토)'],
        '요일 특성': ['가동 일요일', '가동 일요일', '연휴 직전', '연휴 직전'],
        '셧다운 시간': ['17:00', '17:00', '17:00', '17:00'],
        '영향': ['조기종료', '조기종료', '조기종료', '조기종료']
    }
    
    st.dataframe(early_close_data, use_container_width=True)


# ── 3) 파생변수 정의(사전) ──
with st.expander("③ 파생변수 정의(사전)", expanded=False):
    st.markdown("## 변수 정의서")
    
    # 기본 변수
    st.markdown("### 기본 변수")
    basic_vars = {
        '변수명': ['month', 'day', 'hour', 'minute', 'hour_sin', 'hour_cos', 
                  'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
                  '기온', '지상역률(%)', '진상역률(%)', '공장 휴무일_인코딩', 
                  '시간대_인코딩', '시간대2_인코딩', '작업유형_인코딩'],
        '설명': ['월', '일', '시간', '분', '시간 사인 변환', '시간 코사인 변환',
                '월 사인 변환', '월 코사인 변환', '연중 일자 사인 변환', '연중 일자 코사인 변환',
                '시간대별 기온 (12월: 22-23년 12월 주차별, 시간대 평균 대체)', 
                '지상역률 백분율', '진상역률 백분율',
                '하루 작업유형이 모두 경부하인 일자: 휴무, 그 외: 가동',
                '주간, 야간 구분', 
                '가동일 작업 사이클 반영 (준비, 오전생산, 점심, 오후생산, 퇴근 등)',
                '경부하, 중간부하, 최대부하 별 인코딩'],
        '타입': ['int', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 
                'float', 'float', 'float', 'float', 'float', 'categorical', 
                'categorical', 'categorical', 'categorical']
    }
    
    st.dataframe(basic_vars, use_container_width=True)
    
    st.markdown("---")
    
    # 강화 변수
    st.markdown("### 강화 변수")
    enhanced_vars = {
        '변수명': ['기온_hour_interaction', '기온_구간', '작업유형_hour'],
        '설명': ['기온×시간변수 교차항', '기온 구간화 인코딩 변수', '작업유형×시간변수 교차항'],
        '목적': ['시간대별 기온 영향 반영', '기온 범위별 패턴 캡처', '시간대별 작업유형 상호작용']
    }
    
    st.dataframe(enhanced_vars, use_container_width=True)
    
    st.markdown("---")
    
    # 통계 변수
    st.markdown("### 통계 변수")
    stat_vars = {
        '변수명': ['시간대2_평균전력', '작업유형_평균전력', 'hour_평균전력'],
        '설명': ['시간대2 유형별 평균전력', '작업유형별 평균전력', '시간별 평균전력'],
        '활용': ['시간대 기준 전력 레벨', '부하 유형 기준 전력 레벨', '시각별 전력 패턴']
    }
    
    st.dataframe(stat_vars, use_container_width=True)
    
    st.markdown("---")
    
    # 타겟 변수
    st.markdown("### 타겟 변수")
    st.info("""
    **타겟 변수:** `전력사용량`
    
    - 전력사용량을 예측하고, 도출한 단가 공식에 따라 **전기요금으로 변환**
    - 최종 목표: 전기요금 예측
    """)


# ── 4) 모델 설명 (선정 이유·평가 계획) ──
with st.expander("④ 모델 설명 (선정 이유·평가 계획)", expanded=False):
    st.markdown("## 모델링 전략 및 구조")
    
    # 데이터 분할
    st.markdown("### 데이터 분할 전략")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Train/Validation Set")
        st.info("""
        **Training Set:** 3월~11월 데이터
        - 1, 2월 제외 (야근일 대부분으로 학습에서 제외)
        
        **Validation Set:** 11월 데이터
        - 최종 검증용
        
        **교차 검증:** Timeseries Split
        - 하이퍼파라미터 튜닝
        """)
    
    with col2:
        st.markdown("#### ⚠️ 제외 데이터")
        st.warning("""
        **1-2월 데이터 제외 사유:**
        - 야근일이 대부분
        - 정상적인 공장 사이클과 상이한 패턴
        - 학습 시 노이즈 요소로 작용
        """)
    
    st.markdown("---")
    
    # 모델 구조
    st.markdown("### 모델 구조")
    
    st.markdown("#### 시간대별 분할 모델링")
    st.success("""
    **분할 기준:** 공장 가동 여부 × 시간대
    
    공장 가동/휴무 여부 및 주간/야간에 따라 전력사용량 변동성이 매우 다르기 때문에 
    **3개의 개별 모델**을 학습 후 결합:
    
    1. **가동-주간** 모델
    2. **가동-야간** 모델  
    3. **휴무** 모델
    """)
    
    st.markdown("---")
    
    # 알고리즘 구성
    st.markdown("### 알고리즘 구성")
    
    tab1, tab2, tab3 = st.tabs(["Boosting 앙상블", "딥러닝 모델", "후처리"])
    
    with tab1:
        st.markdown("#### Boosting 기반 Regressor 모델 앙상블")
        
        boosting_info = {
            '모델': ['XGBoost', 'LightGBM', 'CatBoost'],
            '적용 범위': ['3개 시간대별 모델 각각', '3개 시간대별 모델 각각', '3개 시간대별 모델 각각'],
            '앙상블 방식': ['Validation 성능지표 기반 가중치 최적화', 'Validation 성능지표 기반 가중치 최적화', 'Validation 성능지표 기반 가중치 최적화']
        }
        
        st.dataframe(boosting_info, use_container_width=True)
        
        st.markdown("**추가 실험 버전:**")
        st.markdown("""
        - 변수를 다르게 입력한 버전
        - 3구간이 아닌 하나의 통합모델 버전
        - 타겟변수 로그변환 및 역변환 적용 버전
        - 최종 전기요금 앙상블에 모두 활용
        """)
    
    with tab2:
        st.markdown("#### 시계열 패턴 반영 딥러닝 모델")
        
        st.markdown("**모델 구조 및 학습 전략:**")
        st.info("""
        **GRU 2층 구조** + **가중치 샘플링 전략**
        - 시간대별로 나눈 3개 모델에 대해 각각 LSTM, GRU 모델 학습
        - 최종 전기요금 앙상블에 활용
        """)
        
        dl_strategy = {
            '전략 구분': ['가중치 샘플링', '피처 엔지니어링', '요일 패턴 보정'],
            '적용 방법': ['월별 차등 가중치', '요일 피처 추가', '요일 shift 보정'],
            '세부 내용': [
                '1,2월: 가중치 감소 | 12월 접근: 가중치 증가 | 9월: 연휴패턴으로 가중치 증가',
                '2018년 토,일 → 2024년 일,월 패턴 학습',
                '3월 기준 토,일→일,월 shift로 3월 가중치 감소 또는 요일 보정'
            ],
            '기대 효과': ['12월 예측 정확도 향상', '요일별 패턴 학습 강화', '계절별 패턴 정합성 향상']
        }
        
        st.dataframe(dl_strategy, use_container_width=True)
        
        st.markdown("**핵심 아이디어:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **월별 가중치 전략**
            - 1, 2월: 야근 패턴으로 가중치 ↓
            - 9월: 연휴 패턴 유사로 가중치 ↑  
            - 12월 접근: 시간적 근접성으로 가중치 ↑
            """)
        
        with col2:
            st.warning("""
            **요일 패턴 보정**
            - 2018년 토,일 → 2024년 일,월 shift
            - 3월 기준점으로 요일 보정 적용
            - 요일 피처로 성능 향상 기대
            """)
    
    with tab3:
        st.markdown("#### 공장 사이클 및 12월 특수 패턴 후반영")
        
        st.markdown("**패턴별 보정 사항:**")
        
        correction_data = {
            '보정 항목': ['오전 피크타임 추가', '야간 과대예측 보정', '연말 연휴 패턴', '조기종료일 반영'],
            '적용 시간대': ['오전 시간대', '새벽/야간초입', '21일 이후 가동일 오후', '조기종료일 17-18시 이후'],
            '보정 내용': ['오전 피크타임 전기요금 추가', '가동일 새벽, 야간초입 과대예측 보정', 
                         '연말 가동일 오후 요금 차감', '17시 이후 전기요금 2100원 상한'],
            '목적': ['실제 패턴 반영', '예측 정확도 향상', '연말 특수성 반영', '조기종료 영향 반영']
        }
        
        st.dataframe(correction_data, use_container_width=True)
    
    st.markdown("---")
    
    # 평가 전략
    st.markdown("### 평가 전략")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 평가 지표")
        metrics = ["MAE", "RMSE", "MAPE", "SMAPE"]
        selected_metrics = st.multiselect("평가 지표 선택", metrics, default=["MAE", "RMSE"])
    
    with col2:
        st.markdown("#### 최종 목표")
        st.info("""
        **최종 예측 목표:** 전기요금
        
        전력사용량 → 단가 공식 적용 → 전기요금
        """)
    
    if selected_metrics:
        st.write(f"**선택된 평가 지표:** {', '.join(selected_metrics)}")