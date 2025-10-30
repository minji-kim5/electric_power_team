# pages/3_📚_부록.py  (또는 appendix.py)
import streamlit as st

st.set_page_config(page_title="부록", layout="wide")
st.title("📚 부록 ")




# ── 1) 원본 데이터 정보 (1~11월) ──
with st.expander("① 원본 데이터 정보 (1~11월)", expanded=True):
    st.write("여기에 1~11월 데이터 요약(기간, 빈도, 결측, 통계, 분포 등)을 표시할 예정입니다.")


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
