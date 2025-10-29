# controls_skeleton.py
import time
import streamlit as st

st.set_page_config(page_title="12월 예측 - 컨트롤", layout="wide")
st.title("실시간 전력 및 전기요금 모니터링")

# ---- 세션 상태 초기화 ----
ss = st.session_state
ss.setdefault("running", False)   # 실행 중 여부
ss.setdefault("step", 0)          # 반복 스텝(예: 분/틱)
ss.setdefault("log", [])          # 진행 로그(옵션)

# ---- 사이드바 컨트롤 ----
st.sidebar.header("")
start = st.sidebar.button("▶ 재생", type="primary")
stop  = st.sidebar.button("⏸ 정지")
reset = st.sidebar.button("⟲ 리셋")

if start:
    ss.running = True
if stop:
    ss.running = False
if reset:
    ss.running = False
    ss.step = 0
    ss.log.clear()

# ---- 본문: 상태 표시 ----
status = "실행 중" if ss.running else "대기 중"
st.subheader(f"상태: {status}")
st.caption("※ 여기 영역에 이후 '데이터 불러오기 → 예측 → 그래프'를 붙이면 됩니다.")

# ---- (데모) 한 번에 한 스텝씩 수행하는 자리 ----
# 실제로는 이 블록에: 12월 데이터 집계/예측/요금계산/차트 생성 코드를 넣으면 됨.
if ss.running:
    ss.step += 1
    ss.log.append(f"tick {ss.step}")
    st.write(f"작업 스텝: {ss.step}")
    st.progress((ss.step % 100) / 100.0)
else:
    st.info("▶ 재생을 누르면 예측 파이프라인을 실행합니다.")

# ---- 자동 반복(옵션) ----
# 실행 중일 때 1초마다 새로고침하여 다음 스텝을 진행합니다.
# (Streamlit 1.29+ 에서는 st.rerun(), 그 이하 버전은 st.experimental_rerun() 사용)
if ss.running:
    time.sleep(1.0)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
