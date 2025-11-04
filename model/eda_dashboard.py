import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(layout="wide", page_title="전력사용량 패턴 비교 대시보드")

# 데이터 로드 및 전처리
@st.cache_data
def load_and_prepare_data():
    # Train 데이터 로드
    train_df = pd.read_csv('train_영찬2.csv', encoding='utf-8-sig')
    train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    # Test 데이터 로드 (모든 컬럼이 이미 포함되어 있음)
    test_df = pd.read_csv('예측_월초저녁추가.csv', encoding='utf-8-sig')
    test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    return train_df, test_df

# 데이터 로드
train_df, test_df = load_and_prepare_data()

# 타이틀
st.title("📊 전력사용량 패턴 비교 대시보드")
st.markdown("---")

# 사이드바 설정
st.sidebar.header("🔧 필터 설정")

# 변수 선택
variable_mapping = {
    '전력사용량': ('전력사용량(kWh)', '전력사용량'),
    '전기요금': ('전기요금(원)', '전기요금'),
    '지상역률(%)': ('지상역률(%)', '지상역률(%)'),
    '진상역률(%)': ('진상역률(%)', '진상역률(%)')
}

selected_var_display = st.sidebar.selectbox(
    "분석 변수 선택",
    list(variable_mapping.keys())
)

train_var, test_var = variable_mapping[selected_var_display]

# 작업휴무 필터
work_status = st.sidebar.radio(
    "작업 상태",
    ['전체', '휴무', '가동']
)

# 작업휴무 필터 적용
if work_status == '전체':
    train_filtered = train_df.copy()
    test_filtered = test_df.copy()
else:
    train_filtered = train_df[train_df['작업휴무'] == work_status].copy()
    test_filtered = test_df[test_df['작업휴무'] == work_status].copy()

# 날짜 선택
train_dates = sorted(train_filtered['date'].dt.date.unique())
test_dates = sorted(test_filtered['date'].dt.date.unique())

selected_train_date = st.sidebar.selectbox(
    "Train 데이터 날짜 선택 (1-11월)",
    train_dates,
    format_func=lambda x: x.strftime('%Y-%m-%d')
)

selected_test_date = st.sidebar.selectbox(
    "Test 데이터 날짜 선택 (12월)",
    test_dates,
    format_func=lambda x: x.strftime('%Y-%m-%d')
)

# 선택된 날짜의 데이터 필터링
train_day_data = train_filtered[train_filtered['date'].dt.date == selected_train_date].copy()
test_day_data = test_filtered[test_filtered['date'].dt.date == selected_test_date].copy()

# 통계 정보 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 선택 데이터 통계")
if len(train_day_data) > 0:
    st.sidebar.metric("Train 데이터 포인트", len(train_day_data))
    st.sidebar.metric(f"Train 평균 {selected_var_display}", f"{train_day_data[train_var].mean():.2f}")
    st.sidebar.metric(f"Train 최대 {selected_var_display}", f"{train_day_data[train_var].max():.2f}")

if len(test_day_data) > 0:
    st.sidebar.metric("Test 데이터 포인트", len(test_day_data))
    st.sidebar.metric(f"Test 평균 {selected_var_display}", f"{test_day_data[test_var].mean():.2f}")
    st.sidebar.metric(f"Test 최대 {selected_var_display}", f"{test_day_data[test_var].max():.2f}")

# 메인 화면 - 두 개의 컬럼으로 분할
col1, col2 = st.columns(2)

# Train 데이터 시각화
with col1:
    st.subheader(f"🟦 Train 데이터 (1-11월)")
    st.markdown(f"**날짜:** {selected_train_date.strftime('%Y-%m-%d')} | **작업상태:** {work_status}")
    
    if len(train_day_data) > 0:
        fig_train = go.Figure()
        
        fig_train.add_trace(go.Scatter(
            x=train_day_data['측정일시'],
            y=train_day_data[train_var],
            mode='lines+markers',
            name='Train Data',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4),
            hovertemplate='<b>시간</b>: %{x|%H:%M}<br>' +
                         f'<b>{selected_var_display}</b>: %{{y:.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig_train.update_layout(
            title=f"{selected_var_display} - Train 데이터",
            xaxis_title="시간",
            yaxis_title=selected_var_display,
            hovermode='x unified',
            height=500,
            showlegend=True,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        st.plotly_chart(fig_train, use_container_width=True)
        
        # 요약 통계
        st.markdown("##### 📊 요약 통계")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("평균", f"{train_day_data[train_var].mean():.2f}")
        with stats_col2:
            st.metric("최소", f"{train_day_data[train_var].min():.2f}")
        with stats_col3:
            st.metric("최대", f"{train_day_data[train_var].max():.2f}")
        with stats_col4:
            st.metric("표준편차", f"{train_day_data[train_var].std():.2f}")
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")

# Test 데이터 시각화
with col2:
    st.subheader(f"🟨 Test 데이터 (12월)")
    st.markdown(f"**날짜:** {selected_test_date.strftime('%Y-%m-%d')} | **작업상태:** {work_status}")
    
    if len(test_day_data) > 0:
        fig_test = go.Figure()
        
        fig_test.add_trace(go.Scatter(
            x=test_day_data['측정일시'],
            y=test_day_data[test_var],
            mode='lines+markers',
            name='Test Data',
            line=dict(color='orange', width=2),
            marker=dict(size=4),
            hovertemplate='<b>시간</b>: %{x|%H:%M}<br>' +
                         f'<b>{selected_var_display}</b>: %{{y:.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig_test.update_layout(
            title=f"{selected_var_display} - Test 데이터",
            xaxis_title="시간",
            yaxis_title=selected_var_display,
            hovermode='x unified',
            height=500,
            showlegend=True,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        st.plotly_chart(fig_test, use_container_width=True)
        
        # 요약 통계
        st.markdown("##### 📊 요약 통계")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("평균", f"{test_day_data[test_var].mean():.2f}")
        with stats_col2:
            st.metric("최소", f"{test_day_data[test_var].min():.2f}")
        with stats_col3:
            st.metric("최대", f"{test_day_data[test_var].max():.2f}")
        with stats_col4:
            st.metric("표준편차", f"{test_day_data[test_var].std():.2f}")
    else:
        st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다.")

# 비교 분석
st.markdown("---")
st.subheader("📉 비교 분석")

if len(train_day_data) > 0 and len(test_day_data) > 0:
    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
    
    with comparison_col1:
        diff_mean = test_day_data[test_var].mean() - train_day_data[train_var].mean()
        st.metric(
            "평균값 차이 (Test - Train)",
            f"{diff_mean:.2f}",
            delta=f"{diff_mean:.2f}"
        )
    
    with comparison_col2:
        diff_max = test_day_data[test_var].max() - train_day_data[train_var].max()
        st.metric(
            "최대값 차이 (Test - Train)",
            f"{diff_max:.2f}",
            delta=f"{diff_max:.2f}"
        )
    
    with comparison_col3:
        if train_day_data[train_var].mean() != 0:
            pct_change = ((test_day_data[test_var].mean() - train_day_data[train_var].mean()) / 
                         train_day_data[train_var].mean() * 100)
            st.metric(
                "평균 변화율 (%)",
                f"{pct_change:.2f}%",
                delta=f"{pct_change:.2f}%"
            )
        else:
            st.metric("평균 변화율 (%)", "N/A")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <small>💡 Tip: 시계열 그래프 하단의 슬라이더를 사용하여 원하는 시간대를 확대해서 볼 수 있습니다.</small>
</div>
""", unsafe_allow_html=True)