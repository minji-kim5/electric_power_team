import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
df = pd.read_csv("./data/train.csv")

# datetime 컬럼 변환
df['datetime'] = pd.to_datetime(df['측정일시'])
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

# 월별, 시간대별 평균 전력사용량 계산
hourly_monthly = df.groupby(['month', 'hour'])['전력사용량(kWh)'].mean().reset_index()

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 설정
fig, ax = plt.subplots(figsize=(14, 7))

# 컬러맵 생성 (1월~11월, 12개 색상)
colors = plt.cm.tab20(np.linspace(0, 1, 11))

# 월별로 라인 그리기
for month in range(1, 12):
    month_data = hourly_monthly[hourly_monthly['month'] == month]
    ax.plot(month_data['hour'], month_data['전력사용량(kWh)'], 
            marker='o', linewidth=2.5, markersize=5, 
            label=f'{month}월', color=colors[month-1], alpha=0.85)

# 활동 구간 배경색 표시
sections = [
    (8, 9.25, '준비', '#FFB3B3'),
    (9.25, 12, '오전', '#FFE5CC'),
    (12, 13.25, '점심', '#FFFFB3'),
    (13.25, 17.25, '오후', '#B3FFB3'),
    (17.25, 18.5, '퇴근', '#B3E5FF'),
    (18.5, 21, '야간초입', '#E5CCFF'),
    (21, 24, '야간', '#CCCCCC'),
]

for start, end, label, color in sections:
    ax.axvspan(start, end, alpha=0.35, color=color)
    # 경계선 그리기
    ax.axvline(x=start, color='gray', linewidth=1.5, alpha=0.5, linestyle='-')
    mid = (start + end) / 2
    ax.text(mid, ax.get_ylim()[1] * 1.1, label, 
            ha='center', va='top', fontsize=9, fontweight='bold', color='#333333')
    
# 축 설정
ax.set_xlabel('시간 (Hour)', fontsize=12, fontweight='bold', color='#333333', labelpad=10)
ax.set_ylabel('평균 전력사용량 (kWh)', fontsize=12, fontweight='bold', color='#333333', labelpad=10)
ax.set_title('시간대별 평균 전력사용량 패턴', 
             fontsize=14, fontweight='bold', color='#222222', pad=20)

# x축 설정
ax.set_xticks(range(0, 24, 1))
ax.set_xticklabels([f'{i}' for i in range(24)], fontsize=10)

# 그리드
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#CCCCCC')
ax.set_axisbelow(True)

ax.set_ylim(0, max(hourly_monthly['전력사용량(kWh)']) * 1.2)

# 범례
ax.legend(loc='upper left', ncol=2, fontsize=12, framealpha=0.95, edgecolor='#CCCCCC')

# 배경
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

plt.tight_layout()
plt.savefig('monthly_hourly_pattern.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 통계 출력
print("=" * 80)
print("월별 시간대별 전력사용량 통계")
print("=" * 80)
print("\n시간대별 평균 전력사용량:")
hourly_avg = df.groupby('hour')['전력사용량(kWh)'].mean().round(2)
for hour, usage in hourly_avg.items():
    print(f"{int(hour):2d}시: {usage:8.2f} kWh", end="  ")
    if (hour + 1) % 4 == 0:
        print()
print("\n" + "=" * 80)
print("월별 평균 전력사용량:")
monthly_avg = df.groupby('month')['전력사용량(kWh)'].mean().round(2)
for month, usage in monthly_avg.items():
    print(f"{int(month)}월: {usage:8.2f} kWh")
print("=" * 80)