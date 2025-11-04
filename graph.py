import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
df = pd.read_csv("./data/train.csv")

# datetime 컬럼 변환
df['datetime'] = pd.to_datetime(df['측정일시'])
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

# 19:00-00:00 데이터만 필터링
night_df = df[(df['hour'] >= 19) & (df['hour'] < 24)]

# 월별 야간 전력사용량 합계 계산
monthly_night_usage = night_df.groupby('month')['전력사용량(kWh)'].sum()

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 설정
fig, ax = plt.subplots(figsize=(12, 6))

# 색상 정의
colors = ['#FF6B6B' if month in [1, 2] else '#3B82F6' for month in monthly_night_usage.index]

# 바 차트 생성
bars = ax.bar(monthly_night_usage.index, monthly_night_usage.values, 
              color=colors, alpha=0.9, edgecolor='none', width=0.7)

# 배경 설정
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# 축 설정
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# 레이블과 제목
ax.set_ylabel('전력사용량 (kWh)', fontsize=12, fontweight='bold', color='#333333', labelpad=10)
ax.set_xlabel('')
ax.set_title('월별 야간 전력사용량 비교\n(19:00-00:00 기준)', fontsize=14, fontweight='bold', color='#222222', pad=20)

# x축 레이블
ax.set_xticks(monthly_night_usage.index)
ax.set_xticklabels([f'{int(i)}월' for i in monthly_night_usage.index], fontsize=11, color='#555555')
ax.tick_params(axis='y', labelsize=11, colors='#555555')

# 그리드
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='#EEEEEE')
ax.set_axisbelow(True)

# y축 범위 설정
ax.set_ylim(0, max(monthly_night_usage.values) * 1.12)

# 평균선 추가
avg_train = monthly_night_usage[3:].mean()
ax.axhline(y=avg_train, color='#FF9500', linestyle='--', linewidth=2.5, alpha=0.7)
ax.text(11.5, avg_train + 300, f'평균: {int(avg_train):,} kWh', fontsize=11, color='#FF9500', fontweight='bold')

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', alpha=0.9, label='제외된 데이터 (1, 2월)'),
    Patch(facecolor='#3B82F6', alpha=0.9, label='학습에 포함된 데이터')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
          frameon=True, fancybox=False, shadow=False, edgecolor='#CCCCCC')

plt.tight_layout()
plt.savefig('night_power_usage.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 통계 출력
print("=" * 60)
print("월별 야간(19:00-00:00) 전력사용량 통계")
print("=" * 60)
for month, value in monthly_night_usage.items():
    status = "(제외됨)" if month in [1, 2] else "(학습)"
    print(f"{int(month)}월: {int(value):,} kWh {status}")
print("=" * 60)
print(f"전체 평균: {monthly_night_usage.mean():,.0f} kWh")
print(f"학습 데이터 평균 (3월~12월): {monthly_night_usage[3:].mean():,.0f} kWh")
print(f"제외 데이터 평균 (1월, 2월): {monthly_night_usage[[1, 2]].mean():,.0f} kWh")
print(f"차이: {(monthly_night_usage[[1, 2]].mean() - monthly_night_usage[3:].mean()):,.0f} kWh")
print("=" * 60)