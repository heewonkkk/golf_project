# 모델 평가 1
# 1. 학습 1 결과 다운로드
# 필요한 패키지 설치
!pip install gdown pandas matplotlib seaborn

# csv 다운받기
import gdown

# 구글 드라이브 공유 링크에서 파일 ID 추출
file_id = "1PKo3BnPXfVXF-Rn7evg4kdr2pSP9C7E4"
url = f"https://drive.google.com/uc?id={file_id}"
output = "result.csv"
gdown.download(url, output, quiet=False)

# 2. 학습 1 결과 시각화
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("result.csv")

# mAP50-95 변화 그래프
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="Validation mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("mAP50-95")
plt.title("Validation mAP50-95 over Epochs")
plt.legend()
plt.savefig("mAP_curve.png")
plt.show()

# Loss 곡선
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Validation Box Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# 3. 테스트 세트 예측 및 평가
# 3-1 환경설정 및 모델 다운로드
# 필요 패키지 설치
!pip install ultralytics torch
!pip install gdown

import gdown

# CUDA 버전 확인
!nvidia-smi  # CUDA 12.1 이상 권장

# 모델 다운로드
model_file_id = "1cyqgpapKKJA9woIEYzRCx_VzRM5-Qt5k"
model_url = f"https://drive.google.com/uc?id={model_file_id}"
model_output = "best.pt"
gdown.download(model_url, model_output, quiet=False)

# 3-2 테스트 세트 준비
# 테스트 세트 데이터 구조 
# data.yaml 수정

# 3-3 테스트 세트 예측 및 평가
# 모델 다운로드 (가중치)
model_file_id = "1cyqgpapKKJA9woIEYzRCx_VzRM5-Qt5k"
model_url = f"https://drive.google.com/uc?id={model_file_id}"
model_output = "best.pt"
gdown.download(model_url, model_output, quiet=False)

# 테스트 세트 예측 실행
# 파이썬 명령
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.val(data="data.yaml", split="test", batch=16, device=0)

# 모델 평가 2

# 모델 평가 3
# 1. 학습 2 결과 시각화
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (한글이 깨지는 경우 사용)
plt.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_csv(r"D:\\resized_dataset2\\resized_dataset5\\results.csv")

# 최근 10개 epoch의 성능 확인
print("=== 최근 10개 Epoch 성능 ===")
print(df[["epoch", "metrics/mAP50-95(B)", "metrics/mAP50-95(P)"]].tail(10))

# 1. mAP50-95 변화 그래프 (Box Detection + Pose Estimation)
plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="Box Detection mAP50-95(B)", 
         color='blue', linewidth=2, marker='o', markersize=3)
plt.plot(df["epoch"], df["metrics/mAP50-95(P)"], label="Pose Estimation mAP50-95(P)", 
         color='red', linewidth=2, marker='s', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("mAP50-95")
plt.title("YOLO v8 Pose - Golf Swing Detection Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)  # mAP는 0~1 범위
plt.savefig("golf_pose_mAP_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Box Detection과 Pose Estimation 따로 그리기 (서브플롯)
plt.figure(figsize=(15, 6))

# 서브플롯 1: Box Detection
plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="Box Detection mAP50-95(B)", 
         color='blue', linewidth=2, marker='o', markersize=4)
plt.xlabel("Epoch")
plt.ylabel("mAP50-95(B)")
plt.title("Golf Player Detection Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)

# 서브플롯 2: Pose Estimation
plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["metrics/mAP50-95(P)"], label="Pose Estimation mAP50-95(P)", 
         color='red', linewidth=2, marker='s', markersize=4)
plt.xlabel("Epoch")
plt.ylabel("mAP50-95(P)")
plt.title("Golf Swing Pose Estimation Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)

plt.tight_layout()
plt.savefig("golf_pose_separate_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Loss 곡선 (기존 코드 개선)
plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", 
         color='orange', linewidth=2, alpha=0.8)
plt.plot(df["epoch"], df["val/box_loss"], label="Validation Box Loss", 
         color='purple', linewidth=2, alpha=0.8)

# Pose Loss도 있다면 추가 (있는 경우만)
if "train/pose_loss" in df.columns and "val/pose_loss" in df.columns:
    plt.plot(df["epoch"], df["train/pose_loss"], label="Train Pose Loss", 
             color='green', linewidth=2, alpha=0.8, linestyle='--')
    plt.plot(df["epoch"], df["val/pose_loss"], label="Validation Pose Loss", 
             color='brown', linewidth=2, alpha=0.8, linestyle='--')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss - Golf Pose Detection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("golf_loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. 최종 성능 요약 출력
final_box_map = df["metrics/mAP50-95(B)"].iloc[-1]
final_pose_map = df["metrics/mAP50-95(P)"].iloc[-1]
print(f"\n=== 최종 성능 요약 ===")
print(f"Box Detection mAP50-95(B): {final_box_map:.4f}")
print(f"Pose Estimation mAP50-95(P): {final_pose_map:.4f}")
print(f"총 학습 Epoch: {df['epoch'].iloc[-1]}")

# 2. 
