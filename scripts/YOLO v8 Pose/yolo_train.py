# YOLO v8 Pose 학습 코드

# 1차 학습
#!/usr/bin/env python3
"""
Golf Pose Estimation - Baseline Training Script
1차 학습: 200 에포크 풀 트레이닝
"""

import os
from ultralytics import YOLO

def train_baseline_model():
    """
    YOLO pose 모델 1차 학습 (Baseline)
    - 200 에포크 풀 트레이닝
    - 사전 훈련된 가중치 미사용
    """
    
    # 학습 파라미터 설정
    config = {
        'data': '/content/drive/MyDrive/golf/data.yaml',
        'model': '/content/drive/MyDrive/golf/custom.yaml',
        'epochs': 200,
        'batch': 16,
        'imgsz': 480,
        'lr0': 0.01,
        'device': 0,
        'pretrained': False,
        'workers': 4,
        'amp': False,
        'cache': True,
        'save': True,
        'project': '/content/drive/MyDrive/golf/runs',
        'name': 'exp22',
        'save_period': 1
    }
    
    print("=" * 60)
    print("Golf Pose Estimation - 1차 학습 (Baseline)")
    print("=" * 60)
    print(f"실험명: {config['name']}")
    print(f"에포크: {config['epochs']}")
    print(f"배치 크기: {config['batch']}")
    print(f"이미지 크기: {config['imgsz']}")
    print(f"학습률: {config['lr0']}")
    print(f"사전 훈련: {config['pretrained']}")
    print("=" * 60)
    
    try:
        # YOLO 모델 초기화
        model = YOLO()
        
        # 학습 시작
        print("학습을 시작합니다...")
        results = model.train(
            data=config['data'],
            model=config['model'],
            epochs=config['epochs'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            lr0=config['lr0'],
            device=config['device'],
            pretrained=config['pretrained'],
            workers=config['workers'],
            amp=config['amp'],
            cache=config['cache'],
            save=config['save'],
            project=config['project'],
            name=config['name'],
            save_period=config['save_period']
        )
        
        print("=" * 60)
        print("1차 학습이 완료되었습니다!")
        print(f"결과 저장 경로: {config['project']}/{config['name']}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {str(e)}")
        raise

def main():
    """메인 함수"""
    # Google Drive 마운트 확인
    drive_path = '/content/drive/MyDrive/golf'
    if not os.path.exists(drive_path):
        print("Google Drive가 마운트되지 않았습니다.")
        print("먼저 다음 코드를 실행하세요:")
        print("from google.colab import drive")
        print("drive.mount('/content/drive')")
        return
    
    # 필요한 디렉토리 생성
    os.makedirs('/content/drive/MyDrive/golf/runs', exist_ok=True)
    
    # 1차 학습 실행
    train_baseline_model()

if __name__ == "__main__":
    main()


# 2차 학습
#!/usr/bin/env python3
"""
Golf Pose Estimation - Fine-tuning with Early Stopping
2차 학습: 조기 종료를 포함한 효율적 학습
"""

import os
from ultralytics import YOLO

def train_with_early_stopping():
    """
    YOLO pose 모델 2차 학습 (조기 종료 포함)
    - 최대 100 에포크
    - patience=10으로 조기 종료
    - 과적합 방지 및 효율적 학습
    """
    
    # 학습 파라미터 설정
    config = {
        'data': '/content/drive/MyDrive/golf/data.yaml',
        'model': '/content/drive/MyDrive/golf/custom.yaml',
        'epochs': 100,
        'patience': 10,
        'batch': 16,
        'imgsz': 480,
        'lr0': 0.01,
        'device': 0,
        'pretrained': False,
        'workers': 4,
        'amp': False,
        'cache': True,
        'save': True,
        'project': '/content/drive/MyDrive/golf/runs',
        'name': 'exp5',
        'save_period': 1
    }
    
    print("=" * 60)
    print("Golf Pose Estimation - 2차 학습 (Early Stopping)")
    print("=" * 60)
    print(f"실험명: {config['name']}")
    print(f"최대 에포크: {config['epochs']}")
    print(f"조기 종료 patience: {config['patience']}")
    print(f"배치 크기: {config['batch']}")
    print(f"이미지 크기: {config['imgsz']}")
    print(f"학습률: {config['lr0']}")
    print(f"사전 훈련: {config['pretrained']}")
    print("=" * 60)
    
    try:
        # YOLO 모델 초기화
        model = YOLO()
        
        # 학습 시작
        print("조기 종료를 포함한 학습을 시작합니다...")
        print(f"성능 개선이 {config['patience']} 에포크 동안 없으면 자동 중단됩니다.")
        
        results = model.train(
            data=config['data'],
            model=config['model'],
            epochs=config['epochs'],
            patience=config['patience'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            lr0=config['lr0'],
            device=config['device'],
            pretrained=config['pretrained'],
            workers=config['workers'],
            amp=config['amp'],
            cache=config['cache'],
            save=config['save'],
            project=config['project'],
            name=config['name'],
            save_period=config['save_period']
        )
        
        print("=" * 60)
        print("2차 학습이 완료되었습니다!")
        print(f"결과 저장 경로: {config['project']}/{config['name']}")
        
        # 학습 결과 요약
        if hasattr(results, 'epochs_trained'):
            print(f"실제 학습된 에포크: {results.epochs_trained}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {str(e)}")
        raise

def check_previous_training():
    """이전 학습 결과 확인"""
    exp22_path = '/content/drive/MyDrive/golf/runs/exp22'
    if os.path.exists(exp22_path):
        print(f"✓ 1차 학습 결과 확인: {exp22_path}")
        return True
    else:
        print("⚠ 1차 학습 결과를 찾을 수 없습니다.")
        print("먼저 train_baseline.py를 실행하여 1차 학습을 완료해주세요.")
        return False

def main():
    """메인 함수"""
    # Google Drive 마운트 확인
    drive_path = '/content/drive/MyDrive/golf'
    if not os.path.exists(drive_path):
        print("Google Drive가 마운트되지 않았습니다.")
        print("먼저 다음 코드를 실행하세요:")
        print("from google.colab import drive")
        print("drive.mount('/content/drive')")
        return
    
    # 필요한 디렉토리 생성
    os.makedirs('/content/drive/MyDrive/golf/runs', exist_ok=True)
    
    # 이전 학습 결과 확인 (선택적)
    check_previous_training()
    
    # 2차 학습 실행
    train_with_early_stopping()

if __name__ == "__main__":
    main()