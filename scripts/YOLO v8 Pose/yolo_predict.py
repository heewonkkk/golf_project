# 모델 예측-1차 학습
# 모델 로드 (이미 학습된 best.pt)
model = YOLO("best.pt")

# 예측 실행
results = model.predict(
    source="D:/resized_dataset2/test/images",  # 예측할 이미지 폴더 또는 이미지 파일 경로
    save=True,                                 # 결과 이미지 저장
    imgsz=640,                                 # 입력 이미지 크기 (선택)
    conf=0.25,                                 # confidence threshold (선택)
    device=0                                   # GPU 사용 (0번 GPU)
)

results = model.predict(
    source="D:/resized_dataset2/test/images",
    save=True,
    save_txt=True,
    imgsz=640,
    device=0,
    project="D:\\resized_dataset2\\pre",  # 원하는 상위 폴더
    name="exp23"                        # 하위 폴더명
)

# 모델 예측-2차 학습
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"D:\resized_dataset2\resized_dataset5\best.pt")

# 2. 예측 실행
model.predict(
    source=r"D:\resized_dataset2\resized_dataset5\predict\images",  # 예측할 이미지 폴더
    save=True,                   # 결과 이미지 저장
    save_txt=True,               # 결과 라벨(txt) 저장
    project=r"D:\resized_dataset2\resized_dataset5",  # 결과 저장할 상위 폴더
    name="predict_result",       # 결과가 저장될 하위 폴더명
    exist_ok=True,                # 이미 폴더가 있어도 덮어쓰기 허용
    batch=16,              # batch size 조정
    conf=0.5               # confidence threshold 조정
)

# 정답과 예측 비교
import os
import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕 폰트 설정
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

# 경로 설정
img_dir = r"D:\resized_dataset2\resized_dataset5\test\images"
gt_dir = r"D:\resized_dataset2\resized_dataset5\test\labels"
pred_dir = r"D:\resized_dataset2\resized_dataset5\predict_result\labels"

class_names = ["class0", "class1", "class2"]  # 실제 클래스명으로 수정

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # 예측 라벨(txt)은 6개, 정답 라벨(txt)은 5개일 수 있음
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                boxes.append((int(cls), x, y, w, h))
    return boxes

def draw_boxes(img, boxes, color, class_names):
    img = img.copy()
    h, w, _ = img.shape
    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[cls], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def plot_comparison(img_name):
    img_path = os.path.join(img_dir, img_name)
    gt_path = os.path.join(gt_dir, img_name.replace('.jpg', '.txt'))
    pred_path = os.path.join(pred_dir, img_name.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_boxes = load_yolo_labels(gt_path)
    pred_boxes = load_yolo_labels(pred_path)

    img_gt = draw_boxes(img, gt_boxes, (0, 0, 255), class_names)      # 파란색(정답)
    img_pred = draw_boxes(img, pred_boxes, (255, 0, 0), class_names)  # 빨간색(예측)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("원본")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_gt)
    plt.title("정답(GT)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.title("예측(Prediction)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 사용 예시
img_name = "legend_swing16_00003.jpg" 
plot_comparison(img_name)