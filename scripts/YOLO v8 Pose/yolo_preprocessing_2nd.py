# YOLO 2차 학습용 데이터 전처리 
# - 바운딩 박스의 이미지 대비 차지 비율 확인 / 바운딩 박스 제외한 배경 이미지 크롭
# - 바운딩 박스와 기존 이미지 (직사각형) 의 비율은 유지하고 패딩없이 크롭
# - 이미지가 조정됨에 따라 이동하는 관절 좌표에 맞게 labels 폴더 txt 데이터 수정

# 1. train 셋 images/label 데이터 crop 코드
import os
import cv2
import numpy as np
from tqdm import tqdm

# 설정
input_image_dir = "D:\\resized_dataset2\\resized_data22\\images"
input_label_dir = "D:\\resized_dataset2\\resized_data22\\labels"
output_image_dir = "D:\\resized_dataset2\\resized_data23\\images"
output_label_dir = "D:\\resized_dataset2\\resized_data23\\labels"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

target_w, target_h = 960, 560  # 출력 이미지 크기

def yolo_to_pixel(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2

def pixel_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_c = ((x1 + x2) / 2) / img_w
    y_c = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_c, y_c, w, h

def process_keypoints(keypoints, original_w, original_h, crop_x1, crop_y1, cropped_w, cropped_h):
    new_keypoints = []
    keypoint_format = 3
    num_keypoints = len(keypoints) // keypoint_format

    for i in range(num_keypoints):
        idx = i * keypoint_format
        kx = keypoints[idx] * original_w
        ky = keypoints[idx + 1] * original_h
        visibility = keypoints[idx + 2]

        new_kx = (kx - crop_x1) / cropped_w
        new_ky = (ky - crop_y1) / cropped_h

        new_kx = min(max(new_kx, 0.0), 1.0)
        new_ky = min(max(new_ky, 0.0), 1.0)

        if 0 <= new_kx <= 1 and 0 <= new_ky <= 1 and visibility > 0:
            new_keypoints.extend([new_kx, new_ky, visibility])
        else:
            new_keypoints.extend([new_kx, new_ky, 0.0])
    return new_keypoints

image_files = [f for f in os.listdir(input_image_dir) if f.endswith(".jpg")]
skipped_files = []

for fname in tqdm(image_files, desc="처리 중"):
    img_path = os.path.join(input_image_dir, fname)
    label_path = os.path.join(input_label_dir, fname.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        print(f"[경고] 라벨 누락: {label_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[에러] 이미지 로드 실패: {img_path}")
        continue

    h, w, _ = img.shape
    with open(label_path, 'r') as f:
        lines = f.readlines()

    skip_this_file = False

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"[경고] 잘못된 라벨 형식: {line.strip()}")
            continue

        class_id = int(parts[0])
        bbox = list(map(float, parts[1:5]))
        keypoints = list(map(float, parts[5:]))

        x1, y1, x2, y2 = yolo_to_pixel(bbox, w, h)

        # 관절 포함 bbox 확장
        if keypoints:
            keypoint_format = 3
            num_kp = len(keypoints) // keypoint_format
            kps = [(keypoints[i*3] * w, keypoints[i*3+1] * h) for i in range(num_kp) if keypoints[i*3+2] > 0]
            if kps:
                kps_x = [kp[0] for kp in kps]
                kps_y = [kp[1] for kp in kps]
                x1 = int(min(x1, min(kps_x)))
                y1 = int(min(y1, min(kps_y)))
                x2 = int(max(x2, max(kps_x)))
                y2 = int(max(y2, max(kps_y)))

        box_w, box_h = x2 - x1, y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 타겟 비율 유지하며 크롭 크기 계산
        target_bbox_ratio = 0.7
        crop_h = int(box_h / target_bbox_ratio)
        crop_w = int(crop_h * (target_w / target_h))

        new_x1 = center_x - crop_w // 2
        new_y1 = center_y - crop_h // 2
        new_x2 = center_x + crop_w // 2
        new_y2 = center_y + crop_h // 2

        # 예외 조건 확인
        if new_x1 < 0 or new_y1 < 0 or new_x2 > w or new_y2 > h:
            skipped_files.append(fname)
            skip_this_file = True
            break

        cropped = img[new_y1:new_y2, new_x1:new_x2]
        resized = cv2.resize(cropped, (target_w, target_h))

        # bbox 변환
        new_bbox = pixel_to_yolo(
            (x1 - new_x1, y1 - new_y1, x2 - new_x1, y2 - new_y1),
            crop_w, crop_h
        )

        # keypoint 변환
        if len(keypoints) > 0:
            new_keypoints = process_keypoints(keypoints, w, h, new_x1, new_y1, crop_w, crop_h)
        else:
            new_keypoints = []

        # 파일명 변경: {원래이름}_re_{idx}.확장자
        base_name = os.path.splitext(fname)[0]  # 확장자 제거
        save_img_path = os.path.join(output_image_dir, f"{base_name}_re_{idx}.jpg")
        save_label_path = os.path.join(output_label_dir, f"{base_name}_re_{idx}.txt")

        cv2.imwrite(save_img_path, resized)
        with open(save_label_path, 'w') as f:
            bbox_str = ' '.join(map(str, new_bbox))
            keypoint_str = ' '.join(map(str, new_keypoints)) if new_keypoints else ''
            if keypoint_str:
                f.write(f"{class_id} {bbox_str} {keypoint_str}\n")
            else:
                f.write(f"{class_id} {bbox_str}\n")

    if skip_this_file:
        continue

print("\n처리 완료")

if skipped_files:
    print(f"\n[예외 파일 총 {len(skipped_files)}개]")
    for sf in skipped_files:
        print(sf)
else:
    print("예외 파일 없음")

# -bounding box(bbox)를 기준으로 크롭
# -bbox는 관절(keypoints)을 포함하도록 확장
# ⇒ **bbox + 관절을 모두 포함하는 사각형 기준**으로 크롭

# 2. 이미지랑 라벨 이름 비교해서 맞춰보기
import os

# 경로 설정
image_dir = r"D:\\resized_dataset2\\crop\\train\\images"
label_dir = r"D:\\resized_dataset2\\crop\\train\\labels"

# 확장자 제거한 파일명만 추출
image_names = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")}
label_names = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")}

# 불일치 확인
images_without_labels = image_names - label_names
labels_without_images = label_names - image_names

# 결과 출력
print(f"🔍 이미지만 있고 라벨 없는 파일 수: {len(images_without_labels)}")
for name in sorted(images_without_labels):
    print(f" - {name}.jpg")

print(f"\n🔍 라벨만 있고 이미지 없는 파일 수: {len(labels_without_images)}")
for name in sorted(labels_without_images):
    print(f" - {name}.txt")

# 3. 데이터 분배 코드
# - 원본 데이터 : 135,876
# - 크롭 데이터 : 109,780
# - 데이터 분배 결과 : train: 98,093, val: 10,765, test: 24,158
import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# 경로 설정
original_img_dir = r'D:\resized_dataset2\resized_data22\images'
original_lbl_dir = r'D:\resized_dataset2\resized_data22\labels'
crop_img_dir = r'D:\resized_dataset2\resized_data23\images'
crop_lbl_dir = r'D:\resized_dataset2\resized_data23\labels'

output_base = r'D:\resized_dataset2\resized_data5'
output_dirs = {
    'train_img': os.path.join(output_base, 'train', 'images'),
    'train_lbl': os.path.join(output_base, 'train', 'labels'),
    'val_img': os.path.join(output_base, 'val', 'images'),
    'val_lbl': os.path.join(output_base, 'val', 'labels'),
    'test_img': os.path.join(output_base, 'test', 'images'),
    'test_lbl': os.path.join(output_base, 'test', 'labels'),
    'failed_img': os.path.join(output_base, 'failed', 'images'),
    'failed_lbl': os.path.join(output_base, 'failed', 'labels'),
}

for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

TARGET_TOTAL = 135_876
ORIGINAL_RATIO = 0.3
CROP_RATIO = 0.7
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

def get_swingid_from_original(name):
    return '_'.join(name.split('_')[:-1])

def get_swingid_from_crop(name):
    return '_'.join(name.split('_')[:-3])

def collect_files(img_dir, lbl_dir, swingid_func):
    swingid_to_files = defaultdict(list)
    for fname in os.listdir(img_dir):
        if fname.endswith('.jpg'):
            name = os.path.splitext(fname)[0]
            swingid = swingid_func(name)
            lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
            if os.path.exists(lbl_path):
                swingid_to_files[swingid].append(fname)
    return swingid_to_files

def select_swingids_for_ratio(orig_ids, crop_ids, orig_count, crop_count, target_total):
    # 3:7 비율로 swing id를 뽑는데, 프레임 수가 아니라 swing id 개수 기준으로 맞춤
    orig_target = int(target_total * ORIGINAL_RATIO)
    crop_target = target_total - orig_target

    # swing id를 랜덤 셔플
    random.shuffle(orig_ids)
    random.shuffle(crop_ids)

    # 프레임 수가 많은 swing id부터 뽑기(조금이라도 목표에 근접하게)
    orig_ids_sorted = sorted(orig_ids, key=lambda sid: -orig_count[sid])
    crop_ids_sorted = sorted(crop_ids, key=lambda sid: -crop_count[sid])

    selected_orig = []
    selected_crop = []
    orig_sum = 0
    crop_sum = 0

    for sid in orig_ids_sorted:
        if orig_sum + orig_count[sid] > orig_target:
            continue
        selected_orig.append(sid)
        orig_sum += orig_count[sid]
        if orig_sum >= orig_target:
            break

    for sid in crop_ids_sorted:
        if crop_sum + crop_count[sid] > crop_target:
            continue
        # 원본에서 이미 선택된 swing id는 제외
        if sid in selected_orig:
            continue
        selected_crop.append(sid)
        crop_sum += crop_count[sid]
        if crop_sum >= crop_target:
            break

    # 부족분은 남은 id에서 채움
    if orig_sum < orig_target:
        for sid in orig_ids_sorted:
            if sid not in selected_orig:
                selected_orig.append(sid)
                orig_sum += orig_count[sid]
                if orig_sum >= orig_target:
                    break
    if crop_sum < crop_target:
        for sid in crop_ids_sorted:
            if sid not in selected_crop and sid not in selected_orig:
                selected_crop.append(sid)
                crop_sum += crop_count[sid]
                if crop_sum >= crop_target:
                    break

    return selected_orig, selected_crop

def split_swingids(swingid_list):
    random.shuffle(swingid_list)
    n = len(swingid_list)
    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)
    test_n = n - train_n - val_n
    train_ids = set(swingid_list[:train_n])
    val_ids = set(swingid_list[train_n:train_n+val_n])
    test_ids = set(swingid_list[train_n+val_n:])
    return train_ids, val_ids, test_ids

def copy_files(swingid_to_files, swingid_set, img_dir, lbl_dir, split_ids, split_name, failed_imgs, failed_lbls, pbar):
    count = 0
    for sid in swingid_set:
        if sid not in split_ids:
            continue
        for fname in swingid_to_files[sid]:
            img_src = os.path.join(img_dir, fname)
            lbl_src = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
            img_dst = os.path.join(output_dirs[f'{split_name}_img'], fname)
            lbl_dst = os.path.join(output_dirs[f'{split_name}_lbl'], fname.replace('.jpg', '.txt'))
            try:
                if os.path.exists(img_src) and os.path.exists(lbl_src):
                    shutil.copy(img_src, img_dst)
                    shutil.copy(lbl_src, lbl_dst)
                    count += 1
                else:
                    # 실패 파일 복사
                    if os.path.exists(img_src):
                        shutil.copy(img_src, output_dirs['failed_img'])
                        failed_imgs.append(fname)
                    if os.path.exists(lbl_src):
                        shutil.copy(lbl_src, output_dirs['failed_lbl'])
                        failed_lbls.append(fname.replace('.jpg', '.txt'))
            except Exception as e:
                failed_imgs.append(fname)
                failed_lbls.append(fname.replace('.jpg', '.txt'))
            pbar.update(1)
    return count

def main():
    print("데이터 집계 중...")
    orig_files = collect_files(original_img_dir, original_lbl_dir, get_swingid_from_original)
    crop_files = collect_files(crop_img_dir, crop_lbl_dir, get_swingid_from_crop)
    orig_ids = list(orig_files.keys())
    crop_ids = list(crop_files.keys())
    orig_count = {sid: len(orig_files[sid]) for sid in orig_ids}
    crop_count = {sid: len(crop_files[sid]) for sid in crop_ids}

    # 3:7 비율로 swing id 추출
    selected_orig, selected_crop = select_swingids_for_ratio(orig_ids, crop_ids, orig_count, crop_count, TARGET_TOTAL)
    print(f"선택된 원본 swing id: {len(selected_orig)}개, 크롭 swing id: {len(selected_crop)}개")
    print(f"원본 프레임 합: {sum(orig_count[sid] for sid in selected_orig)}")
    print(f"크롭 프레임 합: {sum(crop_count[sid] for sid in selected_crop)}")

    # 분할
    all_selected = selected_orig + selected_crop
    train_ids, val_ids, test_ids = split_swingids(all_selected)
    print(f"train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)} (swing id 기준)")

    # 전체 복사 개수 계산
    total_files = sum(len(orig_files[sid]) for sid in selected_orig if sid in train_ids|val_ids|test_ids) + \
                  sum(len(crop_files[sid]) for sid in selected_crop if sid in train_ids|val_ids|test_ids)
    print(f"총 복사 대상 이미지: {total_files}")

    failed_imgs = []
    failed_lbls = []

    with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
        train_count = copy_files(orig_files, selected_orig, original_img_dir, original_lbl_dir, train_ids, 'train', failed_imgs, failed_lbls, pbar)
        train_count += copy_files(crop_files, selected_crop, crop_img_dir, crop_lbl_dir, train_ids, 'train', failed_imgs, failed_lbls, pbar)
        val_count = copy_files(orig_files, selected_orig, original_img_dir, original_lbl_dir, val_ids, 'val', failed_imgs, failed_lbls, pbar)
        val_count += copy_files(crop_files, selected_crop, crop_img_dir, crop_lbl_dir, val_ids, 'val', failed_imgs, failed_lbls, pbar)
        test_count = copy_files(orig_files, selected_orig, original_img_dir, original_lbl_dir, test_ids, 'test', failed_imgs, failed_lbls, pbar)
        test_count += copy_files(crop_files, selected_crop, crop_img_dir, crop_lbl_dir, test_ids, 'test', failed_imgs, failed_lbls, pbar)

    print(f"\ntrain: {train_count}, val: {val_count}, test: {test_count}")
    print(f"실패 이미지: {len(failed_imgs)}, 실패 라벨: {len(failed_lbls)}")
    with open(os.path.join(output_base, 'failed_files.txt'), 'w', encoding='utf-8') as f:
        for fname in failed_imgs + failed_lbls:
            f.write(fname + '\n')
    print("실패 파일은 failed 폴더와 failed_files.txt에 저장됨.")

if __name__ == "__main__":
    main()