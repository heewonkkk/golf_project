# YOLO 1차 학습용 데이터 전처리 

# text 파일 포멧
# class center_x center_y width height kpt1_x kpt1_y kpt1_v ...
# kpt1_x: 관절 x 좌표 / 너비
# kpt2_y: 관절 y 좌표 / 높이 
# kpt_v: 잘 보이는지

# 1. yolo 포멧에 맞추어 라벨링 데이터 생성하는 코드
import json
import os

def convert_all_jsons_to_yolo_pose(json_dir, output_dir, log_path="error_log.txt", start = 0):
    os.makedirs(output_dir, exist_ok=True)
    error_logs = []

    all_files = []
    for root, _, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith(".json"):
                all_files.append(os.path.join(root, filename))

    total_files = len(all_files)
    

    for index, json_path in enumerate(all_files[start:]):
        # if index < 60000:
        #     continue  # 6만번째 파일부터 시작
        
        filename = os.path.basename(json_path)
        print(f"[{index+1}/{total_files-start}] 처리 중: {filename}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            msg = f"❌ JSON 로드 실패: {json_path} - {e}"
            print(msg)
            error_logs.append(msg)
            continue

        try:
            image_width, image_height = data["image"]["resolution"]
        except Exception as e:
            msg = f"⚠️ 필드 누락: {json_path} - {e}"
            print(msg)
            error_logs.append(msg)
            continue

        base_name = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, base_name)

        if os.path.exists(output_path):
            print(f"⏭️ 이미 존재함: {output_path} - 건너뜀")
            continue

        boxes = []
        keypoints_list = []

        for annotation in data.get("annotations", []):
            if annotation.get("class") != "person":
                continue

            if "box" in annotation:
                boxes.append(annotation["box"])
            elif "points" in annotation:
                keypoints_list.append(annotation["points"])
            elif "polygon" in annotation:
                # polygon 좌표 리스트 [[x1,y1], [x2,y2], ...] 가 있다고 가정
                poly = annotation["polygon"]
                if not poly:
                    msg = f"⚠️ {filename} - polygon 데이터 비어있음"
                    print(msg)
                    error_logs.append(msg)
                    continue  # 빈 polygon은 건너뜀
                xs = [poly[i] for i in range(0, len(poly), 2)]
                ys = [poly[i] for i in range(1, len(poly), 2)]

                if not xs or not ys:
                    msg = f"⚠️ {filename} - polygon 좌표 이상 (xs or ys 비어있음)"
                    print(msg)
                    error_logs.append(msg)
                    continue
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = x_max - x_min
                h = y_max - y_min
                boxes.append([x_min, y_min, w, h])
                print("polygon으로 bbox를 구했습니다.")

                # polygon이 있지만 keypoints가 없으면 빈 리스트로 채움 (나중에 처리시 오류 방지)
                keypoints_list.append([])

        if len(boxes) != len(keypoints_list):
            msg = f"⚠️ {filename} - box와 keypoints 개수 불일치 (box: {len(boxes)}, points: {len(keypoints_list)})"
            print(msg)
            error_logs.append(msg)
            continue

        output_txt_lines = []

        for box, points in zip(boxes, keypoints_list):
            if not box or not points:
                msg = f"⚠️ {filename} - box 또는 points가 비어있음"
                print(msg)
                error_logs.append(msg)
                continue

            if len(points) % 3 != 0:
                msg = f"⚠️ {filename} - points 길이 이상함 (3의 배수 아님)"
                print(msg)
                error_logs.append(msg)
                continue

            x, y, w, h = box
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height

            keypoints = []
            for i in range(0, len(points), 3):
                kx = points[i] / image_width
                ky = points[i + 1] / image_height
                v = points[i + 2]
                keypoints.extend([kx, ky, v])

            class_id = 0
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + \
                        " ".join(f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]))
            output_txt_lines.append(yolo_line)

        if output_txt_lines:
            try:
                with open(output_path, "w", encoding='utf-8') as out_file:
                    out_file.write("\n".join(output_txt_lines))
                print(f"✅ 변환 완료: {base_name}")
            except Exception as e:
                msg = f"❌ 저장 실패: {output_path} - {e}"
                print(msg)
                error_logs.append(msg)
        else:
            msg = f"❌ {filename} - 처리할 'person' 데이터 없음"
            print(msg)
            error_logs.append(msg)

    if error_logs:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(error_logs) + "\n")
        print(f"\n📝 에러 로그 저장 완료: {log_path}")

# 2. text 파일 내부에 숫자가 53개가 아니면 텍스트 파일을 지우는 코드
import os

def remove_invalid_files(dir_path, expected_num=53):
    files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
    removed_files = []

    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        remove_flag = False
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) != expected_num:
                remove_flag = True
                break
        
        if remove_flag:
            os.remove(file_path)
            removed_files.append(file)
            print(f"삭제됨: {file}")

    print(f"총 삭제된 파일 개수: {len(removed_files)}")

# 경로 설정 후 실행
remove_invalid_files("D:/yolodataset/train/labels", expected_num=53)

# 3. text파일과 이름이 같은 jpg파일을 yolodataset/train/image로 옮기는 코드
import os
import shutil

# 경로 설정
labels_dir = "D:/yolodataset/train/labels"
img_source_root = "D:/원본/Training"
img_target_dir = "D:/yolodataset/train/images"

# 대상 디렉터리가 없다면 생성
os.makedirs(img_target_dir, exist_ok=True)

# 라벨 파일 이름 목록 (확장자 제거)
label_basenames = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt'))

# 하위 디렉터리 포함하여 모든 jpg 파일 탐색
for root, dirs, files in os.walk(img_source_root):
    for file in files:
        if file.lower().endswith('.jpg'):
            basename = os.path.splitext(file)[0]
            if basename in label_basenames:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(img_target_dir, file)
                
                # 이미 존재하면 건너뜀
                if os.path.exists(dst_path):
                    print(f"Skipped (already exists): {file}")
                    continue
                
                shutil.copy2(src_path, dst_path)  # 또는 shutil.copy(src_path, dst_path)
                print(f"Moved: {file}")

# 4. Keypoint 없는 데이터 옮기기
import os
import json
import shutil

json_src_dir = 'D:/swing_for_yolo/train/json'
img_src_dir = 'D:/swing_for_yolo/train/img'

json_dst_dir = 'D:/no_points/json'
img_dst_dir = 'D:/no_points/img'

os.makedirs(json_dst_dir, exist_ok=True)
os.makedirs(img_dst_dir, exist_ok=True)

for root, _, files in os.walk(json_src_dir):
    for file in files:
        if not file.endswith('.json'):
            continue
        
        json_path = os.path.join(root, file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[오류] {json_path} 열기 실패: {e}")
            continue

        move = True
        for ann in data.get('annotations', []):
            if ann.get('class') == 'person' and 'points' in ann:
                move = False
                break  # 하나라도 있으면 더 볼 필요 없음

        if move:
            # JSON 파일 이동
            shutil.move(json_path, os.path.join(json_dst_dir, file))

            # 같은 이름의 이미지 파일도 이동
            img_filename = os.path.splitext(file)[0] + '.jpg'
            for img_root, _, img_files in os.walk(img_src_dir):
                if img_filename in img_files:
                    img_src_path = os.path.join(img_root, img_filename)
                    img_dst_path = os.path.join(img_dst_dir, img_filename)
                    shutil.move(img_src_path, img_dst_path)
                    break  # 찾으면 멈추기
