import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from tqdm import tqdm

# --- train.py에 있는 Dataset 클래스를 그대로 가져옵니다 ---
class OsuDataset(Dataset):
    def __init__(self, file_paths, max_seq_len):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # npz 파일 로드는 에러가 날 수 있으므로 항상 try-except로 감쌉니다.
        try:
            data = np.load(self.file_paths[idx])
            # v1.0.1: 데이터 형식 검증 강화
            required_keys = ['x', 'y']
            if not all(key in data for key in required_keys):
                # 필요한 키가 없으면 에러 발생
                raise KeyError(f"필수 키 {required_keys} 중 일부가 파일에 없습니다.")

            x, y = data['x'], data['y']
        except Exception as e:
            # 파일 로드 실패 시 에러를 발생시켜 바깥에서 잡도록 함
            raise IOError(f"파일 로드 실패: {self.file_paths[idx]}. 원인: {e}")
        
        # 데이터를 max_seq_len 길이로 자르거나 패딩
        current_len = x.shape[0]
        if current_len > self.max_seq_len:
            start = np.random.randint(0, current_len - self.max_seq_len + 1)
            x = x[start:start + self.max_seq_len]
            y = y[start:start + self.max_seq_len]
        elif current_len < self.max_seq_len:
            pad_len = self.max_seq_len - current_len
            
            # x는 4차원이므로, 4개의 차원에 대한 패딩 규칙
            x_pad_width = ((0, pad_len), (0, 0), (0, 0), (0, 0))
            x = np.pad(x, x_pad_width, 'constant', constant_values=0)
            
            # y는 3차원이므로, 3개의 차원에 대한 패딩 규칙
            y_pad_width = ((0, pad_len), (0, 0), (0, 0))
            y = np.pad(y, y_pad_width, 'constant', constant_values=0)
            
        return torch.FloatTensor(x), torch.FloatTensor(y)

# --- 메인 검증 로직 ---
if __name__ == '__main__':
    PROCESSED_DIR = 'processed_dataset'
    FILE_LIST_PATH = 'valid_files.txt'
    # train.py의 CONFIG와 동일한 값을 사용해야 함
    MAX_SEQ_LEN = 512 

    all_files = glob.glob(os.path.join(PROCESSED_DIR, '*.npz'))
    if not all_files:
        print(f"오류: '{PROCESSED_DIR}'에 처리된 파일이 없습니다.")
    else:
        print(f"총 {len(all_files)}개의 파일을 대상으로 전체 데이터 파이프라인을 검증합니다...")
        
        # 먼저 Dataset 객체를 생성
        temp_dataset = OsuDataset(all_files, MAX_SEQ_LEN)
        
        valid_files = []
        for i in tqdm(range(len(temp_dataset)), desc="데이터 로딩 테스트 중"):
            try:
                # __getitem__을 직접 호출하여 로딩, 슬라이싱, 패딩, 텐서 변환까지 모두 테스트
                temp_dataset[i]
                # 위 라인에서 에러가 발생하지 않으면, 해당 파일은 학습에 사용 가능
                valid_files.append(all_files[i])
            except Exception as e:
                print(f"\n파이프라인 오류 발견! 파일 제외: {os.path.basename(all_files[i])}")
                print(f"  - 오류 내용: {e}")

        print(f"\n검증 완료. 총 {len(all_files)}개 중 {len(valid_files)}개의 유효한 파일을 찾았습니다.")

        with open(FILE_LIST_PATH, 'w', encoding='utf-8') as f:
            for path in valid_files:
                f.write(f"{path}\n")
        
        print(f"최종적으로 검증된 파일 목록이 '{FILE_LIST_PATH}'에 저장되었습니다.")