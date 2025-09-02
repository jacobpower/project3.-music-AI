import numpy as np
import random
import os

# 처리된 데이터 폴더
PROCESSED_DIR = 'processed_dataset'

# 파일 목록 중 하나를 무작위로 선택

random_file = "1925503 ReoNa - Shall we Dance_ (TV Size)_ReoNa - Shall We Dance (Tv size) (N4iveDx) [Easy].npz"
file_path = os.path.join(PROCESSED_DIR, random_file)
print(f"검증할 파일: {random_file}")

# 데이터 로드
data = np.load(file_path)
# 저장된 키와 각 데이터의 모양(shape) 확인
print("저장된 데이터 키:", list(data.keys()))
print("x (입력) shape:", data['x'].shape)
print("y (정답) shape:", data['y'].shape)
print("od (OD) 값:", data['od'])
print("density (밀도) 벡터:", data['density'])

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import os

# (1단계 코드와 이어짐)
# ... 데이터 로드 코드 ...

# --- 시각화 ---
# .npz 파일에 'mel_spectrogram' 키로 스펙트로그램이 저장되어 있다고 가정
# (이를 위해 build_dataset.py에서 np.savez_compressed에 mel_spectrogram=log_mel 추가 필요)

# 여기서는 검증을 위해 스펙트로그램을 다시 계산하는 것으로 예시를 작성합니다.
HPARAMS = {'sampling_rate': 44100, 'n_fft_list': [1024, 2048, 4096], 'n_mels': 80}
RAW_AUDIO_PATH = f"raw_dataset/1925503 ReoNa - Shall we Dance_ (TV Size)/audio.ogg" # 파일 경로 추정

y, sr = librosa.load(RAW_AUDIO_PATH, sr=HPARAMS['sampling_rate'])
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=HPARAMS['n_fft_list'][1], hop_length=sr//100, n_mels=HPARAMS['n_mels'])
log_S = librosa.power_to_db(S, ref=np.max)


# 그래프 그리기
fig, axs = plt.subplots(5, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})

# 1. 스펙트로그램
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', ax=axs[0])
axs[0].set_title(f'Mel Spectrogram: {random_file}')

# 2. 정답 레이블 (4개 키)
y_labels = data['y']
time_axis = np.arange(y_labels.shape[0]) * 0.01 # 프레임을 시간(초)으로 변환

for i in range(4):
    key_idx = i + 1
    # '노트 시작' 채널은 선으로, '노트 지속' 채널은 채워진 영역으로 표시
    axs[key_idx].plot(time_axis, y_labels[:, i, 0], label=f'Key {key_idx} Start', color='cyan', linewidth=2)
    axs[key_idx].fill_between(time_axis, y_labels[:, i, 1], alpha=0.5, label=f'Key {key_idx} Hold', color='magenta')
    axs[key_idx].set_ylabel(f'Key {key_idx}')
    axs[key_idx].legend(loc='upper right')
    axs[key_idx].set_ylim(0, 1.1)

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()