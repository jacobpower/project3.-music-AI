import os
import numpy as np
import librosa
import math
import glob
from tqdm import tqdm
import soundfile as sf # 테스트용

# ==============================================================================
# 헬퍼 함수 (Helper Functions)
# ==============================================================================

def is_valid_4k_mania_chart(osu_file_path):
    is_mania, is_declared_4k = False, False
    hit_objects_x_coords = set()
    try:
        with open(osu_file_path, 'r', encoding='utf-8') as f:
            in_difficulty_section, in_hit_objects_section = False, False
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('['):
                    in_difficulty_section = (line == '[Difficulty]')
                    in_hit_objects_section = (line == '[HitObjects]')
                if line.strip() == 'Mode: 3': 
                    is_mania = True
                if in_difficulty_section and line.strip() == 'CircleSize:4': 
                    is_declared_4k = True
                if in_hit_objects_section:
                    try: hit_objects_x_coords.add(int(line.split(',')[0]))
                    except (ValueError, IndexError): \
                        continue
    except Exception: 
        return False
    
    if not (is_mania and is_declared_4k): return False
    if len(hit_objects_x_coords) != 4: return False
    return True

def process_mania_beatmap(osu_file_path, song_duration_seconds, hparams):
    X_COORDINATE_TO_KEY_INDEX = {64: 0, 192: 1, 320: 2, 448: 3}
    hit_objects, overall_difficulty = [], 5.0
    in_difficulty, in_hit_objects = False, False
    with open(osu_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('['):
                in_difficulty = (line == '[Difficulty]')
                in_hit_objects = (line == '[HitObjects]')
            if in_difficulty and line.startswith('OverallDifficulty:'):
                overall_difficulty = float(line.split(':')[1])
            if in_hit_objects:
                parts = line.split(',')
                if len(parts) >= 6:
                    x_coord, timestamp_ms, type = int(parts[0]), int(parts[2]), int(parts[3])
                    if x_coord in X_COORDINATE_TO_KEY_INDEX:
                        is_long_note = type == 128
                        if is_long_note:
                            end_time_ms = int(parts[5].split(":")[0])
                        else:
                            end_time_ms = timestamp_ms
                        hit_objects.append({'time': timestamp_ms, 'end_time':end_time_ms, 'x': X_COORDINATE_TO_KEY_INDEX[x_coord]})

    normalized_od = overall_difficulty / 10.0
    total_frames = int(song_duration_seconds * 100)
    
    if not hit_objects:
        density_vec = np.zeros(10); density_vec[0] = 1
        return np.zeros((total_frames, 4)), normalized_od, density_vec
    
    note_count = len(hit_objects)
    last_note_time_sec = hit_objects[-1]['time'] / 1000.0
    density = note_count / last_note_time_sec if last_note_time_sec > 0 else 0
    
    if density < 1: 
        level = 0
    elif density >= 9: 
        level = 9
    else: 
        level = int(math.floor(density))
    density_vec = np.zeros(10)
    density_vec[level] = 1.0

    binary_labels = np.zeros((total_frames, 4, 2)) # 각 프레임에 노트가 있는지 없는지. 롱노트 처리를 위해서 (프레임, 키, [시작, 지속]) 이런식으로 되어 있음
    for obj in hit_objects:
        start_idx = obj['time'] // 10
        end_idx = obj['end_time'] // 10
        key_idx = obj['x']
        binary_labels[start_idx, key_idx, 0] = 1.0
        if end_idx > start_idx:
            effective_end_frame = min(end_idx, total_frames)
            binary_labels[start_idx:effective_end_frame, key_idx, 1] = 1.0
            
        fuzzy_labels = np.zeros_like(binary_labels)
    fuzzy_len = hparams['fuzzy_label_length']
    sigma = fuzzy_len / 2.0
    kernel_radius = fuzzy_len * 2
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma)**2)

    for channel in range(2):
        for key_idx in range(4):
            key_binary_labels = binary_labels[:, key_idx, channel]
            key_fuzzy_labels = np.zeros(total_frames)
            action_indices = np.where(key_binary_labels == 1.0)[0]
            
            for idx in action_indices:
                start, end = idx - kernel_radius, idx + kernel_radius + 1
                k_start = max(0, -start)
                k_end = len(gaussian_kernel) - max(0, end - total_frames)
                s, e = max(0, start), min(total_frames, end)
                key_fuzzy_labels[s:e] = np.maximum(key_fuzzy_labels[s:e], gaussian_kernel[k_start:k_end])
            
            fuzzy_labels[:, key_idx, channel] = key_fuzzy_labels
        
    return fuzzy_labels, normalized_od, density_vec

def process_audio(audio_file_path, hparams):
    sr = hparams['sampling_rate']
    y, _ = librosa.load(audio_file_path, sr=sr)
    hop_length = sr // 100 # 10ms hop length
    
    specs = [librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=hparams['n_mels'], center=False)) for n_fft in hparams['n_fft_list']]
    min_frames = min(s.shape[1] for s in specs)
    specs = [s[:, :min_frames] for s in specs]
    
    log_mel = np.stack(specs, axis=0).transpose(2, 1, 0)
    
    n_ctx, n_mels, n_ch = hparams['n_context'], hparams['n_mels'], len(hparams['n_fft_list'])
    num_frames = log_mel.shape[0]
    padded = np.pad(log_mel, ((n_ctx, n_ctx), (0, 0), (0, 0)), mode='constant')
    
    ctx_win = np.lib.stride_tricks.as_strided(padded, shape=(num_frames, 2*n_ctx+1, n_mels, n_ch), strides=(padded.strides[0], padded.strides[0], padded.strides[1], padded.strides[2]), writeable=False)

    
    context_window = ctx_win.transpose(0, 3, 1, 2)

    return context_window

# ==============================================================================
# 메인 실행 함수 (Main Runner)
# ==============================================================================

def create_processed_dataset(source_dir, target_dir, hparams):
    os.makedirs(target_dir, exist_ok=True)
    song_folders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
    
    print(f"총 {len(song_folders)}개의 곡 폴더를 대상으로 필터링 및 전처리를 시작합니다.")
    
    for song_folder in tqdm(song_folders, desc="데이터셋 구축 중"):
        try:
            osu_files = glob.glob(os.path.join(song_folder, '*.osu'))
            
            valid_osu_files = [p for p in osu_files if is_valid_4k_mania_chart(p)]
            
            if not valid_osu_files:
                continue # 유효한 차트가 없으면 오디오 처리 없이 다음 폴더로

            # 2. 유효한 차트가 있을 경우에만 '오디오 처리'
            audio_files = glob.glob(os.path.join(song_folder, '*.mp3')) + glob.glob(os.path.join(song_folder, '*.ogg')) + glob.glob(os.path.join(song_folder, '*.wav'))
            if not audio_files: 
                continue
            audio_files.sort(key=os.path.getsize, reverse=True)
            for i in audio_files:
                song_duration = librosa.get_duration(path=i, sr=hparams['sampling_rate'])
                if song_duration > 60:
                    input_tensor = process_audio(i, hparams)
                    break

            # 3. 유효한 차트들 각각에 대해 '레이블 처리' 및 저장
            for osu_path in valid_osu_files:
                labels, od, density = process_mania_beatmap(osu_path, song_duration, hparams)
                
                min_frames = min(len(labels), input_tensor.shape[0])
                
                song_name = os.path.basename(song_folder)
                chart_name = os.path.splitext(os.path.basename(osu_path))[0]
                target_filename = f"{song_name}_{chart_name}.npz"
                target_filepath = os.path.join(target_dir, target_filename)

                np.savez_compressed(
                    target_filepath, x=input_tensor[:min_frames], y=labels[:min_frames],
                    od=np.array([od]), density=density
                )
        except Exception as e:
            print(f"에러: {os.path.basename(song_folder)} 처리 중 오류 발생 - {e}. 건너뜁니다.")
    
    print("데이터셋 구축이 완료되었습니다.")

if __name__ == '__main__':
    SOURCE_DATA_DIR = 'raw_dataset'
    TARGET_DATA_DIR = 'processed_dataset'
    HPARAMS = {
        'sampling_rate': 44100, 'n_fft_list': [1024, 2048, 4096], 'n_mels': 80,
        'n_context': 7, 'patch_size': (5, 16), 'fuzzy_label_length': 3
    }
    '''
    # --- 테스트용 가상 데이터셋 생성 ---
    print("테스트용 가상 데이터셋을 생성합니다...")
    os.makedirs(os.path.join(SOURCE_DATA_DIR, 'test_song_valid'), exist_ok=True)
    os.makedirs(os.path.join(SOURCE_DATA_DIR, 'test_song_invalid'), exist_ok=True)
    # 유효한 4키 차트
    with open(os.path.join(SOURCE_DATA_DIR, 'test_song_valid/4k_chart.osu'), 'w') as f:
        f.write("Mode: 3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n64,192,1000,1,0\n192,192,1500,1,0\n320,192,2000,1,0\n448,192,2500,1,0")
    # 유효하지 않은 5키 차트
    with open(os.path.join(SOURCE_DATA_DIR, 'test_song_invalid/5k_chart.osu'), 'w') as f:
        f.write("Mode: 3\n[Difficulty]\nCircleSize:5\n[HitObjects]\n51,192,1000,1,0")
    y, sr = librosa.load(librosa.ex('trumpet'), duration=5)
    sf.write(os.path.join(SOURCE_DATA_DIR, 'test_song_valid/audio.wav'), y, sr)
    sf.write(os.path.join(SOURCE_DATA_DIR, 'test_song_invalid/audio.wav'), y, sr)
'''
    # --- 메인 파이프라인 실행 ---
    create_processed_dataset(SOURCE_DATA_DIR, TARGET_DATA_DIR, HPARAMS)