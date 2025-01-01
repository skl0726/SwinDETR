# SwinDETR

### 디렉토리 구조
SwinDETR
├── args.py                       # 학습/평가에 사용되는 명령행 인자 정의
├── coco                          # COCO 데이터셋 디렉토리
│   ├── annotations               # COCO 데이터셋 주석 파일 (JSON 형식)
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017                 # COCO 학습 이미지 디렉토리
│   └── val2017                   # COCO 평가 이미지 디렉토리
├── csv_analysis.py               # CSV 데이터를 분석하는 스크립트
├── datasets                      # 데이터셋 관련 모듈
│   ├── init.py
│   ├── dataset.py                # 데이터셋 로더 정의
│   └── transforms.py             # 데이터 전처리 및 변환
├── eval_densenet.py              # DenseNet 기반 모델 평가 스크립트
├── eval_swin_t.py                # Swin Transformer 기반 모델 평가 스크립트
├── inference_image.py            # 단일 이미지 추론 스크립트
├── inference_video.py            # 비디오 추론 스크립트
├── inference_video_with_alert.py # 경고 시스템을 포함한 비디오 추론 스크립트
├── models                        # 모델 관련 모듈
│   ├── init.py
│   ├── backbone.py               # 백본 모델 정의
│   ├── criterion.py              # 학습 손실 함수 정의
│   ├── densenet.py               # DenseNet 모델 구현
│   ├── matcher.py                # 예측과 정답 매칭 알고리즘
│   ├── positional_encoding.py    # 위치 인코딩 정의
│   ├── swin_detr.py              # Swin-DETR 모델 구현
│   ├── swin_transformer.py       # Swin Transformer 구현
│   └── transformer.py            # Transformer 구조 정의
├── train_densenet.py             # DenseNet 모델 학습 스크립트
├── train_densenet.sh             # DenseNet 모델 학습 실행 스크립트
├── train_swin_t.py               # Swin Transformer 모델 학습 스크립트
├── train_swin_t.sh               # Swin Transformer 모델 학습 실행 스크립트
└── utils                         # 유틸리티 모듈
├── init.py
├── box_ops.py                # 바운딩 박스 관련 함수
└── misc.py                   # 기타 유틸리티 함수
