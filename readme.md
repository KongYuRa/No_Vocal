# AI 보컬 리무버

딥러닝 기반 고품질 보컬 제거 웹 애플리케이션

## 기능

- **보컬 제거**: 메인 보컬만 제거하고 악기 소리 유지
- **코러스 제거**: 보컬 + 백보컬 + 하모니까지 모두 제거
- **AI 기반**: U-Net과 CNN을 활용한 고품질 처리
- **웹 기반**: 설치 없이 브라우저에서 바로 사용

## 바로 사용하기

### 온라인 버전 (추천)
👉 **[https://your-app.vercel.app](https://your-app.vercel.app)** 접속 후 바로 사용!

### 로컬 실행
1. 저장소 클론
```bash
git clone https://github.com/yourusername/ai-vocal-remover.git
cd ai-vocal-remover
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 서버 실행
```bash
python vocal_remover_backend.py
```

4. 브라우저에서 `http://localhost:5000` 접속

## Docker로 실행

```bash
docker build -t ai-vocal-remover .
docker run -p 5000:5000 ai-vocal-remover
```

## 프로젝트 구조

```
ai-vocal-remover/
├── vocal_remover_backend.py  # Python Flask 백엔드
├── index.html               # HTML 프론트엔드
├── requirements.txt         # Python 의존성
├── Dockerfile              # Docker 설정
├── vercel.json            # Vercel 배포 설정
├── .github/workflows/     # 자동 배포 설정
└── README.md             # 이 파일
```

## 기술 스택

**백엔드:**
- Python + Flask
- TensorFlow (딥러닝)
- LibROSA (오디오 처리)
- NumPy, SciPy

**프론트엔드:**
- HTML5 + CSS3 + JavaScript
- Web Audio API
- Responsive Design

**배포:**
- Vercel (서버리스)
- Docker 지원
- GitHub Actions

## 사용법

1. 오디오 파일 업로드 (MP3, WAV, M4A)
2. 처리 방식 선택 (보컬 제거 / 코러스 제거)  
3. AI 처리 시작
4. 결과 미리 듣기 및 다운로드

## 지원 형식

**입력:** MP3, WAV, M4A (최대 50MB)
**출력:** WAV (고품질)

## 성능

- **처리 시간:** 3분 음악 기준 30초~2분
- **품질:** 전문 소프트웨어 수준
- **용량:** 50MB까지 지원
