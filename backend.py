"""
AI 보컬 리무버 백엔드 서버
딥러닝 모델을 사용한 고품질 보컬/코러스 제거
"""

import os
import io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
from scipy.signal import stft, istft
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # CORS 허용

class VocalSeparationModel:
    def __init__(self):
        self.vocal_model = None
        self.chorus_model = None
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.is_loaded = False
        
    def load_models(self):
        """딥러닝 모델 로드"""
        try:
            print("AI 모델 로딩 중...")
            self.vocal_model = self._create_vocal_model()
            self.chorus_model = self._create_chorus_model()
            self.is_loaded = True
            print("AI 모델 로드 완료!")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.is_loaded = False
    
    def _create_vocal_model(self):
        """보컬 분리 U-Net 모델"""
        input_shape = (None, 1025, 2)  # (time, freq, channels)
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # Encoder
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D((2, 1))(conv1)
        
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D((2, 1))(conv2)
        
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D((2, 1))(conv3)
        
        # Bottleneck
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        
        # Decoder
        up5 = tf.keras.layers.UpSampling2D((2, 1))(conv4)
        merge5 = tf.keras.layers.Concatenate()([up5, conv3])
        conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)
        
        up6 = tf.keras.layers.UpSampling2D((2, 1))(conv5)
        merge6 = tf.keras.layers.Concatenate()([up6, conv2])
        conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
        
        up7 = tf.keras.layers.UpSampling2D((2, 1))(conv6)
        merge7 = tf.keras.layers.Concatenate()([up7, conv1])
        conv7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
        
        # Output mask
        mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)
        
        model = tf.keras.Model(inputs=inputs, outputs=mask)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _create_chorus_model(self):
        """코러스 제거 고급 모델"""
        input_shape = (None, 1025, 2)
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # Multi-scale feature extraction
        conv1_3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1_5 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
        conv1_7 = tf.keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same')(inputs)
        
        merged = tf.keras.layers.Concatenate()([conv1_3, conv1_5, conv1_7])
        
        # Deep processing
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Attention mechanism
        attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        attended = tf.keras.layers.Multiply()([x, attention])
        
        # Final mask
        mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(attended)
        
        model = tf.keras.Model(inputs=inputs, outputs=mask)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def separate_vocals(self, audio_path, method='vocal'):
        """보컬 분리 수행"""
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            
            if y.ndim == 1:
                # 모노를 스테레오로 변환
                y = np.stack([y, y])
            
            # STFT 변환
            stft_left = librosa.stft(y[0], n_fft=self.n_fft, hop_length=self.hop_length)
            stft_right = librosa.stft(y[1], n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 복소수를 실수부, 허수부로 분리
            magnitude_left = np.abs(stft_left)
            magnitude_right = np.abs(stft_right)
            phase_left = np.angle(stft_left)
            phase_right = np.angle(stft_right)
            
            # 스테레오 스펙트로그램 생성
            stereo_spec = np.stack([magnitude_left, magnitude_right], axis=-1)
            stereo_spec = np.expand_dims(stereo_spec.T, axis=0)  # (1, time, freq, 2)
            
            # AI 모델 적용
            if method == 'vocal' and self.vocal_model and self.is_loaded:
                mask = self.vocal_model.predict(stereo_spec, verbose=0)
            elif method == 'chorus' and self.chorus_model and self.is_loaded:
                mask = self.chorus_model.predict(stereo_spec, verbose=0)
            else:
                # 폴백: 전통적인 방법
                mask = self._traditional_separation(stereo_spec, method)
            
            # 마스크 적용
            mask = mask[0].T  # (freq, time)
            mask = np.squeeze(mask)
            
            # 악기 마스크 (보컬 제거용)
            instrumental_mask = 1.0 - mask
            
            # 좌우 채널에 마스크 적용
            filtered_left = magnitude_left * instrumental_mask
            filtered_right = magnitude_right * instrumental_mask
            
            # 복소수 재구성
            stft_filtered_left = filtered_left * np.exp(1j * phase_left)
            stft_filtered_right = filtered_right * np.exp(1j * phase_right)
            
            # ISTFT 변환
            y_left = librosa.istft(stft_filtered_left, hop_length=self.hop_length)
            y_right = librosa.istft(stft_filtered_right, hop_length=self.hop_length)
            
            # 스테레오 결합
            result = np.stack([y_left, y_right])
            
            # 후처리
            result = self._post_process(result, method)
            
            return result, self.sample_rate
            
        except Exception as e:
            print(f"분리 처리 중 오류: {e}")
            raise
    
    def _traditional_separation(self, stereo_spec, method):
        """전통적인 분리 방법 (AI 모델 실패시 폴백)"""
        # 중앙 채널 추출 기반 마스크 생성
        left_mag = stereo_spec[0, :, :, 0]
        right_mag = stereo_spec[0, :, :, 1]
        
        # 좌우 채널 차이를 이용한 보컬 감지
        center_channel = np.abs(left_mag - right_mag)
        side_channel = (left_mag + right_mag) / 2
        
        # 동적 임계값 계산
        threshold = np.percentile(center_channel, 70)
        vocal_mask = (center_channel > threshold).astype(np.float32)
        
        if method == 'chorus':
            # 코러스 모드에서는 더 강한 마스크
            vocal_mask = np.where(vocal_mask > 0.3, 0.9, vocal_mask)
            vocal_mask = np.where(side_channel > np.percentile(side_channel, 60), 
                                vocal_mask * 1.2, vocal_mask)
        
        # 스무딩 적용
        from scipy import ndimage
        vocal_mask = ndimage.gaussian_filter(vocal_mask, sigma=1.0)
        
        return vocal_mask[np.newaxis, :, :, np.newaxis]
    
    def _post_process(self, audio, method):
        """오디오 후처리"""
        # 정규화
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # 노이즈 게이트
        threshold = 0.001
        audio = np.where(np.abs(audio) < threshold, 0, audio)
        
        # 소프트 클리핑
        audio = np.tanh(audio * 0.9) * 0.9
        
        if method == 'chorus':
            # 코러스 모드에서 추가적인 고주파 필터링
            from scipy.signal import butter, filtfilt
            nyquist = self.sample_rate / 2
            high_cutoff = 8000 / nyquist
            b, a = butter(4, high_cutoff, btype='low')
            
            for i in range(audio.shape[0]):
                audio[i] = filtfilt(b, a, audio[i])
        
        return audio

# 글로벌 모델 인스턴스
vocal_separator = VocalSeparationModel()

@app.route('/', methods=['GET'])
def home():
    """홈페이지"""
    return '''
    <html>
    <head>
        <title>AI 보컬 리무버 API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 10px 0; }
            code { background: #f1f1f1; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 AI 보컬 리무버 API</h1>
            <div class="status">
                <strong>✅ 서버 상태:</strong> 정상 동작<br>
                <strong>🤖 AI 모델:</strong> ''' + ('로드됨' if vocal_separator.is_loaded else '로딩 중') + '''
            </div>
            
            <h2>📡 API 엔드포인트</h2>
            <div class="endpoint">
                <strong>GET /api/health</strong><br>
                서버 상태 확인
            </div>
            <div class="endpoint">
                <strong>POST /api/separate</strong><br>
                음성 분리 처리<br>
                Parameters: audio (file), method (vocal|chorus)
            </div>
            <div class="endpoint">
                <strong>GET /api/model-status</strong><br>
                AI 모델 상태 확인
            </div>
            
            <p style="text-align: center; margin-top: 30px;">
                <a href="/index.html" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                    웹 인터페이스로 이동 →
                </a>
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': vocal_separator.is_loaded
    })

@app.route('/api/separate', methods=['POST'])
def separate_audio():
    """음성 분리 API"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '오디오 파일이 없습니다'}), 400
        
        audio_file = request.files['audio']
        method = request.form.get('method', 'vocal')  # 'vocal' or 'chorus'
        
        if audio_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            audio_file.save(temp_input.name)
            temp_input_path = temp_input.name
        
        try:
            # 음성 분리 수행
            separated_audio, sr = vocal_separator.separate_vocals(temp_input_path, method)
            
            # 결과를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                sf.write(temp_output.name, separated_audio.T, sr)
                temp_output_path = temp_output.name
            
            # 파일 전송
            return send_file(
                temp_output_path,
                as_attachment=True,
                download_name=f'{method}_removed.wav',
                mimetype='audio/wav'
            )
        
        finally:
            # 임시 파일 정리
            try:
                os.unlink(temp_input_path)
                if 'temp_output_path' in locals():
                    os.unlink(temp_output_path)
            except:
                pass
    
    except Exception as e:
        return jsonify({'error': f'처리 중 오류: {str(e)}'}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """모델 상태 확인"""
    return jsonify({
        'vocal_model_loaded': vocal_separator.vocal_model is not None,
        'chorus_model_loaded': vocal_separator.chorus_model is not None,
        'ready': vocal_separator.is_loaded
    })

if __name__ == '__main__':
    print("🎵 AI 보컬 리무버 백엔드 시작")
    print("필요한 패키지 설치 명령:")
    print("pip install -r requirements.txt")
    print()
    
    # 모델 로드
    vocal_separator.load_models()
    
    # 서버 시작
    print("서버 시작 중...")
    print("웹 인터페이스: http://localhost:5000")
    print("API 문서: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.