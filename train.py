import os
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import pickle
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 定义类别标签
LABELS = ["bark", "run", "fight", "purr", "eat", "drink"]  # 叫声、跑动声、打架声、呼噜声、吃饭声音、喝水声音

# 录音参数
SAMPLE_RATE = 22050
DURATION = 3  # 录音时长（秒）
FILENAME = "recorded_audio.wav"

def record_audio(duration=3, filename="recorded_audio.wav"):
    """ 录制音频 """
    print("Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("Recording saved as", filename)
    return filename

def extract_features(file_path):
    """ 提取音频特征：MFCC, Chroma, Mel """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    return np.hstack([mfccs_mean, chroma_mean, mel_mean])

def load_training_data(data_dir="pet_sounds"):
    """ 加载训练数据 """
    X, y = [], []
    for label in LABELS:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                feature = extract_features(os.path.join(folder, file))
                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)

def train_model():
    """ 训练分类器 """
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 评估模型
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # 保存模型
    with open("pet_sound_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

def predict_sound(file_path, model_path="pet_sound_classifier.pkl"):
    """ 预测声音类型 """
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    print(f"Predicted Sound: {prediction}")
    return prediction

if __name__ == "__main__":
    # 训练模型（仅需执行一次）
    if not os.path.exists("pet_sound_classifier.pkl"):
        print("Training model...")
        train_model()

    # 录制音频并进行预测
    audio_file = record_audio(DURATION, FILENAME)
    predicted_label = predict_sound(audio_file)
