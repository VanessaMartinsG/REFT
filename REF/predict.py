import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import librosa

# Define a função de predição


def predict_emotion_ref(audio_path):

    # Verifique se a GPU está disponível e use-a para inferência, se possível
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define o dicionário de emoções
    EMOTIONS = {
        0: "felicidade",
        1: "medo",
        2: "neutro",
        3: "raiva",
        4: "tristeza"
    }

    # Carregue o modelo treinado
    model = models.resnet34(pretrained=False)
    num_features = model.fc.in_features
    # Ajuste o número de classes
    model.fc = nn.Linear(num_features, len(EMOTIONS))
    model.load_state_dict(torch.load(
        "audio_emotion_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Recebe um áudio
    audio, sampling_rate = librosa.load(audio_path)

    # Transforma o áudio em um espectrograma
    spectrogram = librosa.stft(audio, n_fft=512, hop_length=256)
    spectrogram_magnitude = np.abs(spectrogram)
    spectrogram_magnitude = spectrogram_magnitude.astype(np.float32)

    # Transforma o espectrograma em um tensor PyTorch
    spectrogram_tensor = torch.from_numpy(spectrogram_magnitude)
    spectrogram_tensor = spectrogram_tensor.unsqueeze(0).repeat(1, 3, 1, 1)

    # Move o tensor de entrada para a GPU
    spectrogram_tensor = spectrogram_tensor.to(device)

    # Faz a predição da emoção
    outputs = model(spectrogram_tensor)

    # Retorna a emoção predita
    _, predicted = torch.max(outputs, 1)
    predicted_emotion = EMOTIONS[predicted.item()]

    return predicted_emotion


def print_emotion(audio):
   
    predicted_emotion = predict_emotion_ref(audio)

    print(predicted_emotion)


