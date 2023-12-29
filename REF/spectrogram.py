import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Dicionário que mapeia as emoções para as pastas correspondentes
emotions_folders = {
    'felicidade': 'REF/audios/felicidade',
    'medo': 'REF/audios/medo',
    'neutro': 'REF/audios/neutro',
    'raiva': 'REF/audios/raiva',
    'tristeza': 'REF/audios/tristeza'
}

# Cria a pasta de destino para os espectrogramas
os.makedirs('REF/espectrogramas', exist_ok=True)

# Percorre cada emoção e pasta correspondente
for emotion, folder in emotions_folders.items():
    # Caminho completo para a pasta da emoção
    folder_path = os.path.join(os.getcwd(), folder)

    # Lista de arquivos de áudio na pasta
    audio_files = os.listdir(folder_path)

    # Cria a pasta de destino para a emoção
    emotion_save_folder = os.path.join('REF/espectrogramas', emotion)
    os.makedirs(emotion_save_folder, exist_ok=True)

    # Percorre cada arquivo de áudio na pasta
    for file in audio_files:
        # Caminho completo para o arquivo de áudio
        audio_path = os.path.join(folder_path, file)

        # Carrega o arquivo de áudio
        y, sr = librosa.load(audio_path)
        yt, _ = librosa.effects.trim(y)
        y = yt

        # Cria o espectrograma mel
        mel_spect = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=100)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        # Exibe e salva o espectrograma
        librosa.display.specshow(
            mel_spect, y_axis='mel', fmax=20000, x_axis='time')
        plt.title('Mel Spectrogram')

        # Caminho para salvar o espectrograma
        save_path = os.path.join(emotion_save_folder, file[:-4] + '.jpeg')

        # Salva o espectrograma como uma imagem JPEG
        plt.savefig(save_path)
        plt.close()
