import sys
import csv
import speech_recognition as sr
import time
import os
from RET.predict import predict_emotion_ret
from REF.predict import predict_emotion_ref

# Mapeamento das emoções
EMOTION_MAPPING = {
    "felicidade": "positivo",
    "neutro": "neutro",
    "tristeza": "negativo",
    "raiva": "negativo",
    "medo": "negativo",
}

# Mapeamento de emoções para categorias (positiva, neutra, negativa)
category_mapping = {
    "positivo": 1,
    "neutro": 0,
    "negativo": -1
}

# Função para calcular a emoção final ponderando os resultados de texto e fala


def calculate_final_emotion_weight(text_emotion, audio_emotion):
    # Pontos de ponderação para texto e fala (ajuste conforme necessário)
    weight_text = 0.4
    weight_audio = 0.6

    # Converta emoções para categorias
    text_category = category_mapping[text_emotion]
    audio_category = category_mapping[EMOTION_MAPPING[audio_emotion]]

    # Calcule a emoção final ponderada
    final_emotion_score = (weight_text * text_category +
                           weight_audio * audio_category) / (weight_text + weight_audio)

    # Mapeie a emoção final de volta para a categoria
    if final_emotion_score > 0:
        final_emotion = "positivo"
        confidence = final_emotion_score
    elif final_emotion_score == 0:
        final_emotion = "neutro"
        confidence = 1.0 - abs(final_emotion_score)
    else:
        final_emotion = "negativo"
        confidence = abs(final_emotion_score)

    return final_emotion, confidence

# Função para calcular a emoção final sem peso


def calculate_final_emotion_normal(text_emotion, audio_emotion):
    # Converta emoções para categorias
    text_category = category_mapping[text_emotion]
    audio_category = category_mapping[EMOTION_MAPPING[audio_emotion]]

    # Calcule a emoção final
    final_emotion_score = (text_category + audio_category)

    # Mapeie a emoção final de volta para a categoria
    if final_emotion_score > 0:
        final_emotion = "positivo"
        confidence = final_emotion_score
    elif final_emotion_score == 0:
        final_emotion = "neutro"
        confidence = 1.0 - abs(final_emotion_score)
    else:
        final_emotion = "negativo"
        confidence = abs(final_emotion_score)

    return final_emotion, confidence

# Transcrever o áudio para texto captado pelo microfone - speech recognition


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language='pt')
    except:
        print('skipping unknown error')
        return None


def main():
    file_exists = os.path.isfile('resultsTEST.csv')
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Recognized Text', 'Text Emotion', 'Audio Emotion',
                      'Final Emotion (Weighted)', 'Confidence (Weighted)', 'Final Emotion (Normal)', 'Confidence (Normal)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        while True:
            # Gravar áudio
            filename = "input.wav"
            with sr.Microphone() as source:
                print("Gravação ON:")
                recognizer = sr.Recognizer()
                source.pause_threshold = 1
                # Permitir fala contínua
                audio = recognizer.listen(source, phrase_time_limit=None)
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())

                # Transcrever áudio para texto
                text = transcribe_audio_to_text(filename)
                if text:
                    print(f"TEXTO RECONHECIDO: {text}")

                    # Detectando emoção por texto
                    print("Emoção Texto:")
                    response_ret = predict_emotion_ret(text)

                    # Detectando emoção por fala
                    response_ref = predict_emotion_ref(filename)
                    print(f"Emoção Fala: {response_ref}")

                    # Calcular a emoção final ponderada
                    final_emotion_weight, confidence_weight = calculate_final_emotion_weight(
                        response_ret, response_ref)
                    final_emotion, confidence = calculate_final_emotion_normal(
                        response_ret, response_ref)
                    print(
                        f"Emoção Final com Peso: {final_emotion_weight} (Confiança: {confidence_weight * 100:.2f}%)")
                    print(
                        f"Emoção Final: {final_emotion} (Confiança: {confidence * 100:.2f}%)")

                    # Registre os resultados no arquivo CSV
                    writer.writerow({'Recognized Text': text, 'Text Emotion': response_ret, 'Audio Emotion': response_ref, 'Final Emotion (Weighted)': final_emotion_weight,
                                    'Confidence (Weighted)': confidence_weight, 'Final Emotion (Normal)': final_emotion, 'Confidence (Normal)': confidence})

            # if keyboard.is_pressed("enter"):
            #     print("Gravação OFF")
            #     break


if __name__ == "__main__":
    main()
