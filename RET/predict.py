from .classificacao.modelo import get_model

def predict_emotion_ret(input_text):
    
    model = get_model()

    sentiment, confidence, probabilities = model.predict(input_text)

    print(f"Sentimento: {sentiment}")
    print(f"Confiança: {confidence}")
    print("Probabilidades:")
    for emotion, probability in probabilities.items():
        print(f"{emotion}: {probability}")
    
    return sentiment

# if __name__ == "__main__":
    
#     sentence = input()
#     predicted_emotion = predict_emotion(sentence)