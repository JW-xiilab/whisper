import whisper

def main():
    MODEL = 'large-v2'
    # AUDIO = 'harvard.wav'
    AUDIO = 'news_mbc_busan.mp3'
    model = whisper.load_model(MODEL)
    result = model.transcribe_kjw(AUDIO, verbose=True, language='ko')
    print(result['text'])

    # audio = whisper.load_audio(AUDIO)
    # pre_audio = whisper.pad_or_trim(audio)
    # mel = whisper.log_mel_spectrogram(mel)
    # _, probs = model.detect_language(mel)
    # print(f'Detected language: {max(probs, key=probs.get)}')

    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)
    # print(result.text)

if __name__ == "__main__":
    main()