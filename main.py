from fastapi import FastAPI, File, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import soundfile as sf
import os

app = FastAPI()

# 모델과 토크나이저 로드
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 업로드된 파일을 저장
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # 음성 파일 로드 및 전처리
    audio_input, sample_rate = sf.read(file_location)

    # 모델에 입력하기 위해 텐서 변환
    input_values = tokenizer(audio_input, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values

    # 모델 추론
    with torch.no_grad():
        logits = model(input_values).logits
    
    # 예측된 ID를 텍스트로 변환
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    # 임시 파일 삭제
    os.remove(file_location)

    return {"transcription": transcription}
