import requests

# FastAPI 서버 URL
url = "http://localhost:8000/transcribe/"

# 전송할 음성 파일 (로컬 경로에 있는 파일)
file_path = 'synthesis.wav'

# 파일을 읽어서 전송
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 서버로부터의 응답 출력
if response.status_code == 200:
    print("Transcription:", response.json().get("transcription"))
else:
    print("Failed to transcribe audio. Status code:", response.status_code)
