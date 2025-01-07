# 베이스 이미지 설정
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . /app

# 컨테이너가 노출할 포트
EXPOSE 7777

# 애플리케이션 실행 명령어
CMD ["uvicorn", "ws_user_chatbot:app", "--host", "0.0.0.0", "--port", "7777", "--reload"]