from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Tuple, Optional
import asyncio
import uuid
import os
import psycopg2
from datetime import datetime
from rag_utils import answer_question  # 유저 메세지와 폰 번호를 입력받아 RAG 답변 생성
from rag_utils import add_to_history  # 프리셋 답변역시 과거기록 데이터로 활용가능하도록 RAG 시스템에 전달
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요 시 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기본 페이지 (루트 엔드포인트)로 접속 시 인사 메시지 제공
# 실질적으로 하는 기능은 없으나 배포 단계에서 코드 동작 체크용으로 사용.
@app.get("/")
async def root():
    return {"message": "Welcome to the ws_chat_app!"}

# PostgreSQL 데이터베이스 설정
DATABASE_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

@app.get("/chat_logs")
async def get_chat_logs(user_no: int):
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, sender, message, timestamp
            FROM chat_logs
            WHERE user_no = %s
            ORDER BY timestamp ASC
        """, (user_no,))
        rows = cursor.fetchall()
        conn.close()

        chat_logs = [
            {"session_id": row[0], "sender": row[1], "message": row[2], "timestamp": row[3].isoformat()}
            for row in rows
        ]

        # UTF-8로 인코딩된 JSON 응답
        return JSONResponse(content={"chat_logs": chat_logs}, media_type="application/json; charset=utf-8")
    except Exception as e:
        print(f"Error fetching chat logs: {e}")
        return {"error": "Failed to fetch chat logs"}

# 데이터 베이스 초기화 함수 print("Database connection successful!")
# 챗봇에서 발생한 로그들을 저장할 테이블 생성
def init_db():
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        
        # 테이블 확인 및 생성
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'chat_logs';")
        table_exists = cursor.fetchone()
        if table_exists:
            print("Table 'chat_logs' already exists.")
        else:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id SERIAL PRIMARY KEY,
                user_no INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW(),
                sender VARCHAR(10) NOT NULL,
                message TEXT NOT NULL
        )
        ''')
            print("Table 'chat_logs' created successfully.")

        conn.commit()
        conn.close()
        print("Database connection successful!")
    except Exception as e:
        print("Error during database initialization:", e)

init_db()  # 서버 실행 시 데이터베이스 초기화

# ChatbotManager 클래스
# 세션 연결 관리 및 로그 기록 메서드 정의
class ChatbotManager:
    def __init__(self):
        self.active_connections: Dict[str, Tuple[WebSocket, str]] = {}  # 세션 ID와 웹소켓 연결 및 전화번호 매핑
        self.inactive_tasks: Dict[str, asyncio.Task] = {}  # 세션별 비활성화 타이머 관리
        self.logs_path = "chatbot_logs"  # 로그 파일 디렉토리

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    # 자리비움 타이머 설정 및 일정 시간 초과시 세션 연결 해제
    async def inactivity_check(self, session_id: str):
        try:
            await asyncio.sleep(300)  # 5분(300초) 대기
            websocket, _ = self.active_connections.get(session_id, (None, None))
            if websocket:
                await websocket.send_json({
                    "sender": "bot",
                    "message": "5분 동안 활동이 없어 연결이 종료됩니다."
                })
                await websocket.close()
            self.disconnect(session_id)  # 세션 해제
        except asyncio.CancelledError:
            # 타이머가 취소된 경우 무시
            pass
    # 연결 설정 -  클라이언트 연결 수락하고 세션 ID 생성
    async def connect(self, websocket: WebSocket, user_no: int) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())  # 고유 세션 ID 생성
        self.active_connections[session_id] = (websocket, user_no)  # user_no 저장

    # 로그 파일 생성
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{self.logs_path}/{session_id}.log", "a") as log_file:
            log_file.write(f"세션 시작 시간 : {start_time}\n유저 번호: {user_no}\n")

    # 비활성화 타이머 시작
        self.inactive_tasks[session_id] = asyncio.create_task(self.inactivity_check(session_id))
        return session_id


    # 연결 해제 - 로그 파일을 DB에 저장하고 실행중인 타이머가 있다면 취소
    def disconnect(self, session_id: str):
        try:
            websocket, _ = self.active_connections.pop(session_id, (None, None))
            log_path = f"{self.logs_path}/{session_id}.log"

            print(f"Disconnecting session: {session_id}")
            print(f"Checking log file: {log_path}")

            if websocket:
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if os.path.exists(log_path):
                    print(f"Log file found: {log_path}")

                    with open(log_path, "r") as log_file:
                        log_lines = log_file.readlines()
                        has_content = any("유저:" in line or "챗봇:" in line for line in log_lines)

                    if not has_content:
                        os.remove(log_path)
                        print(f"Log file deleted: {log_path}")
                    else:
                        with open(log_path, "a") as log_file:
                            log_file.write(f"세션 종료 시간 : {end_time}\n")

                        self.save_log_to_db(session_id, log_path, end_time)
                        print(f"Log saved to DB for session: {session_id}")
                        os.remove(log_path)
                else:
                    print(f"No log file found for session: {session_id}")

            if session_id in self.inactive_tasks:
                self.inactive_tasks[session_id].cancel()
                del self.inactive_tasks[session_id]
        except Exception as e:
            print(f"Error during disconnect: {e}")

    # 유저의 요청에 대한 응답 생성 및 전송
    async def send_response(self, session_id: str, message: str, user_no: int, is_preset: bool = False, preset_response: str = ""):
        websocket, _ = self.active_connections.get(session_id, (None, ""))

        if websocket:
            if is_preset:
                response = preset_response  # 프리셋 응답
                add_to_history(session_id=session_id, question=message, response=response)
            else:
                response = answer_question(message, user_no, session_id)  # RAG 응답 생성

            # 유저 메시지와 챗봇 응답을 데이터베이스에 저장
            self.save_message_to_db(user_no, session_id, "user", message)
            self.save_message_to_db(user_no, session_id, "bot", response)

        # 메시지 전송
            await websocket.send_json({"sender": "user", "message": message})
            await websocket.send_json({"sender": "bot", "message": response})


        

    # 자리비움 타이머 리셋 기능
    async def reset_inactivity_timer(self, session_id: str):
        # 기존 타이머 취소하고 새로운 타이머 시작
        if session_id in self.inactive_tasks:
            self.inactive_tasks[session_id].cancel()
            self.inactive_tasks[session_id] = asyncio.create_task(self.inactivity_check(session_id))

    # 데이터베이스에 로그 저장
    def save_log_to_db(self, session_id: str, log_path: str, end_time: str):
        try:
            with open(log_path, "r") as log_file:
                log_content = log_file.read()

            start_time = log_content.splitlines()[0].split("시간 : ")[-1]
            user_no = log_content.splitlines()[1].split("번호: ")[-1]

            print(f"Preparing to save log to database: session_id={session_id}, start_time={start_time}, end_time={end_time}, user_no={user_no}")

            conn = psycopg2.connect(**DATABASE_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (session_id, start_time, end_time, user_no, log)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, start_time, end_time, user_no, log_content))
            conn.commit()
            conn.close()
            print(f"Log successfully saved to database for session_id={session_id}.")
        except Exception as e:
            print(f"Error saving log to database: {e}")

    def save_message_to_db(self, user_no: int, session_id: str, sender: str, message: str):
        try:
            conn = psycopg2.connect(**DATABASE_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (user_no, session_id, sender, message)
                VALUES (%s, %s, %s, %s)
            """, (user_no, session_id, sender, message))
            conn.commit()
            conn.close()
            print(f"Message saved to DB: {sender}: {message}")
        except Exception as e:
            print(f"Error saving message to database: {e}")

# 세션을 관리해줄 객체 생성
manager = ChatbotManager()



# 웹소켓 엔드포인트: "/ws/chatbot" 경로로 프론트엔드와의 실시간 통신을 처리
# 사용자가 접속하면 새로운 세션 ID를 생성하여 연결을 관리하고, 요청에 따라 프리셋 응답 또는 RAG 응답을 제공
@app.websocket("/ws/chatbot")
async def websocket_endpoint(websocket: WebSocket, user_no: Optional[int] = Query(default=None)):
    """
    WebSocket 연결 처리 및 메시지 기록 저장.
    """
    if user_no is None:
        print("Error: user_no is missing.")
        await websocket.close()
        return

    print(f"WebSocket connected for user_no: {user_no}")
    session_id = await manager.connect(websocket, user_no)  # 세션 ID 생성 및 연결
    try:
        while True:
            data = await websocket.receive_json()
            message_text = data.get("message", "")
            is_preset = data.get("isPreset", False)
            preset_response = data.get("response", "")

            print(f"Received message: {message_text}, session_id: {session_id}, user_no: {user_no}")

            # 메시지 저장 및 응답
            await manager.send_response(session_id, message_text, user_no, is_preset, preset_response)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session_id: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")


# 서버를 실행할 때는 `uvicorn`을 사용:
# uvicorn ws_user_chatbot:app --reload
# uvicorn ws_user_chatbot:app --reload --port 8001
