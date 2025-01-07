# 랭체인 추적
from langchain_teddynote import logging
from fastapi import FastAPI
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os

# 패키지 로드
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever



from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
## 랭체인 추적

# API 키 정보 로드
load_dotenv()
# 프로젝트 이름을 입력합니다.
logging.langsmith("RAG_CHAT_00")


# PDF 파일을 읽어 텍스트를 추출하는 함수
def extract_text_from_pdf(file_path):
  pdf_reader = PdfReader(file_path)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text



## DB 연결

# PostgreSQL 데이터베이스에 연결합니다.
# URI 형식: postgresql://username:password@host:port/database

# URI 생성
db_uri = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
db = SQLDatabase.from_uri(db_uri)

# 단계 1: 문서 로드(Load Documents)
# 문서를 로드하고, 청크로 나누고, 인덱싱합니다.

# PDF 파일 로드. 파일의 경로 입력
file_path = "data/recycle(2018).pdf"
loader = PyPDFLoader(file_path=file_path)


# 단계 2: 문서 분할(Split Documents)
# 페이지 별 문서 로드
docs = loader.load()

# SemanticChunker 설정
semantic_text_splitter = SemanticChunker(
  OpenAIEmbeddings(), add_start_index=True
)

# SemanticChunker를 사용하여 텍스트 스플릿
split_docs = semantic_text_splitter.split_documents(docs)



# 벡터 스토어 경로
vectorstore_path = "vdb/faiss_vectorstore.pkl"


# # 벡터 스토어 로드 또는 생성
# if os.path.exists(vectorstore_path):
#   # 기존 벡터 스토어 로드
#   # vectorstore = FAISS.load_local(vectorstore_path, embedding=OpenAIEmbeddings())
#   embeddings = OpenAIEmbeddings()
#   vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

# else:
## Vector DB 구축

# 단계 3, 4: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_docs, embedding=OpenAIEmbeddings())


# 벡터 스토어 저장
vectorstore.save_local(vectorstore_path)



# 단계 5: 리트리버 생성(Create Retriever)
# 사용자의 질문(query) 에 부합하는 문서를 검색합니다.

# 유사도 높은 K 개의 문서를 검색합니다.
k = 3

# (Sparse) bm25 retriever and (Dense) faiss retriever 를 초기화 합니다.
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = k

faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
  retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)


# 단계 6: 프롬프트 생성
from langchain_core.prompts import PromptTemplate

# 개인정보 필터링용 프롬프트 설정
personally_filter_prompt = PromptTemplate.from_template(
  """
  Given a user question, determine if the question requests personally identifiable information (PII),
  such as names, addresses, phone numbers, email addresses, or other sensitive information.
  
  If the question is asking for information about "all users" or "specific other users", respond with "RESTRICTED".
  If the question is asking about the user's own information (e.g., "my information", "details about me"),
  respond with "ALLOW".

  Question: {question}
  """
)

# 질문 필터링용 프롬프트 설정
filter_prompt = PromptTemplate.from_template(
  """
  Given a user question, determine if it requires a database SQL query or if it can be answered 
  using only the provided document context (vector store). If it needs database access, respond 
  with 'DB_REQUIRED'. Otherwise, respond with 'VECTOR_ONLY'.
  
  Question: {question}
  """
)

# SQL 쿼리 체인을 위한 프롬프트 설정
sql_prompt = PromptTemplate.from_template(
  """
  Based on the user question, generate a syntactically correct {dialect} SQL query by inserting the received user_pn value in place of {user_pn}. Ensure the query only allows access to the user's own information, prohibiting retrieval of data about other users. Include only the executable SQL query in the output without additional formatting or labels.

  SQL Query Format:
  SELECT *
  FROM "User" AS u
  JOIN "Phone_Model" AS p ON u.model_idx = p.model_idx
  WHERE u.user_pn = {user_pn} LIMIT {top_k};

  Only use the following tables:
  {table_info}

  Here is the description of the columns in the tables:
  `insurance_expiration_date`: means '내 보험 만료일' or '내 보험 만기일'
  `insurance_start_date`: means '내 보험 가입일' or '내 보험 가입 날짜'

  Question: {input}
  """
).partial(dialect="postgresql")

answer_prompt = PromptTemplate.from_template(
  """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

  Question: {question}
  SQL Query: {query}
  SQL Result: {result}
  Answer: """
)

# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context and consider previous answers when crafting a response to ensure consistency. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Previous Chat History and Answers:
{chat_history}

#Current Question: 
{question} 

#Context: 
{context} 

#Answer considering previous responses:"""
)


# 단계 7: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {
        "context": itemgetter("question") | ensemble_retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# 세션 기록을 저장할 딕셔너리
# 세션 기록을 저장할 딕셔너리
store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 새로운 질문과 답변을 store에 추가하는 함수
def add_to_history(session_id, question, response):
    history = get_session_history(session_id)
    # 질문과 답변을 대화 기록에 추가
    history.add_user_message(question)
    history.add_ai_message(response)


# DB 검색을 위한 체인 생성 함수
def create_db_chain(question, user_pn, session_id):
  # 도구
  execute_query = QuerySQLDataBaseTool(db=db)

  # SQL 쿼리 생성 체인
  write_query = create_sql_query_chain(llm, db, sql_prompt)

  answer = answer_prompt | llm | StrOutputParser()

  # 생성한 쿼리를 실행하고 결과를 출력하기 위한 체인을 생성합니다.
  db_chain = (
    RunnablePassthrough.assign(query=write_query).assign(
      result=itemgetter("query") | execute_query
    )
    | answer
  )

  # RunnableWithMessageHistory로 대화 기록을 포함한 체인을 생성
  db_chain_with_history = RunnableWithMessageHistory(
      runnable=db_chain,
      get_session_history=get_session_history,
      input_messages_key="question",
      history_messages_key="chat_history",
  )

  # 체인을 실행하여 결과를 반환
  response = db_chain_with_history.invoke(
      {"question": question, "user_pn": user_pn},
      config={"configurable": {"session_id": session_id}},
  )

  return response

# 벡터 스토어 검색을 위한 체인 생성 함수
def create_vector_chain(question, session_id):
  # 대화를 기록하는 RAG 체인 생성
  rag_with_history = RunnableWithMessageHistory(
      chain,
      get_session_history,  # 세션 기록을 가져오는 함수
      input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
      history_messages_key="chat_history",  # 기록 메시지의 키
  )

  response = rag_with_history.invoke(
    # 질문 입력
    {"question": question},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": session_id}},
  )

  return response


# 메인 함수: 질문을 필터링하고 적합한 체인을 실행
def answer_question(question, user_pn, session_id):
  # 질문 필터링
  # 질문이 개인정보와 관련된 경우 필터링
  # personally_filter_response = llm(personally_filter_prompt.format({"question": question}))
  personally_chain_filter_response = (
    {"question": RunnablePassthrough()}
    | personally_filter_prompt
    | llm
    | StrOutputParser()
  )

  personally_filter_response = personally_chain_filter_response.invoke({"question": question})

  if personally_filter_response == "RESTRICTED":
    return "개인정보 보호를위해 답변이 불가능한 질문입니다."
  
  chain_filter_response = (
    {"question": RunnablePassthrough()}
    | filter_prompt
    | llm
    | StrOutputParser()
  )

  filter_response = chain_filter_response.invoke({"question": question})
  # print('filter_response', filter_response)
  
  if "DB_REQUIRED" in filter_response:  # 필터 결과 확인 방식 수정
    # DB 검색 체인 실행
    print("DB 검색 체인 실행을 통한 답변")
    if user_pn == "":
      return "개인정보 확인을 위해 로그인이 필요한 질문입니다."
    response = create_db_chain(question, user_pn, session_id)
  else:
    # 벡터 스토어 검색 체인 실행
    print("벡터 스토어 검색 체인 실행을 통한 답변")
    response = create_vector_chain(question, session_id)
  
  return response