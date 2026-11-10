"""
③ 검색기 + 최종 체인 모듈 (03_임베딩_최종체인 노트북)

[기능 정의]
"요약으로 검색하고, 원본으로 답한다."
요약문은 Chroma(벡터 DB)에, 원본(텍스트·표·Base64 이미지)은 InMemoryStore 에
같은 doc_id 로 저장해 MultiVectorRetriever 로 연결하고,
검색된 원본을 멀티모달 LLM 에 넘겨 답변을 만든다.

입력 : data/processed/extracted.pkl, data/processed/summaries.pkl, 사용자 질문
출력 : 답변 문자열 (비교용 텍스트 전용 RAG 답변도 함께 생성 가능)

[처리 흐름]
질문 → 요약 검색 (Chroma, k=4) → doc_id 로 원본 꺼내기 (InMemoryStore)
     → 이미지 / 텍스트 분리 → 한 메시지로 조립 → gpt-4o-mini → 답변

실행 : python src/retriever.py "질문"   (OpenAI API 비용 발생)
"""

import pickle
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
ID_KEY = "doc_id"   # 요약과 원본을 잇는 번호표 (도서관의 청구기호)
TOP_K = 4           # 검색할 요약 개수 (MultiVectorRetriever 기본값과 같음)

# Base64 판별 정규식: 영문·숫자·+·/ 만 100자 이상 이어지고 끝에 = 이 0~2개 붙은 문자열
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")

# 두 체인(멀티모달 / 텍스트 전용)이 똑같이 쓰는 지시문 → Before/After 비교 조건을 맞춤
INSTRUCTION = (
    "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
    "자료에 없는 내용은 모른다고 답하세요.\n\n"
)


def load_processed(processed_dir=DEFAULT_PROCESSED_DIR):
    """추출·요약 결과(pkl 두 개)를 불러와 하나의 딕셔너리로 합칩니다.

    반환 키 : texts_2k_token, tables, text_summaries, table_summaries,
              image_summaries, img_base64_list
    """
    processed_dir = Path(processed_dir)
    with open(processed_dir / "extracted.pkl", "rb") as f:
        extracted = pickle.load(f)
    with open(processed_dir / "summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    return {
        "texts_2k_token": extracted["texts_2k_token"],
        "tables": extracted["tables"],
        **summaries,
    }


def add_documents(retriever, doc_summaries, doc_contents):
    """요약은 벡터 DB 에, 원본은 docstore 에 '같은 id' 로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=summary, metadata={ID_KEY: doc_ids[i]})
        for i, summary in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)               # 검색용 → 색인 카드 서랍
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))       # 전달용 → 서가


def create_multi_vector_retriever(
    text_summaries, texts,
    table_summaries, tables,
    image_summaries, images,
    k=TOP_K,
    collection_name="multimodal_rag",
):
    """요약(Chroma)과 원본(InMemoryStore)을 doc_id 로 연결한 검색기를 만듭니다.

    k : 질문과 비슷한 요약을 몇 개 가져올지
    ⚠️ 같은 프로세스에서 같은 collection_name 으로 두 번 만들면 요약이 중복 저장된다.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )

    # 데이터가 있는 종류만 저장 (요약과 원본은 같은 순서로 짝지어져 있어야 함)
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트(표 포함)로 나눕니다."""
    b64_images = []
    texts = []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if BASE64_PATTERN.match(content):
            b64_images.append(content)
        else:
            texts.append(content)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM 이 읽을 하나의 메시지로 조립합니다.

    1) 이미지를 먼저 넣고 (모델이 그림을 먼저 보게)
    2) 지시문 + 질문 + 참고 텍스트를 넣는다.
    """
    messages = []

    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},   # 키 이름은 반드시 "image_url"
        })

    # ⚠️ 아래는 for 문 밖: 이미지가 0장이어도, 여러 장이어도 텍스트는 한 번만 넣는다
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": (
            INSTRUCTION
            + f"질문: {data_dict['question']}\n\n"
            + f"참고 텍스트:\n{formatted_texts}"
        ),
    })
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever):
    """최종 멀티모달 RAG 체인 (After).

    {검색 + 질문} → img_prompt_func → gpt-4o-mini → StrOutputParser
    """
    model = ChatOpenAI(temperature=0, model=ANSWER_MODEL, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def format_docs(docs):
    """Document 리스트의 본문만 꺼내 줄바꿈으로 이어 붙입니다."""
    return "\n".join(doc.page_content for doc in docs)


def build_text_only_chain(texts, k=TOP_K, collection_name="text_only_rag"):
    """비교용 텍스트 전용 RAG 체인 (Before).

    표·이미지 없이 텍스트 원본만 그대로 임베딩한다.
    임베딩 모델 · 검색 개수 · 답변 모델 · 지시문은 멀티모달 체인과 똑같이 맞춘다.
    """
    text_vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    text_vectorstore.add_texts(texts)
    text_retriever = text_vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        INSTRUCTION + "질문: {question}\n\n참고 텍스트:\n{context}"
    )
    model = ChatOpenAI(temperature=0, model=ANSWER_MODEL, max_tokens=1024)
    return (
        {
            "context": text_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def build_from_processed(processed_dir=DEFAULT_PROCESSED_DIR, k=TOP_K):
    """저장된 pkl 로 검색기와 두 체인을 한 번에 만듭니다.

    반환 : (retriever, multimodal_chain, text_only_chain)
    """
    data = load_processed(processed_dir)
    retriever = create_multi_vector_retriever(
        data["text_summaries"], data["texts_2k_token"],
        data["table_summaries"], data["tables"],
        data["image_summaries"], data["img_base64_list"],
        k=k,
    )
    multimodal_chain = build_multimodal_chain(retriever)
    text_only_chain = build_text_only_chain(data["texts_2k_token"], k=k)
    return retriever, multimodal_chain, text_only_chain


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")  # OPENAI_API_KEY 를 환경변수로 불러옴

    query = sys.argv[1] if len(sys.argv) > 1 else "24년말 대비 26년 1분기의 누적 평가이익을 알려줘"
    retriever, chain, text_only_chain = build_from_processed()

    print("=" * 60)
    print("질문:", query)
    print("=" * 60)
    print("\n[BEFORE] 텍스트 전용 RAG")
    print(text_only_chain.invoke(query))
    print("\n[AFTER] 멀티모달 RAG")
    print(chain.invoke(query))
