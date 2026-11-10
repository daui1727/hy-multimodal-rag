"""
② 요약 생성 모듈 (02_요약생성 노트북)

[기능 정의]
추출된 텍스트 · 표 · 이미지마다 "검색용 요약문"을 만든다.
요약문은 벡터 DB에 임베딩되어 검색에 쓰이고, 원본은 답변 생성에 쓰인다.

입력 : - data/processed/extracted.pkl (texts_2k_token, tables)
       - data/figures/*.jpg          (추출 이미지)
출력 : data/processed/summaries.pkl
       {"text_summaries", "table_summaries", "image_summaries", "img_base64_list"}

[처리 흐름]
텍스트·표 → LLM 요약 (LCEL 체인, batch)
이미지   → Base64 인코딩 → VLM 요약 (HumanMessage 에 지시문 + 이미지)

실행 : python src/summarize.py   (OpenAI API 비용 발생)
"""

import base64
import os
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "figures"

MODEL_NAME = "gpt-4o-mini"  # 텍스트·표·이미지 요약 모두 같은 모델 (비전 지원, 저렴)

# 텍스트·표 요약 프롬프트: 질문이 한국어이므로 요약도 한국어로 (같은 언어끼리 검색이 잘 됨)
TEXT_PROMPT = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

# 이미지 요약 프롬프트: VLM 은 영어 지시를 더 안정적으로 따르므로 지시는 영어, 출력만 한국어
IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=True):
    """텍스트와 표를 검색용 요약문으로 만듭니다.

    texts           : 텍스트 청크 리스트 (texts_2k_token)
    tables          : 표 문자열 리스트
    summarize_texts : False 면 텍스트는 요약하지 않고 원본을 그대로 요약 자리에 사용
    반환             : (text_summaries, table_summaries)
    """
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT)
    model = ChatOpenAI(temperature=0, model=MODEL_NAME)  # temperature=0 → 같은 입력엔 같은 요약

    # 요약 체인 = 입력 포장 → 프롬프트 → LLM → 문자열 꺼내기
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 바꿉니다. (API 는 문자열만 주고받기 때문)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt=IMAGE_PROMPT):
    """Base64 이미지 한 장을 VLM 으로 요약합니다.

    하나의 HumanMessage 안에 [지시문 텍스트, 이미지] 를 함께 담아 보낸다.
    """
    chat = ChatOpenAI(model=MODEL_NAME, max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ])
    ])
    return msg.content


def generate_img_summaries(image_dir, prompt=IMAGE_PROMPT):
    """폴더 안 이미지를 이름순으로 Base64 인코딩하고 요약합니다.

    반환 : (img_base64_list, image_summaries)
           두 리스트는 같은 순서여야 다음 단계에서 같은 doc_id 로 짝지을 수 있다.
    """
    img_base64_list = []
    image_summaries = []

    for img_file in sorted(os.listdir(image_dir)):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            base64_image = encode_image(os.path.join(image_dir, img_file))
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


def load_extracted(processed_dir=DEFAULT_PROCESSED_DIR):
    """① 추출 단계가 저장한 extracted.pkl 을 불러옵니다."""
    with open(Path(processed_dir) / "extracted.pkl", "rb") as f:
        return pickle.load(f)


def save_summaries(text_summaries, table_summaries, image_summaries, img_base64_list, processed_dir):
    """요약 결과를 processed_dir/summaries.pkl 로 저장하고 저장 경로를 반환합니다."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "summaries.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list,
            },
            f,
        )
    return out_path


def run_summarization(processed_dir=DEFAULT_PROCESSED_DIR, image_dir=DEFAULT_IMAGE_DIR):
    """extracted.pkl 불러오기 → 텍스트·표 요약 → 이미지 요약 → summaries.pkl 저장."""
    extracted = load_extracted(processed_dir)
    texts_2k_token = extracted["texts_2k_token"]
    tables = extracted["tables"]

    start = time.time()
    text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables)
    print(f"텍스트 요약 {len(text_summaries)}건 / 표 요약 {len(table_summaries)}건 - {time.time() - start:.1f}초")

    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(str(image_dir))
    print(f"이미지 요약 {len(image_summaries)}건 - {time.time() - start:.1f}초")

    out_path = save_summaries(text_summaries, table_summaries, image_summaries, img_base64_list, processed_dir)
    print("저장 완료:", out_path)

    return {
        "text_summaries": text_summaries,
        "table_summaries": table_summaries,
        "image_summaries": image_summaries,
        "img_base64_list": img_base64_list,
    }


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")  # OPENAI_API_KEY 를 환경변수로 불러옴 (키를 코드에 쓰지 않음)
    run_summarization()
