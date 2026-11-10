"""
① 데이터 추출 모듈 (01_데이터추출 노트북)

[기능 정의]
PDF 한 개를 Unstructured `partition_pdf(hi_res)`로 분석해
텍스트 블록 · 표 · 이미지로 나누고, 다음 단계(요약)에서 쓸 수 있게 저장한다.

입력 : PDF 파일 (기본값 data/economy_FFF.pdf)
출력 : - data/figures/*.jpg           추출된 그림(figure-*) · 표(table-*) 이미지 파일
       - data/processed/extracted.pkl {"texts", "tables", "texts_2k_token"}

[처리 흐름]
partition_pdf (hi_res) → 요소 분류 (텍스트 / 표) → 텍스트 토큰 분할 → pickle 저장

실행 : python src/extract.py
"""

import os
import pickle
import time
from collections import Counter
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# 프로젝트 루트 = 이 파일(src/extract.py)의 한 단계 위 폴더
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PDF = DATA_DIR / "economy_FFF.pdf"
DEFAULT_IMAGE_DIR = DATA_DIR / "figures"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed"


def extract_pdf_elements(pdf_path, image_dir):
    """PDF 파일에서 이미지, 표, 텍스트 블록을 추출합니다.

    pdf_path  : 분석할 PDF 경로
    image_dir : 잘라낸 이미지를 저장할 폴더 (요약 단계가 읽는 폴더와 같아야 함)
    반환       : Unstructured Element 리스트
    """
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                          # 레이아웃 분석 모델로 표·그림 영역까지 탐지
        infer_table_structure=True,                 # 표의 행·열 구조를 추론
        # 그림과 표 영역을 이미지 파일로 잘라냄 (figure-*.jpg, table-*.jpg)
        # 표를 이미지로도 저장하는 이유: 텍스트로 뽑으면 2단 머리글("2025" 아래 "연간 · 4/4")이
        # 한 줄로 뭉개져 연도 열을 잘못 읽는다. 이미지는 머리글 구조가 그대로 남는다.
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=str(image_dir),  # 이미지 저장 경로를 명시
        chunking_strategy="by_title",               # 제목 단위로 본문을 묶음
        max_characters=4000,                        # 한 블록의 최대 글자 수
        new_after_n_chars=3800,                     # 이 길이를 넘으면 새 블록 시작
        combine_text_under_n_chars=2000,            # 짧은 조각은 합쳐 요약 호출 수를 줄임
    )


def categorize_elements(raw_pdf_elements):
    """추출된 요소를 텍스트와 표로 분류합니다.

    - CompositeElement (by_title 로 묶인 본문) → texts
    - Table                                    → tables
    반환 : (texts, tables) 문자열 리스트 두 개
    """
    texts = []
    tables = []
    for element in raw_pdf_elements:
        element_type = type(element).__name__
        if element_type == "Table":
            tables.append(str(element))
        elif element_type == "CompositeElement":
            texts.append(str(element))
    # ⚠️ return 은 for 문 밖에 있어야 모든 요소를 분류한다 (01 노트북에서 겪은 버그)
    return texts, tables


def count_element_types(raw_pdf_elements):
    """요소 종류별 개수를 셉니다. (예: {"CompositeElement": 12, "Table": 5})"""
    return Counter(type(el).__name__ for el in raw_pdf_elements)


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트 블록을 이어 붙인 뒤 토큰 수 기준으로 다시 자릅니다.

    chunk_size    : 한 청크의 최대 토큰 수 (모델 입력 한도 관리)
    chunk_overlap : 청크끼리 겹치는 토큰 수 (경계에서 문장이 잘려 의미가 사라지는 것 방지)
    반환           : texts_2k_token (분할된 텍스트 리스트)
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    joined_texts = " ".join(texts)
    return text_splitter.split_text(joined_texts)


def save_extracted(texts, tables, texts_2k_token, output_dir):
    """추출 결과를 output_dir/extracted.pkl 로 저장하고 저장 경로를 반환합니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "extracted.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token},
            f,
        )
    return out_path


def run_extraction(pdf_path=DEFAULT_PDF, output_dir=DEFAULT_OUTPUT_DIR, image_dir=DEFAULT_IMAGE_DIR):
    """추출 → 분류 → 토큰 분할 → 저장을 한 번에 실행합니다.

    반환 : {"texts", "tables", "texts_2k_token"} 딕셔너리
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_path}")

    start = time.time()
    raw_pdf_elements = extract_pdf_elements(pdf_path, image_dir)
    print(f"추출 완료 - {time.time() - start:.1f}초 소요, 전체 요소 {len(raw_pdf_elements)}개")

    for element_type, count in count_element_types(raw_pdf_elements).most_common():
        print(f"  {element_type:22s}: {count}개")

    texts, tables = categorize_elements(raw_pdf_elements)
    texts_2k_token = split_texts(texts)
    print(f"텍스트 블록 {len(texts)}개 → 토큰 분할 후 {len(texts_2k_token)}개 / 표 {len(tables)}개")

    image_count = len([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"추출 이미지 {image_count}장 → {image_dir}")

    out_path = save_extracted(texts, tables, texts_2k_token, output_dir)
    print("저장 완료:", out_path)

    return {"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token}


if __name__ == "__main__":
    run_extraction()
