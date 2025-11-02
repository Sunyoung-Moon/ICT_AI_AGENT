# app.py
import streamlit as st
import requests
import json
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# -----------------------------------------------------------------
# [셀 3: (다중 검색) '도구' 정의] - 수정된 최종 버전
# -----------------------------------------------------------------
def search_precedents_by_keywords(keywords: list, max_per_keyword: int = 3) -> list:
    """
    키워드 리스트를 받아, 각 키워드별로 API를 호출하여
    '판례일련번호'의 '중복 없는' 리스트를 반환.
    '판례명(1)'과 '본문(2)' 검색을 모두 수행.
    """
    print(f"Tool 1: search_precedents_by_keywords 호출 (키워드: {keywords})")
    
    # [Streamlit 수정 1] st.secrets에서 API 키 가져오기
    api_key = st.secrets.get("LAW_API_KEY")
    if not api_key:
        print("Error: LAW_API_KEY가 secrets에 없습니다.")
        st.error("LAW_API_KEY가 설정되지 않았습니다.")
        return []

    base_url = "http://www.law.go.kr/DRF/lawSearch.do"
    unique_ids = set()

    for keyword in keywords:
        for search_type in ["1", "2"]:
            search_type_name = "판례명" if search_type == "1" else "본문"
            print(f"  -> 키워드 '{keyword}' ({search_type_name} 검색) 시도...")
            
            params = {
                "OC": api_key, "target": "prec", "type": "JSON",
                "query": keyword, "search": search_type,
                "sort": "ddes", "display": max_per_keyword
            }
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                precedents = data.get('PrecSearch', {}).get('prec', [])
                
                if precedents:
                    for prec in precedents:
                        pid = prec.get('판례일련번호')
                        if pid:
                            unique_ids.add(pid)
            except Exception as e:
                print(f"  -> 키워드 '{keyword}' ({search_type_name} 검색) 중 오류: {e}")
                continue
            
    if not unique_ids:
        print("  -> '판례 목록' 최종 검색 결과 없음")
        return []

    final_id_list = list(unique_ids)
    print(f"  -> '판례 목록' 총 {len(final_id_list)}건의 고유 ID 추출 성공.")
    return final_id_list

def get_precedent_detail(precedent_id: str) -> dict:
    print(f"Tool 2: get_precedent_detail 호출 (ID: {precedent_id})")
    
    # [Streamlit 수정 2] st.secrets에서 API 키 가져오기
    api_key = st.secrets.get("LAW_API_KEY")
    if not api_key: return {}

    base_url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": api_key, "target": "prec", "ID": precedent_id, "type": "JSON"}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        detail_data = data.get('PrecService', {})
        if not detail_data:
            print(f"  -> '판례 본문' (ID: {precedent_id}) 검색 결과 없음")
            return {}
        print(f"  -> '판례 본문' 검색 성공: {detail_data.get('사건명')}")
        return detail_data
    except Exception as e:
        print(f"Error in get_precedent_detail: {e}")
        return {}

# -----------------------------------------------------------------
# [셀 4: (종합 추론) RAG 파이프라인 정의]
# -----------------------------------------------------------------

# [Streamlit 수정 3] st.secrets에서 OpenAI 키 가져오기
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# 1. 쟁점 도출용 LLM 체인 (가장 성능이 좋았던 '예시 7개' 버전)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
system_prompt_text = """
당신은 법률 전문가입니다. 사용자의 일상적인 질문을 법률 API의 '본문' 및 '판례명' 검색에 최적화된 '검색용 법률 키워드' 3개를 생성해주세요.
키워드는 가장 구체적인 용어에서 가장 일반적인 용어 순서로 생성합니다.
키워드는 법률적 의미를 명확하게 담아야 합니다. (예: '수리의무' 대신 '임대인 수선의무')
출력은 오직 '키워드1, 키워드2, 키워드3' 형식이어야 하며, 어떠한 접두사나 따옴표도 포함하지 마세요.
[예시 1]
질문: "알바 월급을 못 받았어요."
출력: 임금체불, 근로기준법위반, 임금
[예시 2]
질문: "중고거래 벽돌 배송"
출력: 중고거래 사기, 채무불이행, 손해배상
[예시 3]
질문: "사진 도용"
출력: 저작권 침해, 손해배상(지), 초상권
[예시 4]
질문: "윗집이 너무 시끄러워요"
출력: 층간소음 손해배상, 인격권 침해, 위자료
[예시 5]
질문: "길 가다가 옆 건물에서 떨어진 간판에 맞아서 다쳤어요."
출력: 공작물책임, 손해배상(기), 안전의무위반
[예시 6]
질문: "월세집 보일러가 고장났는데 집주인이 수리를 안 해줘요."
출력: 임대인 수선의무, 목적물 수리의무, 임대차계약
[예시 7]
질문: "아버지가 돌아가셨는데 빚이 더 많아요."
출력: 상속포기, 한정승인, 상속채무
"""
issue_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("human", "질문: {question}\n출력:"),
])
extract_issue_chain = issue_extraction_prompt | llm | StrOutputParser() | (lambda x: [k.strip() for k in x.split(',') if k.strip()])

# 2. RAG 컴포넌트
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# 3. 최종 '종합 추론'용 LLM 체인 (🌟 버그 수정본 🌟)
final_reasoning_prompt = ChatPromptTemplate.from_template("""
당신은 매우 유능한 대한민국 변호사입니다.
다음은 사용자의 질문과 관련하여 API로 검색된 '여러 개의 판례 요약'입니다.
이 판례들을 종합적으로 분석하여, 사용자에게 법률 조언을 제공해야 합니다.

[사용자 질문]:
{question}

[참고 판례 목록 (최대 3개)]:
{context}

[법률 자문 (아래 양식 준수)]:
1.  **핵심 쟁점:** (사용자의 질문을 바탕으로 핵심 법률 쟁점을 1문장으로 요약)

2.  **관련 판례 분석:** (검색된 [참고 판례 목록]이 이 쟁점과 어떻게 관련되는지 분석합니다. 
    
    💡 **[중요 지시]** 판례를 언급할 때는 "[판례 1: {{사건명}} ({{사건번호}})]" 형식에서 **반드시 '사건번호'(예: "2021도3451")를 함께 인용**하세요.
    
    [예시]
    * "대법원 2021도3451 판결(사건명: 강제추행)에서는..."
    * "참고 판례(사건번호: 2017다12345)에 따르면...")

3.  **종합 조언 및 결론:** (위 분석을 바탕으로, 사용자에게 "어떻게 하는 것이 유리하다" 또는 "어떤 점을 주장할 수 있다"는 식의 구체적인 조언을 2~3문장으로 제공)
""")
reasoning_chain = final_reasoning_prompt | llm | StrOutputParser()

# 4. Document 변환 함수
def create_documents_and_format(details: list) -> list:
    """
    [Streamlit 수정]
    판례 본문 리스트(details)를 받아 벡터화를 위한 'Document' 리스트만 반환
    """
    documents = []
    
    for i, detail in enumerate(details):
        if not detail: continue
            
        # 벡터화(유사도 검색)에 사용할 내용 (판시사항 + 판결요지)
        content_to_embed = (
            f"판시사항: {detail.get('판시사항', '')}\n\n"
            f"판결요지: {detail.get('판결요지', '')}"
        )
        
        # 💡 [수정 완료]
        # 에러가 발생한 ... 부분을 전체 코드로 복원했습니다.
        # 이 metadata는 나중에 5단계(유사도 검색)에서 청크(chunk)와 함께 사용됩니다.
        metadata = {
            "source_id": detail.get('판례정보일련번호', 'N/A'),
            "사건명": detail.get('사건명', 'N/A'),
            "사건번호": detail.get('사건번호', 'N/A'),
            "선고일자": detail.get('선고일자', 'N/A'),
            "법원명": detail.get('법원명', 'N/A'),
            # [참고] '판례상세링크'는 본문 API(detail)에 원래 없습니다.
            # 따라서 이 링크는 항상 '#'으로 처리되며, 이는 정상입니다.
            "상세링크": f"http://www.law.go.kr{detail.get('판례상세링크', '')}" if detail.get('판례상세링크') else "#"
        }
        
        # 내용이 너무 짧으면 (50자 미만) 유효하지 않은 판례로 간주
        if len(content_to_embed) < 50: continue
            
        # 1. 벡터화(유사도 검색)를 위한 Document 객체 생성
        documents.append(Document(page_content=content_to_embed, metadata=metadata))
        
    print(f"  -> 총 {len(details)}개의 본문 중 {len(documents)}개의 유효한 Document 생성 완료.")
    return documents # 🌟 Document 리스트만 반환


# -----------------------------------------------------------------
# [셀 5: (종합 추론) 전체 파이프라인 실행]
# -----------------------------------------------------------------

def run_legal_pipeline_reasoning(user_question: str) -> str:
    """
    [Streamlit 수정 4] print() 대신 최종 답변 문자열을 return 하도록 수정
    """
    
    print(f"--- [질문: {user_question}] ---")
    
    # 1단계: 쟁점 도출
    issue_keywords = extract_issue_chain.invoke({"question": user_question})
    print(f"🤖 AI가 도출한 쟁점 리스트: {issue_keywords}")
    
    # 2단계: 판례 목록 검색
    precedent_ids = search_precedents_by_keywords(issue_keywords, max_per_keyword=3)
    if not precedent_ids:
        return "[최종 답변]\n- 이 쟁점들로 관련 판례를 찾을 수 없습니다."

    # 3단계: 각 판례 본문 조회
    precedent_details = []
    for pid in precedent_ids:
        detail = get_precedent_detail(pid)
        if detail:
            precedent_details.append(detail)
            
    if not precedent_details:
        return "[최종 답변]\n- 판례 목록은 찾았으나, 상세 내용을 불러오는 데 실패했습니다."

    # 4단계: 본문 -> Document 변환
    documents = create_documents_and_format(precedent_details)
    if not documents:
        return "[최종 답변]\n- 유효한 판례 내용을 찾지 못했습니다."
    
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        print("  -> 본문 내용이 너무 짧아 청크 생성 실패.")
        return "[최종 답변]\n- 검색된 판례의 내용이 너무 짧아 분석할 수 없습니다."
        
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    # 5단계: 유사도 검색 (Re-ranking)
    print(f"  -> 벡터 유사도 검색 시작 (질문: {user_question})")
    try:
        similar_chunks = vectorstore.similarity_search(user_question, k=3)
        if not similar_chunks:
            return "[최종 답변]\n- 질문과 유사한 판례를 찾지 못했습니다."

        context_for_reasoning = ""
        seen_ids = set()
        for i, chunk in enumerate(similar_chunks):
            metadata = chunk.metadata
            source_id = metadata.get('source_id')
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            
            print(f"  -> 가장 유사한 판례 TOP {i+1} 찾음: {metadata.get('사건명')}")
            context_for_reasoning += (
                f"[판례 {i+1}: {metadata.get('사건명')} ({metadata.get('사건번호')})]\n"
                f"- 선고일자/법원: {metadata.get('선고일자')} / {metadata.get('법원명')}\n"
                f"- 판시/요지: {chunk.page_content}\n"
                f"- [판례 원문 보기]({metadata.get('상세링크')})\n\n"
            )

    except Exception as e:
        print(f"  -> 벡터 유사도 검색 실패: {e}")
        return f"[최종 답변]\n- 유사 판례 검색 중 오류가 발생했습니다: {e}"

    if not context_for_reasoning:
        return "[최종 답변]\n- 질문과 유사한 판례를 찾지 못했습니다. (검색 결과가 비어있음)"

    # 6단계: 최종 '종합 추론' (LLM)
    final_answer = reasoning_chain.invoke({
        "question": user_question,
        "context": context_for_reasoning
    })

    # 7. 최종 답변 문자열 구성
    final_response_str = (
        f"**🤖 AI가 분석한 법률 쟁점은 {issue_keywords}입니다.**\n\n"
        "이 쟁점들을 바탕으로 검색된 판례 중, 사용자님의 질문과 **가장 유사한 판례들**을 종합하여 조언해 드립니다.\n\n"
        f"{final_answer}"
    )
    
    return final_response_str

# -----------------------------------------------------------------
# [Streamlit UI 부분]
# -----------------------------------------------------------------

st.title("⚖️ AI 법률 자문 챗봇")
st.write("궁금한 법률 문제를 질문해주세요. AI가 관련 판례를 검색하여 답변해 드립니다.")

# (선택) 이전 질문 기록을 세션 상태에 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력
if user_question := st.chat_input("월세집 보일러가 고장났는데 수리를 안 해줘요..."):
    # 1. 사용자 질문 표시
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. AI 답변 처리
    with st.chat_message("assistant"):
        with st.spinner("AI가 판례를 검색하고 답변을 생성 중입니다..."):
            # 전체 파이프라인 실행
            response = run_legal_pipeline_reasoning(user_question)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
