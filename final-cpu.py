#python final-cpu.py --file ./sample_data/2.2.4 의료비 지출내역.pdf --ocr paddle --mode gpt-4o --mode all 2 --ocr-view

import os
import sys
import io
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import re
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global Constants for Text Detection
ALLOWED_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
ALLOWED_SYMBOLS = set(' .,!?;:()[]{}\'\"-/')
ALLOWED_ALL = ALLOWED_CHARS | ALLOWED_SYMBOLS

# OCR 라이브러리 시도
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    print("PaddleOCR이 설치되지 않았습니다. 설치: pip install paddleocr")

# EasyOCR은 지연 로딩 (필요할 때만 import)
HAS_EASYOCR = None
easyocr = None

def import_easyocr():
    """EasyOCR을 지연 로딩하는 함수"""
    global HAS_EASYOCR, easyocr
    if HAS_EASYOCR is not None:
        return HAS_EASYOCR  # 이미 시도했음
    
    try:
        import easyocr
        HAS_EASYOCR = True
        globals()['easyocr'] = easyocr
        return True
    except ImportError:
        HAS_EASYOCR = False
        print("EasyOCR이 설치되지 않았습니다. 설치: pip install easyocr")
        return False
    except Exception as e:
        HAS_EASYOCR = False
        print(f"EasyOCR import 오류: {e}")
        return False

# 대안 번역 라이브러리 시도
try:
    from deep_translator import GoogleTranslator
    HAS_GOOGLE_TRANSLATOR = True
except ImportError:
    HAS_GOOGLE_TRANSLATOR = False
    
try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF가 필요합니다. 설치: pip install pymupdf")

# CUDA 사용 가능 여부 확인
def check_cuda_available():
    """CUDA 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, device_count, device_name
        else:
            return False, 0, None
    except ImportError:
        # torch가 설치되지 않은 경우
        try:
            # PyTorch 없이도 CUDA 확인 시도 (torch 없이도 작동할 수 있음)
            return None, 0, None  # 알 수 없음
        except:
            return False, 0, None

# OpenAI API 키 설정 (환경 변수에서 가져오거나 직접 입력)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# OpenAI 모델 설정 (gpt-3.5-turbo: 빠르고 저렴, gpt-4o: 더 정확하지만 느리고 비쌈)
# 사용 가능한 모델: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5" (베타, 접근 권한 필요할 수 있음)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 기본값: gpt-4o-mini (경량 모델)
VALID_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-5"]

def is_numeric_only(text):
    """텍스트가 숫자만 포함하는지 확인 (번역 제외)"""
    if not text or not text.strip():
        return False
    text = text.strip()
    # 숫자와 일반적인 구분자(/ : . - 등)만 포함하는지 확인
    cleaned = text.replace('/', '').replace(':', '').replace('.', '').replace('-', '').replace(' ', '').replace(',', '')
    return cleaned.isdigit()

def is_korean_text(text):
    """텍스트가 한국어(한글)인지 확인
    
    한글이 하나라도 포함되어 있으면 True 반환
    한국어는 절대 번역하지 않고 그대로 유지함
    """
    if not text or len(text.strip()) < 1:
        return False
    
    text = text.strip()
    
    # 한글 유니코드 범위
    # 한글 음절: U+AC00-U+D7AF
    # 한글 자모: U+1100-U+11FF
    # 한글 자모 확장-A: U+A960-U+A97F
    # 한글 자모 확장-B: U+D7B0-U+D7FF
    korean_chars = sum(1 for c in text if 
                       '\uAC00' <= c <= '\uD7AF' or  # 한글 음절
                       '\u1100' <= c <= '\u11FF' or  # 한글 자모
                       '\uA960' <= c <= '\uA97F' or  # 한글 자모 확장-A
                       '\uD7B0' <= c <= '\uD7FF')    # 한글 자모 확장-B
    
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return False
    
    # 한글이 하나라도 있으면 한국어로 간주 (절대 번역하지 않음, 그대로 유지)
    return korean_chars > 0

def is_english_text(text):
    """텍스트가 영어인지 확인 (한국어가 포함되어 있으면 번역하지 않음)
    
    영어를 더 적극적으로 감지: 영문자가 하나라도 있으면 영어로 간주
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # 한국어가 포함되어 있으면 번역하지 않음
    if is_korean_text(text):
        return False
    
    # 숫자만 있는 경우는 번역하지 않음
    if is_numeric_only(text):
        return False
    
    # 영문자 개수 확인 (대소문자 구분 없이)
    english_letter_count = sum(1 for c in text if c.isalpha() and c in ALLOWED_CHARS)
    
    # 영문자가 하나라도 있으면 영어로 간주 (더 적극적으로)
    if english_letter_count > 0:
        # 숫자와 영어가 혼합된 경우도 영어로 간주 (예: "5 Ton", "NC1803005")
        # 단, 숫자만 있는 경우는 이미 위에서 제외됨
        
        # 영문자가 1개 이상이면 무조건 영어로 간주 (더 적극적)
        # 예: "A", "5 Ton", "NC1803005", "Address" 등 모두 영어로 간주
        return True
    
    return False

def is_arabic_text(text):
    """텍스트가 아랍어인지 확인"""
    if not text or len(text.strip()) < 2:
        return False
    # 아랍어 유니코드 범위: U+0600-U+06FF
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len([c for c in text if c.isalnum() or '\u0600' <= c <= '\u06FF'])
    if total_chars == 0:
        return False
    return arabic_chars / total_chars > 0.3  # 30% 이상이 아랍어 문자

def is_chinese_text(text):
    """텍스트가 한자인지 확인 (한국어 제외)"""
    if not text or len(text.strip()) < 1:
        return False
    
    # 먼저 한국어가 포함되어 있는지 확인 (한국어가 있으면 번역하지 않음)
    if is_korean_text(text):
        return False
    
    # 한자 유니코드 범위: U+4E00-U+9FFF (CJK 통합 한자)
    # U+3400-U+4DBF (CJK 확장 A)
    chinese_chars = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF' or '\u3400' <= c <= '\u4DBF')
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return False
    # 한자가 1개 이상이면 한자로 간주 (한 글자만 있어도 번역)
    return chinese_chars > 0

def should_translate_text(text, translation_mode='eng_ar'):
    """텍스트가 번역 대상인지 확인
    
    Args:
        text: 확인할 텍스트
        translation_mode: 번역 모드
            - 'eng_only': 영어만 번역
            - 'eng_chi': 영어만 번역 (한자는 그대로 표시)
            - 'eng_ar': 영어와 아랍어 번역
            - 'all': 영어, 아랍어, 한자 모두 번역 (기본 동작)
    
    번역 제외: 한국어(한글) - 한국어가 포함된 텍스트는 절대 번역하지 않음
    """
    # 한국어는 절대 번역하지 않음 (한글이 하나라도 있으면 그대로 유지)
    if is_korean_text(text):
        return False
    
    # 번역 모드에 따라 다르게 처리
    if translation_mode == 'eng_only':
        # 영어만 번역
        return is_english_text(text)
    elif translation_mode == 'eng_chi':
        # 영어와 중국어 번역 (한국어는 그대로 표시)
        return is_english_text(text) or is_chinese_text(text)
    elif translation_mode == 'eng_ar':
        # 영어와 아랍어만 번역
        return is_english_text(text) or is_arabic_text(text)
    else:
        # 기본: 영어, 아랍어, 한자 모두 번역
        return is_english_text(text) or is_arabic_text(text) or is_chinese_text(text)

def convert_date_format(text):
    """날짜 형식을 한국어 형식으로 변환
    예: "31-Jan" -> "01월 31일", "16-Apr" -> "04월 16일"
    """
    import re
    
    # 월 약어 매핑
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    
    # 날짜 패턴: DD-Mon (예: 31-Jan, 16-Apr)
    pattern = r'(\d{1,2})-([A-Za-z]{3})'
    
    def replace_date(match):
        day = match.group(1)
        month_abbr = match.group(2)
        month_num = month_map.get(month_abbr.capitalize())
        if month_num:
            # 일자가 한 자리 수면 앞에 0 추가하지 않음 (예: 1일, 31일)
            return f"{month_num}월 {day}일"
        return match.group(0)  # 매핑되지 않은 경우 원본 반환
    
    # 패턴 매칭 및 교체
    result = re.sub(pattern, replace_date, text)
    return result

def preprocess_text_for_translation(text):
    """번역 전 텍스트 전처리 - 약어 처리 및 날짜 형식 변환"""
    # 날짜 형식 변환 (예: "31-Jan" -> "01월 31일")
    text = convert_date_format(text)
    
    # "NO" 또는 "No."를 "Number"로 변환 (문서/표에서 자주 나오는 경우)
    text_upper = text.upper().strip()
    if text_upper == "NO" or text_upper == "NO.":
        # 주변 맥락이 없으면 Number로 가정
        return "Number"
    # "No. "로 시작하는 경우 (예: "No. 1" -> "Number 1")
    if text.strip().startswith("No. ") or text.strip().startswith("NO. "):
        return text.replace("No. ", "Number ").replace("NO. ", "Number ")
    return text

def translate_with_google(texts, debug=False):
    """Google Translator를 사용하여 텍스트를 한국어로 번역 (병렬 처리 최적화)"""
    if not texts:
        return []
    
    translations = [""] * len(texts)
    indices_to_translate = []
    
    # 짧은 텍스트만 번역 대상으로 선정
    for i, text in enumerate(texts):
        if len(text) < 500:
            indices_to_translate.append(i)
        else:
            translations[i] = text
            
    if not indices_to_translate:
        return translations

    def translate_single(idx, text):
        try:
            # Rate limit 방지를 위한 약간의 지연
            time.sleep(0.1)
            
            if HAS_GOOGLE_TRANSLATOR:
                translator = GoogleTranslator(source='auto', target='ko')
                return idx, translator.translate(text)
            elif HAS_GOOGLETRANS:
                translator = Translator()
                return idx, translator.translate(text, src='auto', dest='ko').text
            else:
                return idx, text
        except Exception as e:
            if debug:
                print(f"  Google Translator 실패: {text[:20]}... - {e}")
            return idx, text

    # 병렬 처리 (최대 4개 스레드로 제한하여 Rate Limit 방지)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(translate_single, i, texts[i]): i 
            for i in indices_to_translate
        }
        
        with tqdm(total=len(indices_to_translate), desc="Google 번역 진행 (병렬)", unit="텍스트", leave=True) as pbar:
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                translations[idx] = result
                pbar.update(1)
                
    return translations

def translate_to_korean(texts, languages=None, debug=False, model=None):
    """OpenAI API를 사용하여 영어, 아랍어, 또는 한자 텍스트를 한국어로 번역 (실패 시 Google Translator 사용)"""
    if not texts:
        return []
    
    # 모델 선택 (기본값: OPENAI_MODEL)
    if model is None:
        model = OPENAI_MODEL
    
    print(f"\n[API 요청 시작] OpenAI API를 사용하여 {len(texts)}개 텍스트 번역 시도 (모델: {model})")
    
    # OpenAI API 시도
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 모델에 따른 토큰 제한 및 배치 크기 조정 (Rate limit 방지를 위해 더 보수적으로 설정)
    if "gpt-5" in model.lower():
        # GPT-5 계열: 최신 모델, 더 큰 컨텍스트 지원 가능
        max_tokens = 4000
        avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 50
        if avg_text_length < 20:
            batch_size = 8   # Rate limit 방지를 위해 줄임
        elif avg_text_length < 50:
            batch_size = 5   # 중간 길이
        else:
            batch_size = 3   # 긴 텍스트는 더 적게
    elif "gpt-4" in model.lower():
        # GPT-4 계열: Rate limit이 엄격하므로 더 보수적으로 설정
        max_tokens = 4000
        avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 50
        if avg_text_length < 20:
            batch_size = 8   # Rate limit 방지를 위해 줄임 (기존 15 -> 8)
        elif avg_text_length < 50:
            batch_size = 5   # 중간 길이 (기존 10 -> 5)
        else:
            batch_size = 3   # 긴 텍스트는 더 적게 (기존 5 -> 3)
    else:
        # GPT-3.5-turbo: 더 빠르고 저렴, 더 관대한 rate limit
        max_tokens = 2000  # GPT-3.5는 더 작은 토큰 제한
        avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 50
        if avg_text_length < 20:
            batch_size = 12  # GPT-3.5는 더 큰 배치 가능하지만 보수적으로 (기존 20 -> 12)
        elif avg_text_length < 50:
            batch_size = 8   # 중간 길이 (기존 15 -> 8)
        else:
            batch_size = 5   # 긴 텍스트 (기존 10 -> 5)
    
    if debug:
        print(f"[모델 설정] {model} (배치 크기: {batch_size}, 최대 토큰: {max_tokens})")
    
    translations = []
    use_fallback = False
    
    # 총 배치 수 계산
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # tqdm으로 번역 진행 상황 표시
    with tqdm(total=len(texts), desc="번역 진행", unit="텍스트", leave=True) as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_text = "\n".join([f"{j+1}. {text}" for j, text in enumerate(batch)])
            current_batch = i // batch_size + 1
            
            # 언어 정보 확인
            has_english = any(is_english_text(text) for text in batch)
            has_arabic = any(is_arabic_text(text) for text in batch)
            has_chinese = any(is_chinese_text(text) for text in batch)
            
            # 한자가 포함된 경우 Google Translator로 직접 번역
            if has_chinese:
                if debug:
                    print(f"\n[한자 감지] 배치 {current_batch}: Google Translator로 직접 번역 (중국어 -> 한국어)")
                batch_translations = []
                if HAS_GOOGLE_TRANSLATOR:
                    try:
                        # 중국어를 명시적으로 지정하여 한국어로 번역
                        translator = GoogleTranslator(source='zh', target='ko')
                        for text in batch:
                            try:
                                # 중국어 텍스트를 그대로 번역 (OCR 결과를 그대로 사용)
                                translated = translator.translate(text)
                                batch_translations.append(translated)
                                time.sleep(0.2)  # API 제한 방지 (시간 증가)
                            except Exception as e:
                                if debug:
                                    print(f"  Google Translator 실패: {text[:20]}... - {e}")
                                # 실패 시 원문 유지
                                batch_translations.append(text)
                    except Exception as e:
                        if debug:
                            print(f"  Google Translator 초기화 실패: {e}")
                        batch_translations = batch.copy()
                elif HAS_GOOGLETRANS:
                    try:
                        translator = Translator()
                        for text in batch:
                            try:
                                # 중국어를 명시적으로 지정하여 한국어로 번역
                                translated = translator.translate(text, src='zh-CN', dest='ko').text
                                batch_translations.append(translated)
                                time.sleep(0.2)
                            except Exception as e:
                                if debug:
                                    print(f"  googletrans 실패: {text[:20]}... - {e}")
                                # 실패 시 원문 유지
                                batch_translations.append(text)
                    except Exception as e:
                        if debug:
                            print(f"googletrans 초기화 실패: {e}")
                        batch_translations = batch.copy()
                else:
                    batch_translations = batch.copy()
                
                translations.extend(batch_translations)
                pbar.update(len(batch))
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": "완료"})
                continue  # 다음 배치로 진행
            
            # 시스템 메시지 구성 (영어/아랍어만)
            system_message = "You are a professional translator specializing in document translation. "
            if has_english:
                system_message += "When translating English texts to Korean, consider the context:\n- 'NO' or 'No.' in documents/tables usually means 'Number' (번호), not 'No' (아니오)\n- 'ID' means 'Identifier' (식별자/아이디)\n- 'Qty' means 'Quantity' (수량)\n- 'Branch' in business/office context means '지점' (branch office), not '나뭇가지' (tree branch)\n- 'Manager' in business context means '관리자' or '지점장', not just '매니저'\n- Always consider the surrounding context in the document (tables, headers, labels) to determine the correct translation\n"
            if has_arabic:
                system_message += "When translating Arabic texts to Korean, preserve the meaning accurately and consider document context.\n"
            system_message += "Translate the following texts to Korean. The texts may be in English or Arabic. Respond with ONLY the Korean translations, one per line, in the same order. No numbering, no explanations."
            
            # API 요청 시도 (1회만 시도, 에러 발생 시 바로 Google Translator로 전환)
            try:
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": "API 요청 중"})
                
                # 첫 요청 전 지연 (Rate limit 방지)
                time.sleep(1.0)  # 첫 요청 전 1초 대기
                
                if debug:
                    print(f"\n[OpenAI API 호출] 배치 {current_batch}: {len(batch)}개 텍스트")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Translate to Korean (consider document context for abbreviations):\n{batch_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens  # 모델에 따른 토큰 제한
                )
                
                translated_text = response.choices[0].message.content
                
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": "응답 처리 중"})
                
                if debug:
                    print(f"[번역 API 응답 원본]:\n{translated_text[:200]}...")  # 처음 200자만
                
                # 번역 결과를 줄별로 분리
                translated_lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
                
                # 번호 제거
                cleaned_lines = []
                for line in translated_lines:
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    cleaned_lines.append(line)
                
                batch_translations = cleaned_lines[:len(batch)]
                translations.extend(batch_translations)
                
                # 진행 상황 업데이트
                pbar.update(len(batch))
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": "완료"})
                
                if debug:
                    print(f"[번역된 텍스트 개수]: {len(batch_translations)}개 (원본: {len(batch)}개)")
                
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = '429' in error_msg or 'rate_limit' in error_msg.lower()
                is_quota = 'quota' in error_msg.lower() or 'insufficient_quota' in error_msg.lower()
                
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": "에러 발생"})
                
                # 모든 에러: 바로 Google Translator로 전환 (재시도 없음)
                if is_rate_limit:
                    print(f"\n[Rate Limit 에러] OpenAI API Rate Limit 초과. Google Translator로 즉시 전환합니다.")
                elif is_quota:
                    print(f"\n[API 할당량 초과] OpenAI API 할당량 초과 (quota). Google Translator로 즉시 전환합니다.")
                else:
                    print(f"\n[API 에러] OpenAI API 에러 발생. Google Translator로 즉시 전환합니다.")
                    if debug:
                        print(f"  에러 상세: {error_msg[:200]}")
                
                use_fallback = True
                break  # 배치 루프 종료하고 바로 Google Translator로 전환
            
            # 배치 간 대기 시간 추가 (Rate limit 방지 - 더 보수적으로 설정)
            if i + batch_size < len(texts) and not use_fallback:
                # 배치 크기와 모델에 따라 대기 시간 조정 (Rate limit 방지를 위해 더 길게)
                if "gpt-4" in model.lower() or "gpt-5" in model.lower():
                    # GPT-4/5는 Rate limit이 엄격하므로 더 긴 대기
                    if batch_size >= 5:
                        wait_between_batches = 8.0  # 큰 배치는 8초 (기존 3초 -> 8초)
                    else:
                        wait_between_batches = 6.0  # 작은 배치는 6초 (기존 2.5초 -> 6초)
                else:
                    # GPT-3.5는 상대적으로 관대하지만 보수적으로
                    if batch_size >= 8:
                        wait_between_batches = 6.0  # 큰 배치는 6초
                    elif batch_size >= 5:
                        wait_between_batches = 5.0  # 중간 배치는 5초
                    else:
                        wait_between_batches = 4.0  # 작은 배치는 4초
                
                pbar.set_postfix({"배치": f"{current_batch}/{total_batches}", "상태": f"대기 중 ({wait_between_batches}초)"})
                
                if debug:
                    print(f"[대기 중] 다음 배치 전 {wait_between_batches}초 대기... (Rate limit 방지)")
                time.sleep(wait_between_batches)
    
    # OpenAI API가 실패한 경우 Google Translator 사용
    if use_fallback or len(translations) != len(texts):
        if HAS_GOOGLE_TRANSLATOR or HAS_GOOGLETRANS:
            fallback_translations = translate_with_google(texts, debug=debug)
            return fallback_translations
        else:
            if debug:
                print("⚠️ 대안 번역 라이브러리가 없습니다. 설치: pip install deep-translator")
            return texts
    
    print(f"[API 요청 완료] 총 {len(translations)}개 텍스트 번역 완료 (OpenAI API 사용)")
    return translations

def get_text_bbox(draw, text, font):
    """텍스트 바운딩 박스 가져오기 (호환성)"""
    try:
        # PIL 10.0+ 사용 textbbox
        return draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        try:
            # 이전 버전은 getbbox 직접 호출
            return font.getbbox(text)
        except:
            # 폴백: 텍스트 크기 추정
            bbox = draw.textbbox((0, 0), text, font=font) if hasattr(draw, 'textbbox') else None
            if bbox is None:
                # 대략적인 추정
                width = len(text) * font.size // 2
                height = font.size
                return (0, 0, width, height)
            return bbox

def wrap_text(text, font, max_width, draw=None):
    """텍스트를 주어진 너비에 맞게 여러 줄로 나누기"""
    if draw is None:
        # 임시 draw 객체 생성 (폰트 크기 측정용)
        from PIL import Image
        temp_img = Image.new('RGB', (100, 100))
        temp_draw = ImageDraw.Draw(temp_img)
        draw = temp_draw
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # 현재 줄에 단어 추가
        test_line = ' '.join(current_line + [word])
        bbox = get_text_bbox(draw, test_line, font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= max_width:
            current_line.append(word)
        else:
            # 현재 줄이 너무 길면 새 줄 시작
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    # 마지막 줄 추가
    if current_line:
        lines.append(' '.join(current_line))
    
    # 단어가 없거나 한 줄이 너무 길면 글자 단위로 나누기
    if not lines:
        lines = [text]
    else:
        # 첫 줄이 너무 길면 글자 단위로 나누기
        first_line_bbox = get_text_bbox(draw, lines[0], font)
        if first_line_bbox[2] - first_line_bbox[0] > max_width:
            char_lines = []
            current_line = ""
            for char in text:
                test_line = current_line + char
                bbox = get_text_bbox(draw, test_line, font)
                test_width = bbox[2] - bbox[0]
                if test_width <= max_width:
                    current_line += char
                else:
                    if current_line:
                        char_lines.append(current_line)
                    current_line = char
            if current_line:
                char_lines.append(current_line)
            lines = char_lines if char_lines else [text]
    
    return lines

def draw_text_on_image(image, text, bbox, font_size=None, show_highlight=True):
    """이미지에 텍스트를 그리기 (여러 줄 지원, 자동 폰트 크기 조정)
    
    Args:
        image: PIL Image 객체
        text: 그릴 텍스트
        bbox: 바운딩 박스 [x_min, y_min, x_max, y_max]
        font_size: 폰트 크기 (None이면 자동 조정)
        show_highlight: True면 색깔 하이라이트 표시, False면 텍스트만 표시 (PDF 저장용)
    """
    # RGBA 모드로 변환하여 반투명 배경 지원
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    draw = ImageDraw.Draw(image)
    
    # 바운딩 박스 크기 계산
    bbox_x_min = int(bbox[0])
    bbox_y_min = int(bbox[1])
    bbox_x_max = int(bbox[2])
    bbox_y_max = int(bbox[3])
    bbox_width = bbox_x_max - bbox_x_min
    bbox_height = bbox_y_max - bbox_y_min
    
    # 사용 가능한 너비와 높이 (패딩 제외)
    available_width = max(10, bbox_width - 10)  # 좌우 패딩
    available_height = max(10, bbox_height - 10)  # 상하 패딩
    
    # 폰트 크기 동적 조정 (텍스트가 바운딩 박스에 맞을 때까지)
    if font_size is None:
        font_size = max(18, min(36, int(bbox_height * 0.9)))  # 초기 폰트 크기 (1.5배)
    
    # 텍스트 언어에 따라 폰트 선택
    # 아랍어 텍스트인지 확인 (번역 실패 시 원본 아랍어가 표시될 수 있음)
    # 숫자와 함께 있어도 아랍어 문자가 하나라도 있으면 아랍어 폰트 사용
    # 아랍어 유니코드 범위: U+0600-U+06FF, U+0750-U+077F (아랍어 보충), U+08A0-U+08FF (아랍어 확장-A)
    has_arabic_char = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text)
    is_arabic = has_arabic_char  # 아랍어 문자가 하나라도 있으면 아랍어 폰트 사용
    
    if is_arabic:
        # 아랍어 폰트 시도 (Windows 기본 폰트 경로 - 아랍어 지원 폰트)
        # 우선순위: Arial, Tahoma (가장 안정적인 아랍어 지원)
        windows_font_paths = [
            r"C:\Windows\Fonts\arial.ttf",      # Arial (아랍어 지원, 가장 안정적)
            r"C:\Windows\Fonts\tahoma.ttf",     # Tahoma (아랍어 지원, 안정적)
            r"C:\Windows\Fonts\times.ttf",      # Times New Roman (아랍어 지원)
            r"C:\Windows\Fonts\segoeui.ttf",     # Segoe UI (아랍어 지원)
            r"C:\Windows\Fonts\calibri.ttf",     # Calibri (아랍어 지원)
        ]
    else:
        # 한글 폰트 시도 (Windows 기본 폰트 경로)
        windows_font_paths = [
            r"C:\Windows\Fonts\malgun.ttf",  # 맑은 고딕
            r"C:\Windows\Fonts\gulim.ttc",   # 굴림
            r"C:\Windows\Fonts\batang.ttc",  # 바탕
        ]
    
    # 폰트 크기를 줄여가며 텍스트가 바운딩 박스에 맞는지 확인
    best_font = None
    best_lines = []
    min_font_size = 12  # 1.5배 (8 * 1.5)
    
    for try_size in range(font_size, min_font_size - 1, -1):
        try:
            # 폰트 로드 시도
            temp_font = None
            for font_path in windows_font_paths:
                try:
                    if os.path.exists(font_path):
                        temp_font = ImageFont.truetype(font_path, try_size)
                        break
                except:
                    continue
            
            if temp_font is None:
                temp_font = ImageFont.load_default()
            
            # 텍스트를 여러 줄로 나누기
            lines = wrap_text(text, temp_font, available_width, draw)
            
            # 모든 줄의 높이 계산
            total_height = 0
            line_height = 0
            for line in lines:
                bbox = get_text_bbox(draw, line, temp_font)
                line_h = bbox[3] - bbox[1]
                line_height = max(line_height, line_h)
                total_height += line_h
            
            # 줄 간격 추가 (줄당 20% 여유)
            total_height_with_spacing = total_height + (len(lines) - 1) * int(line_height * 0.2)
            
            # 바운딩 박스 높이에 맞으면 사용
            if total_height_with_spacing <= available_height:
                best_font = temp_font
                best_lines = lines
                font_size = try_size
                break
        except:
            continue
    
    # 최적 폰트를 찾지 못한 경우 기본값 사용
    if best_font is None:
        for font_path in windows_font_paths:
            try:
                if os.path.exists(font_path):
                    best_font = ImageFont.truetype(font_path, min_font_size)
                    break
            except:
                continue
        if best_font is None:
            # 아랍어인 경우 기본 폰트가 아랍어를 지원하지 않을 수 있으므로
            # 아랍어 폰트를 다시 시도하거나 기본 폰트 사용
            if is_arabic:
                # 아랍어 폰트 재시도 (더 많은 폰트 시도)
                arabic_fonts = [
                    r"C:\Windows\Fonts\arial.ttf",      # Arial (가장 안정적)
                    r"C:\Windows\Fonts\tahoma.ttf",     # Tahoma (안정적)
                    r"C:\Windows\Fonts\times.ttf",      # Times New Roman
                    r"C:\Windows\Fonts\segoeui.ttf",     # Segoe UI
                    r"C:\Windows\Fonts\calibri.ttf",     # Calibri
                ]
                for font_path in arabic_fonts:
                    try:
                        if os.path.exists(font_path):
                            best_font = ImageFont.truetype(font_path, min_font_size)
                            # 폰트가 제대로 로드되었는지 확인
                            if best_font is not None:
                                break
                    except Exception as e:
                        continue
            if best_font is None:
                best_font = ImageFont.load_default()
        
        # 텍스트를 여러 줄로 나누기
        # 아랍어인 경우 텍스트 래핑을 단순화 (아랍어는 RTL이므로 복잡한 래핑이 필요할 수 있음)
        if is_arabic:
            # 아랍어는 단어 단위로 나누기보다는 전체 텍스트를 그대로 표시 시도
            # 너무 길면 강제로 나누기
            test_bbox = get_text_bbox(draw, text, best_font)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width <= available_width:
                best_lines = [text]  # 한 줄로 표시
            else:
                # 너무 길면 wrap_text 사용
                best_lines = wrap_text(text, best_font, available_width, draw)
        else:
            best_lines = wrap_text(text, best_font, available_width, draw)
    
    # 번역된 영역을 색깔로 표시 (OCR과 다른 색깔: 초록색 계열)
    # show_highlight가 False면 색깔 표시 없이 흰색 배경만 그리기 (PDF 저장용, 원본 텍스트 덮기)
    if show_highlight:
        padding = 2
        # 바운딩 박스 영역에 반투명 초록색 배경 추가 (하이라이트 효과)
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=(144, 238, 144, 150),  # 반투명 연한 초록색 배경 (RGBA) - 번역 영역 하이라이트
            outline=None
        )
        # 바운딩 박스 테두리 그리기 (초록색)
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=None,
            outline=(0, 128, 0, 255),  # 진한 초록색 테두리 (RGBA)
            width=2
        )
        
        # 텍스트 영역에 반투명 흰색 배경 추가 (텍스트 가독성 향상)
        text_padding = 3
        draw.rectangle(
            [bbox_x_min + text_padding, bbox_y_min + text_padding, bbox_x_max - text_padding, bbox_y_max - text_padding],
            fill=(255, 255, 255, 200),  # 반투명 흰색 배경 (RGBA)
            outline=None
        )
    else:
        # PDF 저장용: 원본 텍스트를 덮기 위한 흰색 배경 (test.py와 동일한 방식)
        padding = -3
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=(255, 255, 255, 255),  # 불투명 흰색 배경 (RGBA) - 원본 텍스트 덮기
            outline=None
        )
    
    # 텍스트 줄 높이 계산
    line_heights = []
    for line in best_lines:
        bbox = get_text_bbox(draw, line, best_font)
        line_heights.append(bbox[3] - bbox[1])
    
    if line_heights:
        max_line_height = max(line_heights)
        line_spacing = int(max_line_height * 0.2)  # 줄 간격
        total_text_height = sum(line_heights) + (len(best_lines) - 1) * line_spacing
        
        # 세로 정렬 (중앙)
        start_y = bbox_y_min + (bbox_height - total_text_height) / 2
        
        # 각 줄 그리기
        current_y = start_y
        for i, line in enumerate(best_lines):
            if not line.strip():
                current_y += line_heights[i] + line_spacing
                continue
            
            # 가로 정렬 (중앙)
            line_bbox = get_text_bbox(draw, line, best_font)
            line_width = line_bbox[2] - line_bbox[0]
            x = bbox_x_min + (bbox_width - line_width) / 2
            
            # 텍스트 그리기 - 검은색 (배경이 있어서 외곽선 불필요)
            draw.text((int(x), int(current_y)), line, fill=(0, 0, 0, 255), font=best_font)
            
            # 다음 줄 위치
            current_y += line_heights[i] + line_spacing
    
    return image

def draw_ocr_text_on_image(image, text, bbox, font_size=None, show_highlight=True):
    """OCR 결과 텍스트를 이미지에 그리기 (번역 없이 원본 텍스트 그대로)
    
    Args:
        image: PIL Image 객체
        text: 그릴 텍스트
        bbox: 바운딩 박스 [x_min, y_min, x_max, y_max]
        font_size: 폰트 크기 (None이면 자동 조정)
        show_highlight: True면 색깔 하이라이트 표시, False면 텍스트만 표시 (PDF 저장용)
    """
    # RGBA 모드로 변환하여 반투명 배경 지원
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    draw = ImageDraw.Draw(image)
    
    # 바운딩 박스 크기 계산
    bbox_x_min = int(bbox[0])
    bbox_y_min = int(bbox[1])
    bbox_x_max = int(bbox[2])
    bbox_y_max = int(bbox[3])
    bbox_width = bbox_x_max - bbox_x_min
    bbox_height = bbox_y_max - bbox_y_min
    
    # 사용 가능한 너비와 높이 (패딩 제외)
    available_width = max(10, bbox_width - 10)  # 좌우 패딩
    available_height = max(10, bbox_height - 10)  # 상하 패딩
    
    # 폰트 크기 동적 조정 (텍스트가 바운딩 박스에 맞을 때까지)
    if font_size is None:
        font_size = max(18, min(36, int(bbox_height * 0.9)))  # 초기 폰트 크기
    
    # 텍스트 언어에 따라 폰트 선택
    # 아랍어 텍스트인지 확인 (숫자와 함께 있어도 아랍어 문자가 하나라도 있으면 아랍어 폰트 사용)
    # 아랍어 유니코드 범위: U+0600-U+06FF, U+0750-U+077F (아랍어 보충), U+08A0-U+08FF (아랍어 확장-A)
    has_arabic_char = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text)
    is_arabic = has_arabic_char  # 아랍어 문자가 하나라도 있으면 아랍어 폰트 사용
    
    if is_arabic:
        # 아랍어 폰트 시도 (Windows 기본 폰트 경로 - 아랍어 지원 폰트)
        # 우선순위: Arial, Tahoma (가장 안정적인 아랍어 지원)
        windows_font_paths = [
            r"C:\Windows\Fonts\arial.ttf",      # Arial (아랍어 지원, 가장 안정적)
            r"C:\Windows\Fonts\tahoma.ttf",     # Tahoma (아랍어 지원, 안정적)
            r"C:\Windows\Fonts\times.ttf",      # Times New Roman (아랍어 지원)
            r"C:\Windows\Fonts\segoeui.ttf",     # Segoe UI (아랍어 지원)
            r"C:\Windows\Fonts\calibri.ttf",     # Calibri (아랍어 지원)
        ]
    else:
        # 한글 폰트 시도 (Windows 기본 폰트 경로)
        windows_font_paths = [
            r"C:\Windows\Fonts\malgun.ttf",  # 맑은 고딕
            r"C:\Windows\Fonts\gulim.ttc",   # 굴림
            r"C:\Windows\Fonts\batang.ttc",  # 바탕
        ]
    
    # 폰트 크기를 줄여가며 텍스트가 바운딩 박스에 맞는지 확인
    best_font = None
    best_lines = []
    min_font_size = 12
    
    for try_size in range(font_size, min_font_size - 1, -1):
        try:
            # 폰트 로드 시도
            temp_font = None
            for font_path in windows_font_paths:
                try:
                    if os.path.exists(font_path):
                        temp_font = ImageFont.truetype(font_path, try_size)
                        break
                except:
                    continue
            
            if temp_font is None:
                temp_font = ImageFont.load_default()
            
            # 텍스트를 여러 줄로 나누기
            lines = wrap_text(text, temp_font, available_width, draw)
            
            # 모든 줄의 높이 계산
            total_height = 0
            line_height = 0
            for line in lines:
                bbox = get_text_bbox(draw, line, temp_font)
                line_h = bbox[3] - bbox[1]
                line_height = max(line_height, line_h)
                total_height += line_h
            
            # 줄 간격 추가 (줄당 20% 여유)
            total_height_with_spacing = total_height + (len(lines) - 1) * int(line_height * 0.2)
            
            # 바운딩 박스 높이에 맞으면 사용
            if total_height_with_spacing <= available_height:
                best_font = temp_font
                best_lines = lines
                font_size = try_size
                break
        except:
            continue
    
    # 최적 폰트를 찾지 못한 경우 기본값 사용
    if best_font is None:
        for font_path in windows_font_paths:
            try:
                if os.path.exists(font_path):
                    best_font = ImageFont.truetype(font_path, min_font_size)
                    break
            except:
                continue
        if best_font is None:
            # 아랍어인 경우 기본 폰트가 아랍어를 지원하지 않을 수 있으므로
            # 아랍어 폰트를 다시 시도하거나 기본 폰트 사용
            if is_arabic:
                # 아랍어 폰트 재시도 (더 많은 폰트 시도)
                arabic_fonts = [
                    r"C:\Windows\Fonts\arial.ttf",      # Arial (가장 안정적)
                    r"C:\Windows\Fonts\tahoma.ttf",     # Tahoma (안정적)
                    r"C:\Windows\Fonts\times.ttf",      # Times New Roman
                    r"C:\Windows\Fonts\segoeui.ttf",     # Segoe UI
                    r"C:\Windows\Fonts\calibri.ttf",     # Calibri
                ]
                for font_path in arabic_fonts:
                    try:
                        if os.path.exists(font_path):
                            best_font = ImageFont.truetype(font_path, min_font_size)
                            # 폰트가 제대로 로드되었는지 확인
                            if best_font is not None:
                                break
                    except Exception as e:
                        continue
            if best_font is None:
                best_font = ImageFont.load_default()
        
        # 텍스트를 여러 줄로 나누기
        # 아랍어인 경우 텍스트 래핑을 단순화 (아랍어는 RTL이므로 복잡한 래핑이 필요할 수 있음)
        if is_arabic:
            # 아랍어는 단어 단위로 나누기보다는 전체 텍스트를 그대로 표시 시도
            # 너무 길면 강제로 나누기
            test_bbox = get_text_bbox(draw, text, best_font)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width <= available_width:
                best_lines = [text]  # 한 줄로 표시
            else:
                # 너무 길면 wrap_text 사용
                best_lines = wrap_text(text, best_font, available_width, draw)
        else:
            best_lines = wrap_text(text, best_font, available_width, draw)
    
    # OCR된 영역을 색깔로 표시
    # show_highlight가 False면 색깔 표시 없이 흰색 배경만 그리기 (PDF 저장용, 원본 텍스트 덮기)
    if show_highlight:
        padding = 2
        # 바운딩 박스 영역에 반투명 노란색 배경 추가 (하이라이트 효과)
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=(255, 255, 0, 150),  # 반투명 노란색 배경 (RGBA) - 하이라이트
            outline=None
        )
        # 바운딩 박스 테두리 그리기 (빨간색)
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=None,
            outline=(255, 0, 0, 255),  # 빨간색 테두리 (RGBA)
            width=2
        )
        
        # 텍스트 영역에 반투명 흰색 배경 추가 (텍스트 가독성 향상)
        text_padding = 3
        draw.rectangle(
            [bbox_x_min + text_padding, bbox_y_min + text_padding, bbox_x_max - text_padding, bbox_y_max - text_padding],
            fill=(255, 255, 255, 200),  # 반투명 흰색 배경 (RGBA)
            outline=None
        )
    else:
        # PDF 저장용: 원본 텍스트를 덮기 위한 흰색 배경 (test.py와 동일한 방식)
        padding = -3
        draw.rectangle(
            [bbox_x_min - padding, bbox_y_min - padding, bbox_x_max + padding, bbox_y_max + padding],
            fill=(255, 255, 255, 255),  # 불투명 흰색 배경 (RGBA) - 원본 텍스트 덮기
            outline=None
        )
    
    # 텍스트 줄 높이 계산
    line_heights = []
    for line in best_lines:
        bbox = get_text_bbox(draw, line, best_font)
        line_heights.append(bbox[3] - bbox[1])
    
    if line_heights:
        max_line_height = max(line_heights)
        line_spacing = int(max_line_height * 0.2)  # 줄 간격
        total_text_height = sum(line_heights) + (len(best_lines) - 1) * line_spacing
        
        # 세로 정렬 (중앙)
        start_y = bbox_y_min + (bbox_height - total_text_height) / 2
        
        # 각 줄 그리기
        current_y = start_y
        for i, line in enumerate(best_lines):
            if not line.strip():
                current_y += line_heights[i] + line_spacing
                continue
            
            # 가로 정렬 (중앙)
            line_bbox = get_text_bbox(draw, line, best_font)
            line_width = line_bbox[2] - line_bbox[0]
            x = bbox_x_min + (bbox_width - line_width) / 2
            
            # 텍스트 그리기 - 검은색 (배경이 있어서 외곽선 불필요)
            draw.text((int(x), int(current_y)), line, fill=(0, 0, 0, 255), font=best_font)
            
            # 다음 줄 위치
            current_y += line_heights[i] + line_spacing
    
    return image

def combine_images_side_by_side(img1, img2):
    """두 이미지를 나란히 붙이기 (왼쪽: img1, 오른쪽: img2)"""
    # 두 이미지의 높이를 맞추기 (더 큰 높이에 맞춤)
    max_height = max(img1.height, img2.height)
    
    # 각 이미지의 비율을 유지하면서 높이 조정
    if img1.height != max_height:
        ratio = max_height / img1.height
        new_width = int(img1.width * ratio)
        img1 = img1.resize((new_width, max_height), Image.Resampling.LANCZOS)
    
    if img2.height != max_height:
        ratio = max_height / img2.height
        new_width = int(img2.width * ratio)
        img2 = img2.resize((new_width, max_height), Image.Resampling.LANCZOS)
    
    # 두 이미지를 나란히 붙이기
    total_width = img1.width + img2.width
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    
    return combined

def combine_three_images_side_by_side(img1, img2, img3):
    """세 이미지를 나란히 붙이기 (왼쪽: img1, 중간: img2, 오른쪽: img3)
    
    Returns:
        (결합된 이미지, img1_width, img2_width, img3_width) 튜플
    """
    # 세 이미지의 높이를 맞추기 (더 큰 높이에 맞춤)
    max_height = max(img1.height, img2.height, img3.height)
    
    # 각 이미지의 비율을 유지하면서 높이 조정
    if img1.height != max_height:
        ratio = max_height / img1.height
        new_width = int(img1.width * ratio)
        img1 = img1.resize((new_width, max_height), Image.Resampling.LANCZOS)
    else:
        new_width = img1.width
    
    img1_width = new_width
    
    if img2.height != max_height:
        ratio = max_height / img2.height
        new_width = int(img2.width * ratio)
        img2 = img2.resize((new_width, max_height), Image.Resampling.LANCZOS)
    else:
        new_width = img2.width
    
    img2_width = new_width
    
    if img3.height != max_height:
        ratio = max_height / img3.height
        new_width = int(img3.width * ratio)
        img3 = img3.resize((new_width, max_height), Image.Resampling.LANCZOS)
    else:
        new_width = img3.width
    
    img3_width = new_width
    
    # 세 이미지를 나란히 붙이기
    total_width = img1_width + img2_width + img3_width
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1_width, 0))
    combined.paste(img3, (img1_width + img2_width, 0))
    
    return combined, img1_width, img2_width, img3_width

def add_titles_to_combined_image(combined_img, img1_width, img2_width, img3_width):
    """결합된 이미지에 각 섹션별 제목 추가
    
    Args:
        combined_img: 결합된 이미지
        img1_width: 첫 번째 이미지 너비
        img2_width: 두 번째 이미지 너비
        img3_width: 세 번째 이미지 너비
    
    Returns:
        제목이 추가된 이미지
    """
    # RGB 모드로 변환 (제목을 RGB 모드에서 직접 그리기 위해)
    if combined_img.mode != 'RGB':
        combined_img = combined_img.convert('RGB')
    
    draw = ImageDraw.Draw(combined_img)
    
    # 제목 폰트 크기 2배 이상 증가
    font_size = 70  # 32 -> 70 (2배 이상)
    best_font = None
    
    windows_font_paths = [
        'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
        'C:/Windows/Fonts/gulim.ttc',    # 굴림
        'C:/Windows/Fonts/batang.ttc',   # 바탕
    ]
    
    for font_path in windows_font_paths:
        try:
            if os.path.exists(font_path):
                best_font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    if best_font is None:
        best_font = ImageFont.load_default()
    
    # 제목 설정
    titles = [
        ("원본", img1_width // 2),  # 왼쪽 이미지 중앙
        ("OCR 감지", img1_width + img2_width // 2),  # 중간 이미지 중앙
        ("결과(번역본)", img1_width + img2_width + img3_width // 2)  # 오른쪽 이미지 중앙
    ]
    
    # 제목 위치 (상단 중앙)
    title_y = 20  # 상단에서 약간 아래
    title_height = 85  # 높이 2배 이상 증가
    
    for title_text, center_x in titles:
        # 텍스트 너비 계산
        bbox = get_text_bbox(draw, title_text, best_font)
        text_width = bbox[2] - bbox[0]
        text_x = center_x - text_width // 2
        
        # 제목 배경 (빨간색)
        padding = 20  # 패딩 증가
        draw.rectangle(
            [text_x - padding, title_y - 5, text_x + text_width + padding, title_y + title_height],
            fill=(220, 20, 60),  # 빨간색 배경
            outline=(180, 10, 40),  # 진한 빨간색 테두리
            width=2
        )
        
        # 제목 텍스트 (흰색)
        draw.text(
            (text_x, title_y),
            title_text,
            fill=(255, 255, 255),  # 흰색 텍스트
            font=best_font
        )
    
    return combined_img

def add_color_tags_to_image(image, has_translation=False, has_ocr=False):
    """이미지 상단 오른쪽에 색상 태그 추가
    
    Args:
        image: PIL Image 객체
        has_translation: 번역본이 있는지 여부 (초록색 태그)
        has_ocr: OCR 감지된 부분이 있는지 여부 (주황색 태그)
    
    Returns:
        태그가 추가된 이미지
    """
    if not has_translation and not has_ocr:
        return image
    
    # 이미지를 RGBA 모드로 변환 (투명도 지원)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    draw = ImageDraw.Draw(image)
    
    # 태그들을 오른쪽에서 왼쪽으로 배치
    tags = []
    if has_translation:
        tags.append(("번역이 된 부분", (0, 128, 0)))  # 초록색
    if has_ocr:
        tags.append(("감지된 부분", (255, 165, 0)))  # 주황색
    
    # 태그가 없으면 원본 반환
    if not tags:
        return image
    
    # 폰트 크기 증가
    font_size = 24
    best_font = None
    
    windows_font_paths = [
        'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
        'C:/Windows/Fonts/gulim.ttc',    # 굴림
        'C:/Windows/Fonts/batang.ttc',   # 바탕
    ]
    
    for font_path in windows_font_paths:
        try:
            if os.path.exists(font_path):
                best_font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    if best_font is None:
        best_font = ImageFont.load_default()
    
    # 태그 위치 설정 (상단 오른쪽, 안쪽으로 여유 공간)
    padding = 30  # 가장자리에서 안쪽으로 여유 공간
    tag_height = 40  # 태그 높이 증가
    tag_spacing = 8  # 태그 간격 증가
    
    # 이미지 크기
    img_width = image.width
    img_height = image.height
    
    # 태그 너비 계산
    tag_widths = []
    for tag_text, _ in tags:
        bbox = get_text_bbox(draw, tag_text, best_font)
        tag_width = bbox[2] - bbox[0] + 30  # 텍스트 너비 + 패딩 증가
        tag_widths.append(tag_width)
    
    # 시작 위치 계산 (오른쪽에서 왼쪽으로)
    current_x = img_width - padding
    
    # 태그 그리기 (오른쪽에서 왼쪽으로)
    for i, (tag_text, tag_color) in enumerate(tags):
        tag_width = tag_widths[i]
        current_x -= tag_width
        
        # 태그 배경 (반투명)
        tag_y = padding  # 상단에서 안쪽으로 여유 공간
        draw.rectangle(
            [current_x, tag_y, current_x + tag_width, tag_y + tag_height],
            fill=(*tag_color, 200),  # 반투명 색상
            outline=(*tag_color, 255),  # 진한 테두리
            width=2
        )
        
        # 태그 텍스트
        text_x = current_x + 15
        text_y = tag_y + (tag_height - font_size) // 2
        draw.text(
            (text_x, text_y),
            tag_text,
            fill=(255, 255, 255),  # 흰색 텍스트
            font=best_font
        )
        
        current_x -= tag_spacing  # 다음 태그를 위한 간격
    
    # RGB 모드로 변환
    if image.mode == 'RGBA':
        final_img = Image.new('RGB', image.size, (255, 255, 255))
        final_img.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
        image = final_img
    
    return image

def parse_paddleocr_result(result):
    """PaddleOCR 결과를 파싱하여 텍스트와 바운딩 박스 리스트 반환
    
    Args:
        result: PaddleOCR.ocr()의 반환값 ([[[bbox], (text, confidence)], ...] 형식)
    
    Returns:
        (texts, bboxes, scores) 튜플 - 텍스트 리스트, 바운딩 박스 리스트, 신뢰도 리스트
    """
    texts = []
    bboxes = []
    scores = []
    
    if not result or len(result) == 0:
        return texts, bboxes, scores
    
    res = result[0] if isinstance(result, list) else result
    
    if isinstance(res, list):
        # PaddleOCR 표준 형식: [[[bbox], (text, confidence)], ...]
        for line in res:
            if not line or len(line) < 2:
                continue
            bbox = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text_info = line[1]  # (text, confidence)
            
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                text = text_info[0]
                score = text_info[1] if len(text_info) >= 2 else 1.0
            else:
                text = str(text_info)
                score = 1.0
            
            if not text or not text.strip():
                continue
            
            try:
                if bbox and len(bbox) >= 4:
                    x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                    y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                    
                    if x_coords and y_coords:
                        texts.append(text)
                        bboxes.append(bbox)
                        scores.append(score)
            except Exception:
                pass
    elif isinstance(res, dict):
        # 커스텀 딕셔너리 형식 (두 OCR 결과를 합친 경우)
        rec_texts = res.get('rec_texts', [])
        dt_polys = res.get('dt_polys', [])
        rec_scores = res.get('rec_scores', [])
        
        for i, text in enumerate(rec_texts):
            if not text or not text.strip():
                continue
            bbox = dt_polys[i] if i < len(dt_polys) else None
            score = rec_scores[i] if i < len(rec_scores) else 1.0
            
            if bbox is not None and len(bbox) > 0:
                texts.append(text)
                bboxes.append(bbox)
                scores.append(score)
    
    # 텍스트 박스 병합 (파편화된 인식 결과 개선)
    texts, bboxes, scores = merge_ocr_results(texts, bboxes, scores)
    
    return texts, bboxes, scores

def merge_ocr_results(texts, bboxes, scores, x_threshold=30, y_threshold=15):
    """
    PaddleOCR 결과(texts, bboxes, scores)를 받아 가까운 박스들을 병합합니다.
    같은 라인에 있고 x거리가 가까운 텍스트들을 하나로 합칩니다.
    """
    if not texts or not bboxes:
        return texts, bboxes, scores

    data = []
    for i, (text, bbox, score) in enumerate(zip(texts, bboxes, scores)):
        if not text or not text.strip():
            continue
            
        try:
            pts = bbox
            if hasattr(bbox, 'tolist'): 
                pts = bbox.tolist()
            elif not isinstance(bbox, list):
                pts = list(bbox)
                
            # 4점 폴리곤인지 확인
            if len(pts) == 4 and isinstance(pts[0], list):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            else:
                # [x1, y1, x2, y2] 형식인 경우
                x_min, y_min, x_max, y_max = pts[0], pts[1], pts[2], pts[3]
                pts = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                
            y_center = (y_min + y_max) / 2
            height = y_max - y_min
            
            data.append({
                'text': text,
                'bbox': pts,
                'score': score,
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'y_center': y_center, 'height': height
            })
        except Exception:
            continue

    # Y축 기준 정렬 (같은 라인끼리 모으기)
    # y_threshold/2 단위로 그룹화하여 정렬
    data.sort(key=lambda k: (int(k['y_center'] / (y_threshold/2)), k['x_min']))
    
    merged = []
    if not data: 
        return [], [], []
    
    curr = data[0]
    
    for i in range(1, len(data)):
        item = data[i]
        
        # 병합 조건 확인
        # 1. 같은 라인인지 (Y 중심 차이가 작음)
        y_dist = abs(curr['y_center'] - item['y_center'])
        is_same_line = y_dist < y_threshold
        
        # 2. X축 거리가 가까운지
        # item이 curr보다 오른쪽에 있다고 가정 (정렬됨)
        # 겹치거나 거리가 threshold 이내
        dist_x = item['x_min'] - curr['x_max']
        is_close_x = dist_x < x_threshold
        
        if is_same_line and is_close_x:
            # 병합 실행
            curr['text'] += " " + item['text']
            
            # bbox 업데이트 (두 박스를 포함하는 직사각형)
            new_x_min = min(curr['x_min'], item['x_min'])
            new_y_min = min(curr['y_min'], item['y_min'])
            new_x_max = max(curr['x_max'], item['x_max'])
            new_y_max = max(curr['y_max'], item['y_max'])
            
            curr['x_min'] = new_x_min
            curr['y_min'] = new_y_min
            curr['x_max'] = new_x_max
            curr['y_max'] = new_y_max
            curr['y_center'] = (new_y_min + new_y_max) / 2
            curr['height'] = new_y_max - new_y_min
            
            # 4점 폴리곤 재구성
            curr['bbox'] = [
                [new_x_min, new_y_min],
                [new_x_max, new_y_min],
                [new_x_max, new_y_max],
                [new_x_min, new_y_max]
            ]
            
            # 점수 평균 (가중 평균 가능하지만 단순 평균)
            curr['score'] = (curr['score'] + item['score']) / 2
        else:
            merged.append(curr)
            curr = item
            
    merged.append(curr)
    
    return [m['text'] for m in merged], [m['bbox'] for m in merged], [m['score'] for m in merged]

def process_pdf_page_ocr_only(page, ocr, ocr_type='paddle', ocr_ar=None, debug=False, page_num=None):
    """PDF 페이지를 처리하여 OCR 결과만 이미지에 그리기 (번역 없음)
    
    Args:
        page: PyMuPDF 페이지 객체
        ocr: OCR 객체 (PaddleOCR 또는 EasyOCR, 주 OCR)
        ocr_type: 'paddle' 또는 'easyocr'
        ocr_ar: 아랍어 OCR 객체 (필요한 경우)
        debug: 디버그 모드
        page_num: 페이지 번호 (디버깅용)
    
    Returns:
        (원본 이미지, OCR 결과가 그려진 이미지) 튜플
    """
    # 페이지를 이미지로 변환 (고해상도)
    mat = fitz.Matrix(3.0, 3.0)  # 3x 확대
    pix = page.get_pixmap(matrix=mat)
    
    # PIL Image로 변환 (메모리 최적화)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    
    # 원본 이미지 복사
    original_img = img.copy()
    
    # 이미지를 임시 파일로 저장하여 OCR 처리
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    # PyMuPDF로 직접 저장 (속도 최적화)
    pix.save(tmp_path)
    
    try:
        # OCR 엔진에 따라 다른 방식으로 처리
        all_texts = []
        all_bboxes = []
        
        if ocr_type == 'paddle':
            # PaddleOCR 사용
            if ocr_ar is not None:
                # 두 OCR 모두 실행 (한국어+영어와 중국어/아랍어+영어)
                result_ko = ocr.predict(input=tmp_path)
                result_other = ocr_ar.predict(input=tmp_path)
                
                # 두 결과를 합치기 (중복 제거를 위해 bbox 기반으로)
                all_rec_texts = []
                all_dt_polys = []
                seen_bboxes = set()
                
                # eng_chi 모드에서는 두 OCR 결과를 모두 수집하고 언어별로 더 적절한 결과 선택
                # 전역 변수 TRANSLATION_MODE 사용
                translation_mode = globals().get('TRANSLATION_MODE', 'all')
                
                if translation_mode == 'eng_chi':
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox)
                    ocr_results_ch = {}  # bbox_key -> (text, bbox)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox)
                                    except Exception:
                                        pass
                    
                    # 중국어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ch[bbox_key] = (text, bbox)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (중복되는 경우 언어별로 더 적절한 결과 선택)
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ch.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko = ocr_results_ko.get(bbox_key, (None, None))
                        text_ch, bbox_ch = ocr_results_ch.get(bbox_key, (None, None))
                        
                        if text_ko and text_ch:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            if is_korean_text(text_ko):
                                # 한국어가 포함된 경우 한국어 OCR 결과 우선
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                            elif is_chinese_text(text_ch):
                                # 중국어가 포함된 경우 중국어 OCR 결과 우선
                                all_rec_texts.append(text_ch)
                                all_dt_polys.append(bbox_ch)
                            else:
                                # 둘 다 영어인 경우 한국어 OCR 결과 우선 (더 정확할 가능성)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                        elif text_ch:
                            # 중국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ch)
                            all_dt_polys.append(bbox_ch)
                else:
                    # eng_ar 또는 all 모드: 언어별로 더 적절한 결과 선택
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox)
                    ocr_results_ar = {}  # bbox_key -> (text, bbox)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox)
                                    except Exception:
                                        pass
                    
                    # 아랍어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ar[bbox_key] = (text, bbox)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                    # 같은 bbox에 대해 두 OCR 결과가 모두 있을 때 언어별로 더 적절한 결과 선택
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ar.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko = ocr_results_ko.get(bbox_key, (None, None))
                        text_ar, bbox_ar = ocr_results_ar.get(bbox_key, (None, None))
                        
                        if text_ko and text_ar:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                            has_arabic_ar = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ar)
                            has_korean_ko = is_korean_text(text_ko)
                            has_korean_ar = is_korean_text(text_ar)
                            
                            # 숫자 포함 여부 확인
                            has_numbers_ko = any(c.isdigit() for c in text_ko)
                            has_numbers_ar = any(c.isdigit() for c in text_ar)
                            
                            if has_korean_ko and not has_arabic_ar:
                                # 한국어 OCR 결과에 한국어가 있고, 아랍어 OCR 결과에 아랍어가 없으면 한국어 OCR 결과 우선
                                # 원본이 한국어면 OCR 인식도 한국어로
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                            elif has_arabic_ar:
                                # 아랍어 OCR 결과에 아랍어가 있는 경우
                                if has_numbers_ar:
                                    # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                elif has_numbers_ko and not has_numbers_ar:
                                    # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                    # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                    # 예: "10089-CP-104" 같은 전체 문자열 추출
                                    non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                    # 아랍어 OCR에서 아랍어 부분만 추출
                                    arabic_ar = ''.join(c for c in text_ar if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                    # 숫자/기호와 아랍어를 공백으로 합치기
                                    combined_text = f"{non_arabic_ko.strip()} {arabic_ar.strip()}".strip()
                                    all_rec_texts.append(combined_text)
                                    all_dt_polys.append(bbox_ar)  # 아랍어 OCR의 bbox 사용
                                else:
                                    # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                            elif has_arabic_ko and not has_korean_ko:
                                # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                            else:
                                # 둘 다 영어/숫자만인 경우 더 긴 텍스트 선택
                                if len(text_ar) > len(text_ko):
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                else:
                                    all_rec_texts.append(text_ko)
                                    all_dt_polys.append(bbox_ko)
                        elif text_ar:
                            # 아랍어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ar)
                            all_dt_polys.append(bbox_ar)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                
                result = [{
                    'rec_texts': all_rec_texts,
                    'dt_polys': all_dt_polys
                }] if all_rec_texts else []
            else:
                # 단일 OCR 사용
                result = ocr.predict(input=tmp_path)
            
            # 모든 텍스트와 바운딩 박스 추출
            if result and len(result) > 0:
                res = result[0]
                if isinstance(res, dict):
                    rec_texts = res.get('rec_texts', [])
                    dt_polys = res.get('dt_polys', [])
                    
                    for i, text in enumerate(rec_texts):
                        if not text or not text.strip():
                            continue
                        
                        bbox = dt_polys[i] if i < len(dt_polys) else None
                        if bbox is not None and len(bbox) > 0:
                            try:
                                if hasattr(bbox, 'tolist'):
                                    bbox = bbox.tolist()
                                elif not isinstance(bbox, list):
                                    bbox = list(bbox)
                                
                                x_coords = []
                                y_coords = []
                                for point in bbox:
                                    if hasattr(point, '__getitem__') and len(point) >= 2:
                                        try:
                                            x_coords.append(float(point[0]))
                                            y_coords.append(float(point[1]))
                                        except (TypeError, IndexError, ValueError):
                                            continue
                                
                                if x_coords and y_coords:
                                    all_texts.append(text)
                                    all_bboxes.append([
                                        min(x_coords),
                                        min(y_coords),
                                        max(x_coords),
                                        max(y_coords)
                                    ])
                            except Exception as e:
                                pass
        
        elif ocr_type == 'easyocr':
            # EasyOCR 사용
            if ocr_ar is not None:
                # 두 OCR 모두 실행
                result_ko = ocr.readtext(tmp_path)  # ['ko', 'en']
                result_other = ocr_ar.readtext(tmp_path)  # ['en', 'ch_sim'] 또는 아랍어 OCR
                
                # 두 결과를 합치기 (언어별로 더 적절한 결과 선택)
                # EasyOCR 결과 형식: [(bbox, text, confidence), ...]
                ocr_results_ko = {}  # bbox_key -> (text, bbox, confidence)
                ocr_results_other = {}  # bbox_key -> (text, bbox, confidence)
                
                # 한국어 OCR 결과 수집
                if result_ko:
                    for item in result_ko:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_ko[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 아랍어/중국어 OCR 결과 수집
                if result_other:
                    for item in result_other:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_other[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                all_results = []
                all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_other.keys())
                for bbox_key in all_bbox_keys:
                    text_ko, bbox_ko, conf_ko = ocr_results_ko.get(bbox_key, (None, None, 0.0))
                    text_other, bbox_other, conf_other = ocr_results_other.get(bbox_key, (None, None, 0.0))
                    
                    if text_ko and text_other:
                        # 중복되는 경우: 언어별로 더 적절한 결과 선택
                        has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                        has_arabic_other = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_other)
                        has_korean_ko = is_korean_text(text_ko)
                        has_korean_other = is_korean_text(text_other)
                        
                        # 숫자 포함 여부 확인
                        has_numbers_ko = any(c.isdigit() for c in text_ko)
                        has_numbers_other = any(c.isdigit() for c in text_other)
                        
                        if has_korean_ko and not has_arabic_other:
                            # 한국어 OCR 결과에 한국어가 있고, 아랍어 OCR 결과에 아랍어가 없으면 한국어 OCR 결과 우선
                            # 원본이 한국어면 OCR 인식도 한국어로
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        elif has_arabic_other:
                            # 아랍어 OCR 결과에 아랍어가 있는 경우
                            if has_numbers_other:
                                # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                all_results.append((bbox_other, text_other, conf_other))
                            elif has_numbers_ko and not has_numbers_other:
                                # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                # 예: "10089-CP-104" 같은 전체 문자열 추출
                                non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                # 아랍어 OCR에서 아랍어 부분만 추출
                                arabic_other = ''.join(c for c in text_other if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                # 숫자/기호와 아랍어를 공백으로 합치기
                                combined_text = f"{non_arabic_ko.strip()} {arabic_other.strip()}".strip()
                                all_results.append((bbox_other, combined_text, conf_other))
                            else:
                                # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                all_results.append((bbox_other, text_other, conf_other))
                        elif has_arabic_ko and not has_korean_ko:
                            # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        else:
                            # 둘 다 영어/숫자만인 경우 더 긴 텍스트 선택
                            if len(text_other) > len(text_ko):
                                all_results.append((bbox_other, text_other, conf_other))
                            else:
                                all_results.append((bbox_ko, text_ko, conf_ko))
                    elif text_other:
                        # 아랍어/중국어 OCR 결과만 있는 경우
                        all_results.append((bbox_other, text_other, conf_other))
                    elif text_ko:
                        # 한국어 OCR 결과만 있는 경우
                        all_results.append((bbox_ko, text_ko, conf_ko))
                
                result = all_results
            else:
                # 단일 OCR 사용
                result = ocr.readtext(tmp_path)
            
            # 모든 텍스트와 바운딩 박스 추출
            if result:
                for item in result:
                    if len(item) >= 3:
                        bbox_points = item[0]
                        text = item[1]
                        
                        if not text or not text.strip():
                            continue
                        
                        try:
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            
                            if x_coords and y_coords:
                                all_texts.append(text)
                                all_bboxes.append([
                                    min(x_coords),
                                    min(y_coords),
                                    max(x_coords),
                                    max(y_coords)
                                ])
                        except Exception as e:
                            pass
        else:
            raise ValueError(f"지원하지 않는 OCR 타입: {ocr_type}")
        
        # 디버깅 정보 출력
        if debug and page_num is not None:
            print(f"\n[페이지 {page_num} OCR 결과]")
            print(f"총 인식된 텍스트: {len(all_texts)}개")
            if all_texts:
                print(f"  예시 (처음 5개): {all_texts[:5]}")
        
        # OCR 결과를 이미지에 그리기
        ocr_img = img.copy()
        if all_texts:
            print(f"\n[OCR 결과 그리기] 페이지 {page_num if page_num is not None else '?'}: {len(all_texts)}개 텍스트를 이미지에 그립니다.")
            for text, bbox in zip(all_texts, all_bboxes):
                ocr_img = draw_ocr_text_on_image(ocr_img, text, bbox)
        else:
            print(f"\n[OCR 결과 없음] 페이지 {page_num if page_num is not None else '?'}: 인식된 텍스트가 없습니다.")
        
        # RGB 모드로 변환
        if ocr_img.mode == 'RGBA':
            final_img = Image.new('RGB', ocr_img.size, (255, 255, 255))
            final_img.paste(ocr_img, mask=ocr_img.split()[3] if len(ocr_img.split()) == 4 else None)
            ocr_img = final_img
        
        if original_img.mode == 'RGBA':
            final_original = Image.new('RGB', original_img.size, (255, 255, 255))
            final_original.paste(original_img, mask=original_img.split()[3] if len(original_img.split()) == 4 else None)
            original_img = final_original
        
        return original_img, ocr_img
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_pdf_page_full_view(page, ocr, ocr_type='paddle', translation_mode='all', ocr_ar=None, debug=False, page_num=None, show_highlight=True, translations_dict=None):
    """PDF 페이지를 처리하여 원본, OCR 결과, 번역 결과 이미지를 모두 반환
    
    Args:
        page: PyMuPDF 페이지 객체
        ocr: OCR 객체 (PaddleOCR 또는 EasyOCR, 주 OCR)
        ocr_type: 'paddle' 또는 'easyocr'
        translation_mode: 번역 모드 ('eng_ar', 'eng_only', 'all')
        ocr_ar: 아랍어 OCR 객체
        debug: 디버그 모드
        page_num: 페이지 번호 (디버깅용)
    
    Returns:
        (원본 이미지, OCR 결과 이미지, 번역 결과 이미지) 튜플
    """
    # 페이지를 이미지로 변환 (고해상도)
    mat = fitz.Matrix(3.0, 3.0)  # 3x 확대
    pix = page.get_pixmap(matrix=mat)
    
    # PIL Image로 변환 (메모리 최적화)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    
    # 원본 이미지 복사
    original_img = img.copy()
    
    # 이미지를 임시 파일로 저장하여 OCR 처리
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    # PyMuPDF로 직접 저장 (속도 최적화)
    pix.save(tmp_path)
    
    try:
        # OCR 엔진에 따라 다른 방식으로 처리
        if ocr_type == 'paddle':
            # PaddleOCR 사용
            if ocr_ar is not None:
                # all 모드인 경우: 한국어 OCR, 아랍어 OCR 모두 실행
                if translation_mode == 'all':
                    result_ko = ocr.predict(input=tmp_path)
                    result_arabic = ocr_ar.predict(input=tmp_path)
                    result_other = result_arabic  # 아랍어 OCR 결과
                else:
                    # eng_chi 또는 eng_ar 모드: 두 OCR 모두 실행
                    result_ko = ocr.predict(input=tmp_path)
                    result_other = ocr_ar.predict(input=tmp_path)
                
                # 두 결과를 합치기 (중복 제거를 위해 bbox 기반으로)
                all_rec_texts = []
                all_dt_polys = []
                all_rec_scores = []
                seen_bboxes = set()
                
                # eng_chi 모드에서는 두 OCR 결과를 모두 수집하고 언어별로 더 적절한 결과 선택
                if translation_mode == 'eng_chi':
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox, score)
                    ocr_results_ch = {}  # bbox_key -> (text, bbox, score)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            rec_scores_ko = res_ko.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                score = rec_scores_ko[i] if i < len(rec_scores_ko) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox, score)
                                    except Exception:
                                        pass
                    
                    # 중국어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            rec_scores_other = res_other.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                score = rec_scores_other[i] if i < len(rec_scores_other) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ch[bbox_key] = (text, bbox, score)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (중복되는 경우 언어별로 더 적절한 결과 선택)
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ch.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko, score_ko = ocr_results_ko.get(bbox_key, (None, None, 1.0))
                        text_ch, bbox_ch, score_ch = ocr_results_ch.get(bbox_key, (None, None, 1.0))
                        
                        if text_ko and text_ch:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            if is_korean_text(text_ko):
                                # 한국어가 포함된 경우 한국어 OCR 결과 우선
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            elif is_chinese_text(text_ch):
                                # 중국어가 포함된 경우 중국어 OCR 결과 우선
                                all_rec_texts.append(text_ch)
                                all_dt_polys.append(bbox_ch)
                                all_rec_scores.append(score_ch)
                            else:
                                # 둘 다 영어인 경우 한국어 OCR 결과 우선 (더 정확할 가능성)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                            all_rec_scores.append(score_ko)
                        elif text_ch:
                            # 중국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ch)
                            all_dt_polys.append(bbox_ch)
                            all_rec_scores.append(score_ch)
                else:
                    # eng_ar 또는 all 모드: 언어별로 더 적절한 결과 선택
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox, score, index)
                    ocr_results_ar = {}  # bbox_key -> (text, bbox, score, index)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            rec_scores_ko = res_ko.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                score = rec_scores_ko[i] if i < len(rec_scores_ko) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox, score, i)
                                    except Exception:
                                        pass
                    
                    # 아랍어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            rec_scores_other = res_other.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                score = rec_scores_other[i] if i < len(rec_scores_other) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ar[bbox_key] = (text, bbox, score, i)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                    # 같은 bbox에 대해 두 OCR 결과가 모두 있을 때 언어별로 더 적절한 결과 선택
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ar.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko, score_ko, idx_ko = ocr_results_ko.get(bbox_key, (None, None, 1.0, -1))
                        text_ar, bbox_ar, score_ar, idx_ar = ocr_results_ar.get(bbox_key, (None, None, 1.0, -1))
                        
                        if text_ko and text_ar:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                            has_arabic_ar = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ar)
                            has_korean_ko = is_korean_text(text_ko)
                            has_korean_ar = is_korean_text(text_ar)
                            
                            # 숫자 포함 여부 확인
                            has_numbers_ko = any(c.isdigit() for c in text_ko)
                            has_numbers_ar = any(c.isdigit() for c in text_ar)
                            
                            if has_arabic_ar:
                                # 아랍어 OCR 결과에 아랍어가 있는 경우
                                if has_numbers_ar:
                                    # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                                elif has_numbers_ko and not has_numbers_ar:
                                    # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                    # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                    # 예: "10089-CP-104" 같은 전체 문자열 추출
                                    non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                    # 아랍어 OCR에서 아랍어 부분만 추출
                                    arabic_ar = ''.join(c for c in text_ar if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                    # 숫자/기호와 아랍어를 공백으로 합치기
                                    combined_text = f"{non_arabic_ko.strip()} {arabic_ar.strip()}".strip()
                                    all_rec_texts.append(combined_text)
                                    all_dt_polys.append(bbox_ar)  # 아랍어 OCR의 bbox 사용
                                    all_rec_scores.append(score_ar)
                                else:
                                    # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                            elif has_korean_ko and not has_arabic_ko:
                                # 한국어 OCR 결과에 한국어가 있고 아랍어가 없으면 한국어 OCR 결과 우선
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            elif has_arabic_ko and not has_korean_ko:
                                # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            else:
                                # 둘 다 영어/숫자만인 경우 더 긴 텍스트 또는 더 높은 점수 선택
                                if len(text_ar) > len(text_ko) or (len(text_ar) == len(text_ko) and score_ar > score_ko):
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                                else:
                                    all_rec_texts.append(text_ko)
                                    all_dt_polys.append(bbox_ko)
                                    all_rec_scores.append(score_ko)
                        elif text_ar:
                            # 아랍어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ar)
                            all_dt_polys.append(bbox_ar)
                            all_rec_scores.append(score_ar)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                            all_rec_scores.append(score_ko)
                
                result = [{
                    'rec_texts': all_rec_texts,
                    'dt_polys': all_dt_polys,
                    'rec_scores': all_rec_scores
                }] if all_rec_texts else []
            else:
                # 단일 OCR 사용
                result = ocr.predict(input=tmp_path)
            
            # 모든 텍스트와 바운딩 박스 추출
            all_texts = []
            all_bboxes = []
            texts_to_translate = []
            bboxes_to_translate = []
            
            if result and len(result) > 0:
                res = result[0]
                if isinstance(res, dict):
                    rec_texts = res.get('rec_texts', [])
                    dt_polys = res.get('dt_polys', [])
                    rec_scores = res.get('rec_scores', [])
                    
                    for i, text in enumerate(rec_texts):
                        if not text or not text.strip():
                            continue
                        
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        bbox = dt_polys[i] if i < len(dt_polys) else None
                        
                        if bbox is not None and len(bbox) > 0:
                            try:
                                if hasattr(bbox, 'tolist'):
                                    bbox = bbox.tolist()
                                elif not isinstance(bbox, list):
                                    bbox = list(bbox)
                                
                                x_coords = []
                                y_coords = []
                                for point in bbox:
                                    if hasattr(point, '__getitem__') and len(point) >= 2:
                                        try:
                                            x_coords.append(float(point[0]))
                                            y_coords.append(float(point[1]))
                                        except (TypeError, IndexError, ValueError):
                                            continue
                                
                                if x_coords and y_coords:
                                    bbox_rect = [
                                        min(x_coords),
                                        min(y_coords),
                                        max(x_coords),
                                        max(y_coords)
                                    ]
                                    
                                    # 모든 OCR 결과 저장
                                    all_texts.append(text)
                                    all_bboxes.append(bbox_rect)
                                    
                                    # 번역 대상인지 확인
                                    if should_translate_text(text, translation_mode):
                                        # 영어인 경우 신뢰도 체크 없이 모두 번역 (숫자/기호만 제외)
                                        if is_english_text(text):
                                            texts_to_translate.append(text)
                                            bboxes_to_translate.append(bbox_rect)
                                        else:
                                            # 기타 언어(아랍어, 한자 등)는 신뢰도 체크
                                            confidence_threshold = 0.3
                                            if confidence > confidence_threshold:
                                                texts_to_translate.append(text)
                                                bboxes_to_translate.append(bbox_rect)
                            except Exception as e:
                                pass
        
        elif ocr_type == 'easyocr':
            # EasyOCR 사용
            if (translation_mode == 'all' or translation_mode == 'eng_chi' or translation_mode == 'eng_ar') and ocr_ar is not None:
                # 두 OCR 모두 실행
                result_ko = ocr.readtext(tmp_path)  # ['ko', 'en']
                result_other = ocr_ar.readtext(tmp_path)  # ['en', 'ch_sim'] 또는 아랍어 OCR
                
                # 두 결과를 합치기 (언어별로 더 적절한 결과 선택)
                # EasyOCR 결과 형식: [(bbox, text, confidence), ...]
                ocr_results_ko = {}  # bbox_key -> (text, bbox, confidence)
                ocr_results_other = {}  # bbox_key -> (text, bbox, confidence)
                
                # 한국어 OCR 결과 수집
                if result_ko:
                    for item in result_ko:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_ko[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 아랍어/중국어 OCR 결과 수집
                if result_other:
                    for item in result_other:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_other[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                all_results = []
                all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_other.keys())
                for bbox_key in all_bbox_keys:
                    text_ko, bbox_ko, conf_ko = ocr_results_ko.get(bbox_key, (None, None, 0.0))
                    text_other, bbox_other, conf_other = ocr_results_other.get(bbox_key, (None, None, 0.0))
                    
                    if text_ko and text_other:
                        # 중복되는 경우: 언어별로 더 적절한 결과 선택
                        has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                        has_arabic_other = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_other)
                        has_korean_ko = is_korean_text(text_ko)
                        has_korean_other = is_korean_text(text_other)
                        
                        if has_korean_ko and not has_arabic_other:
                            # 한국어 OCR 결과에 한국어가 있고, 아랍어 OCR 결과에 아랍어가 없으면 한국어 OCR 결과 우선
                            # 원본이 한국어면 OCR 인식도 한국어로
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        elif has_arabic_other:
                            # 아랍어 OCR 결과에 아랍어가 있는 경우
                            # 숫자 포함 여부 확인
                            has_numbers_ko = any(c.isdigit() for c in text_ko)
                            has_numbers_other = any(c.isdigit() for c in text_other)
                            
                            if has_numbers_other:
                                # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                all_results.append((bbox_other, text_other, conf_other))
                            elif has_numbers_ko and not has_numbers_other:
                                # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                # 예: "10089-CP-104" 같은 전체 문자열 추출
                                non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                # 아랍어 OCR에서 아랍어 부분만 추출
                                arabic_other = ''.join(c for c in text_other if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                # 숫자/기호와 아랍어를 공백으로 합치기
                                combined_text = f"{non_arabic_ko.strip()} {arabic_other.strip()}".strip()
                                all_results.append((bbox_other, combined_text, conf_other))
                            else:
                                # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                all_results.append((bbox_other, text_other, conf_other))
                        elif has_arabic_ko and not has_korean_ko:
                            # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        else:
                            # 둘 다 영어/숫자만인 경우 더 긴 텍스트 선택
                            if len(text_other) > len(text_ko):
                                all_results.append((bbox_other, text_other, conf_other))
                            else:
                                all_results.append((bbox_ko, text_ko, conf_ko))
                    elif text_other:
                        # 아랍어/중국어 OCR 결과만 있는 경우
                        all_results.append((bbox_other, text_other, conf_other))
                    elif text_ko:
                        # 한국어 OCR 결과만 있는 경우
                        all_results.append((bbox_ko, text_ko, conf_ko))
                
                result = all_results
            else:
                result = ocr.readtext(tmp_path)
            
            # 모든 텍스트와 바운딩 박스 추출
            all_texts = []
            all_bboxes = []
            texts_to_translate = []
            bboxes_to_translate = []
            
            if result:
                for item in result:
                    if len(item) >= 3:
                        bbox_points = item[0]
                        text = item[1]
                        confidence = item[2]
                        
                        if not text or not text.strip():
                            continue
                        
                        try:
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            
                            if x_coords and y_coords:
                                bbox_rect = [
                                    min(x_coords),
                                    min(y_coords),
                                    max(x_coords),
                                    max(y_coords)
                                ]
                                
                                # 모든 OCR 결과 저장
                                all_texts.append(text)
                                all_bboxes.append(bbox_rect)
                                
                                # 번역 대상인지 확인
                                if should_translate_text(text, translation_mode):
                                    # 영어인 경우 신뢰도 체크 없이 모두 번역 (숫자/기호만 제외)
                                    if is_english_text(text):
                                        texts_to_translate.append(text)
                                        bboxes_to_translate.append(bbox_rect)
                                    else:
                                        # 기타 언어(아랍어, 한자 등)는 신뢰도 체크
                                        confidence_threshold = 0.3
                                        if confidence > confidence_threshold:
                                            texts_to_translate.append(text)
                                            bboxes_to_translate.append(bbox_rect)
                        except Exception as e:
                            pass
        else:
            raise ValueError(f"지원하지 않는 OCR 타입: {ocr_type}")
        
        # OCR 결과 이미지 생성
        ocr_img = img.copy()
        if all_texts:
            for text, bbox in zip(all_texts, all_bboxes):
                ocr_img = draw_ocr_text_on_image(ocr_img, text, bbox)
        
        # 번역 결과 이미지 생성
        # 모든 텍스트를 처리: 번역 대상은 번역하고, 번역 대상이 아닌 것은 원문 그대로 표시
        translated_img = img.copy()
        
        # 번역 대상 텍스트와 번역되지 않은 텍스트를 구분
        # translations_dict가 제공되면 재사용, 없으면 번역 수행
        if translations_dict is None:
            translation_dict = {}
            if texts_to_translate and len(texts_to_translate) > 0:
                # 번역 전 텍스트 전처리
                preprocessed_texts = []
                for text in texts_to_translate:
                    if is_english_text(text):
                        preprocessed_texts.append(preprocess_text_for_translation(text))
                    else:
                        preprocessed_texts.append(text)
                
                # 번역 수행
                translations = translate_to_korean(preprocessed_texts, debug=debug)
                
                # 번역 결과 개수 확인 및 조정
                if len(translations) != len(texts_to_translate):
                    if len(translations) < len(texts_to_translate):
                        translations.extend(preprocessed_texts[len(translations):])
                    else:
                        translations = translations[:len(texts_to_translate)]
                
                # 번역 결과를 딕셔너리로 저장 (bbox를 키로 사용)
                for translation, bbox in zip(translations, bboxes_to_translate):
                    bbox_key = tuple(bbox)  # bbox를 튜플로 변환하여 키로 사용
                    translation_dict[bbox_key] = translation
        else:
            # 번역 결과 재사용
            translation_dict = translations_dict
        
        # 모든 텍스트를 순회하면서 번역 결과 이미지에 그리기
        # 번역 대상이면 번역 결과를, 아니면 원문을 그대로 사용
        for text, bbox in zip(all_texts, all_bboxes):
            bbox_key = tuple(bbox)
            if bbox_key in translation_dict:
                # 번역된 텍스트 사용
                translated_img = draw_text_on_image(translated_img, translation_dict[bbox_key], bbox, show_highlight=show_highlight)
            else:
                # 번역되지 않은 텍스트는 원문 그대로 표시 (OCR 결과와 동일)
                translated_img = draw_ocr_text_on_image(translated_img, text, bbox, show_highlight=show_highlight)
        
        # RGB 모드로 변환
        if ocr_img.mode == 'RGBA':
            final_ocr = Image.new('RGB', ocr_img.size, (255, 255, 255))
            final_ocr.paste(ocr_img, mask=ocr_img.split()[3] if len(ocr_img.split()) == 4 else None)
            ocr_img = final_ocr
        
        if translated_img.mode == 'RGBA':
            final_trans = Image.new('RGB', translated_img.size, (255, 255, 255))
            final_trans.paste(translated_img, mask=translated_img.split()[3] if len(translated_img.split()) == 4 else None)
            translated_img = final_trans
        
        if original_img.mode == 'RGBA':
            final_original = Image.new('RGB', original_img.size, (255, 255, 255))
            final_original.paste(original_img, mask=original_img.split()[3] if len(original_img.split()) == 4 else None)
            original_img = final_original
        
        return original_img, ocr_img, translated_img, translation_dict
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_pdf_page(page, ocr, ocr_type='paddle', translation_mode='all', ocr_ar=None, debug=False, page_num=None, show_highlight=False):
    """PDF 페이지를 처리하여 번역된 이미지 반환
    
    Args:
        page: PyMuPDF 페이지 객체
        ocr: OCR 객체 (PaddleOCR 또는 EasyOCR, 주 OCR)
        ocr_type: 'paddle' 또는 'easyocr'
        translation_mode: 번역 모드 ('eng_ar', 'eng_only', 'all')
        ocr_ar: 아랍어 OCR 객체 (eng_ar 모드에서 PaddleOCR 사용 시 필요, 또는 EasyOCR 'all' 모드에서 아랍어용)
        debug: 디버그 모드
        page_num: 페이지 번호 (디버깅용)
    """
    # 페이지를 이미지로 변환 (고해상도)
    mat = fitz.Matrix(3.0, 3.0)  # 3x 확대
    pix = page.get_pixmap(matrix=mat)
    
    # PIL Image로 변환 (메모리 최적화)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    
    # 이미지를 임시 파일로 저장하여 OCR 처리
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    # PyMuPDF로 직접 저장 (속도 최적화)
    pix.save(tmp_path)
    
    try:
        # OCR 엔진에 따라 다른 방식으로 처리
        if ocr_type == 'paddle':
            # PaddleOCR 사용
            # eng_ar 모드, eng_chi 모드, 또는 all 모드이고 ocr_ar가 제공된 경우 두 OCR 모두 사용
            if (translation_mode == 'eng_ar' or translation_mode == 'eng_chi' or translation_mode == 'all') and ocr_ar is not None:
                # 영어와 아랍어/한자 OCR 모두 실행
                result_ko = ocr.predict(input=tmp_path)
                result_other = ocr_ar.predict(input=tmp_path)
                
                # 두 결과를 합치기 (중복 제거를 위해 bbox 기반으로)
                all_rec_texts = []
                all_dt_polys = []
                all_rec_scores = []
                seen_bboxes = set()
                
                # eng_chi 모드에서는 두 OCR 결과를 모두 수집하고 언어별로 더 적절한 결과 선택
                if translation_mode == 'eng_chi':
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox, score)
                    ocr_results_ch = {}  # bbox_key -> (text, bbox, score)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            rec_scores_ko = res_ko.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                score = rec_scores_ko[i] if i < len(rec_scores_ko) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox, score)
                                    except Exception:
                                        pass
                    
                    # 중국어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            rec_scores_other = res_other.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                score = rec_scores_other[i] if i < len(rec_scores_other) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ch[bbox_key] = (text, bbox, score)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (중복되는 경우 언어별로 더 적절한 결과 선택)
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ch.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko, score_ko = ocr_results_ko.get(bbox_key, (None, None, 1.0))
                        text_ch, bbox_ch, score_ch = ocr_results_ch.get(bbox_key, (None, None, 1.0))
                        
                        if text_ko and text_ch:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            if is_korean_text(text_ko):
                                # 한국어가 포함된 경우 한국어 OCR 결과 우선
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            elif is_chinese_text(text_ch):
                                # 중국어가 포함된 경우 중국어 OCR 결과 우선
                                all_rec_texts.append(text_ch)
                                all_dt_polys.append(bbox_ch)
                                all_rec_scores.append(score_ch)
                            else:
                                # 둘 다 영어인 경우 한국어 OCR 결과 우선 (더 정확할 가능성)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                            all_rec_scores.append(score_ko)
                        elif text_ch:
                            # 중국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ch)
                            all_dt_polys.append(bbox_ch)
                            all_rec_scores.append(score_ch)
                else:
                    # eng_ar 또는 all 모드: 언어별로 더 적절한 결과 선택
                    # 두 OCR 결과를 모두 수집 (중복 제거를 위해 bbox 기반으로, 언어별로 더 적절한 결과 선택)
                    ocr_results_ko = {}  # bbox_key -> (text, bbox, score, index)
                    ocr_results_ar = {}  # bbox_key -> (text, bbox, score, index)
                    
                    # 한국어 OCR 결과 수집
                    if result_ko and len(result_ko) > 0:
                        res_ko = result_ko[0]
                        if isinstance(res_ko, dict):
                            rec_texts_ko = res_ko.get('rec_texts', [])
                            dt_polys_ko = res_ko.get('dt_polys', [])
                            rec_scores_ko = res_ko.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_ko):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_ko[i] if i < len(dt_polys_ko) else None
                                score = rec_scores_ko[i] if i < len(rec_scores_ko) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ko[bbox_key] = (text, bbox, score, i)
                                    except Exception:
                                        pass
                    
                    # 아랍어 OCR 결과 수집
                    if result_other and len(result_other) > 0:
                        res_other = result_other[0]
                        if isinstance(res_other, dict):
                            rec_texts_other = res_other.get('rec_texts', [])
                            dt_polys_other = res_other.get('dt_polys', [])
                            rec_scores_other = res_other.get('rec_scores', [])
                            
                            for i, text in enumerate(rec_texts_other):
                                if not text or not text.strip():
                                    continue
                                bbox = dt_polys_other[i] if i < len(dt_polys_other) else None
                                score = rec_scores_other[i] if i < len(rec_scores_other) else 1.0
                                if bbox is not None and len(bbox) > 0:
                                    try:
                                        if hasattr(bbox, 'tolist'):
                                            bbox = bbox.tolist()
                                        x_coords = [float(p[0]) for p in bbox if len(p) >= 2]
                                        y_coords = [float(p[1]) for p in bbox if len(p) >= 2]
                                        if x_coords and y_coords:
                                            bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                        round(max(x_coords), 1), round(max(y_coords), 1))
                                            ocr_results_ar[bbox_key] = (text, bbox, score, i)
                                    except Exception:
                                        pass
                    
                    # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                    # 같은 bbox에 대해 두 OCR 결과가 모두 있을 때 언어별로 더 적절한 결과 선택
                    all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_ar.keys())
                    for bbox_key in all_bbox_keys:
                        text_ko, bbox_ko, score_ko, idx_ko = ocr_results_ko.get(bbox_key, (None, None, 1.0, -1))
                        text_ar, bbox_ar, score_ar, idx_ar = ocr_results_ar.get(bbox_key, (None, None, 1.0, -1))
                        
                        if text_ko and text_ar:
                            # 중복되는 경우: 언어별로 더 적절한 결과 선택
                            has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                            has_arabic_ar = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ar)
                            has_korean_ko = is_korean_text(text_ko)
                            has_korean_ar = is_korean_text(text_ar)
                            
                            # 숫자 포함 여부 확인
                            has_numbers_ko = any(c.isdigit() for c in text_ko)
                            has_numbers_ar = any(c.isdigit() for c in text_ar)
                            
                            if has_arabic_ar:
                                # 아랍어 OCR 결과에 아랍어가 있는 경우
                                if has_numbers_ar:
                                    # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                                elif has_numbers_ko and not has_numbers_ar:
                                    # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                    # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                    # 예: "10089-CP-104" 같은 전체 문자열 추출
                                    non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                    # 아랍어 OCR에서 아랍어 부분만 추출
                                    arabic_ar = ''.join(c for c in text_ar if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                    # 숫자/기호와 아랍어를 공백으로 합치기
                                    combined_text = f"{non_arabic_ko.strip()} {arabic_ar.strip()}".strip()
                                    all_rec_texts.append(combined_text)
                                    all_dt_polys.append(bbox_ar)  # 아랍어 OCR의 bbox 사용
                                    all_rec_scores.append(score_ar)
                                else:
                                    # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                            elif has_korean_ko and not has_arabic_ko:
                                # 한국어 OCR 결과에 한국어가 있고 아랍어가 없으면 한국어 OCR 결과 우선
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            elif has_arabic_ko and not has_korean_ko:
                                # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                                all_rec_texts.append(text_ko)
                                all_dt_polys.append(bbox_ko)
                                all_rec_scores.append(score_ko)
                            else:
                                # 둘 다 영어/숫자만인 경우 더 긴 텍스트 또는 더 높은 점수 선택
                                if len(text_ar) > len(text_ko) or (len(text_ar) == len(text_ko) and score_ar > score_ko):
                                    all_rec_texts.append(text_ar)
                                    all_dt_polys.append(bbox_ar)
                                    all_rec_scores.append(score_ar)
                                else:
                                    all_rec_texts.append(text_ko)
                                    all_dt_polys.append(bbox_ko)
                                    all_rec_scores.append(score_ko)
                        elif text_ar:
                            # 아랍어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ar)
                            all_dt_polys.append(bbox_ar)
                            all_rec_scores.append(score_ar)
                        elif text_ko:
                            # 한국어 OCR 결과만 있는 경우
                            all_rec_texts.append(text_ko)
                            all_dt_polys.append(bbox_ko)
                            all_rec_scores.append(score_ko)
                
                result = [{
                    'rec_texts': all_rec_texts,
                    'dt_polys': all_dt_polys,
                    'rec_scores': all_rec_scores
                }] if all_rec_texts else []
            else:
                # 단일 OCR 사용
                result = ocr.predict(input=tmp_path)
            
            # 모든 텍스트와 번역 대상 텍스트를 분리하여 저장
            all_texts = []
            all_bboxes = []
            texts_to_translate = []
            bboxes_to_translate = []
            
            # 디버깅 정보
            total_texts = 0
            skipped_texts = []
            low_confidence_texts = []
            
            if result and len(result) > 0:
                res = result[0]  # 첫 번째 이미지 결과
                
                if isinstance(res, dict):
                    rec_texts = res.get('rec_texts', [])
                    dt_polys = res.get('dt_polys', [])
                    rec_scores = res.get('rec_scores', [])
                    
                    total_texts = len(rec_texts)
                    
                    # 모든 텍스트와 바운딩 박스를 저장
                    for i, text in enumerate(rec_texts):
                        if not text or not text.strip():
                            continue
                        
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        bbox = dt_polys[i] if i < len(dt_polys) else None
                        
                        if bbox is not None and len(bbox) > 0:
                            try:
                                if hasattr(bbox, 'tolist'):
                                    bbox = bbox.tolist()
                                elif not isinstance(bbox, list):
                                    bbox = list(bbox)
                                
                                x_coords = []
                                y_coords = []
                                for point in bbox:
                                    if hasattr(point, '__getitem__') and len(point) >= 2:
                                        try:
                                            x_coords.append(float(point[0]))
                                            y_coords.append(float(point[1]))
                                        except (TypeError, IndexError, ValueError):
                                            continue
                                
                                if x_coords and y_coords:
                                    bbox_rect = [
                                        min(x_coords),
                                        min(y_coords),
                                        max(x_coords),
                                        max(y_coords)
                                    ]
                                    
                                    # 모든 OCR 결과 저장
                                    all_texts.append(text)
                                    all_bboxes.append(bbox_rect)
                                    
                                    # 번역 대상인지 확인
                                    if should_translate_text(text, translation_mode):
                                        # 영어인 경우 신뢰도 체크 없이 모두 번역 (숫자/기호만 제외)
                                        if is_english_text(text):
                                            texts_to_translate.append(text)
                                            bboxes_to_translate.append(bbox_rect)
                                        else:
                                            # 기타 언어(아랍어, 한자 등)는 신뢰도 체크
                                            confidence_threshold = 0.3
                                            if confidence > confidence_threshold:
                                                texts_to_translate.append(text)
                                                bboxes_to_translate.append(bbox_rect)
                                            else:
                                                # 신뢰도가 낮은 텍스트
                                                low_confidence_texts.append((text, confidence))
                                    else:
                                        # 번역 대상이 아닌 텍스트 (한국어 등 - 그대로 유지)
                                        skipped_texts.append(text)
                            except Exception as e:
                                pass
        
        elif ocr_type == 'easyocr':
            # EasyOCR 사용
            # 'all' 모드 또는 'eng_chi' 모드 또는 'eng_ar' 모드이고 ocr_ar가 제공된 경우 두 OCR 모두 사용
            if (translation_mode == 'all' or translation_mode == 'eng_chi' or translation_mode == 'eng_ar') and ocr_ar is not None:
                # 두 OCR 모두 실행
                result_ko = ocr.readtext(tmp_path)  # ['ko', 'en']
                result_other = ocr_ar.readtext(tmp_path)  # ['en', 'ch_sim'] 또는 ['en', 'ar']
                
                # 두 결과를 합치기 (언어별로 더 적절한 결과 선택)
                # EasyOCR 결과 형식: [(bbox, text, confidence), ...]
                ocr_results_ko = {}  # bbox_key -> (text, bbox, confidence)
                ocr_results_other = {}  # bbox_key -> (text, bbox, confidence)
                
                # 한국어 OCR 결과 수집
                if result_ko:
                    for item in result_ko:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_ko[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 아랍어/중국어 OCR 결과 수집
                if result_other:
                    for item in result_other:
                        if len(item) >= 3:
                            bbox_points = item[0]
                            text = item[1]
                            confidence = item[2]
                            try:
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                if x_coords and y_coords:
                                    bbox_key = (round(min(x_coords), 1), round(min(y_coords), 1), 
                                                round(max(x_coords), 1), round(max(y_coords), 1))
                                    ocr_results_other[bbox_key] = (text, bbox_points, confidence)
                            except Exception:
                                pass
                
                # 두 OCR 결과를 통합 (언어별로 더 적절한 결과 선택, 원본 텍스트 보존)
                all_results = []
                all_bbox_keys = set(ocr_results_ko.keys()) | set(ocr_results_other.keys())
                for bbox_key in all_bbox_keys:
                    text_ko, bbox_ko, conf_ko = ocr_results_ko.get(bbox_key, (None, None, 0.0))
                    text_other, bbox_other, conf_other = ocr_results_other.get(bbox_key, (None, None, 0.0))
                    
                    if text_ko and text_other:
                        # 중복되는 경우: 언어별로 더 적절한 결과 선택
                        has_arabic_ko = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_ko)
                        has_arabic_other = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' for c in text_other)
                        has_korean_ko = is_korean_text(text_ko)
                        has_korean_other = is_korean_text(text_other)
                        
                        if has_korean_ko and not has_arabic_other:
                            # 한국어 OCR 결과에 한국어가 있고, 아랍어 OCR 결과에 아랍어가 없으면 한국어 OCR 결과 우선
                            # 원본이 한국어면 OCR 인식도 한국어로
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        elif has_arabic_other:
                            # 아랍어 OCR 결과에 아랍어가 있는 경우
                            # 숫자 포함 여부 확인
                            has_numbers_ko = any(c.isdigit() for c in text_ko)
                            has_numbers_other = any(c.isdigit() for c in text_other)
                            
                            if has_numbers_other:
                                # 아랍어 OCR 결과에 이미 숫자+아랍어가 모두 있으면 그대로 사용
                                all_results.append((bbox_other, text_other, conf_other))
                            elif has_numbers_ko and not has_numbers_other:
                                # 한국어 OCR 결과에 숫자가 있고, 아랍어 OCR 결과에 아랍어만 있으면 두 결과 합치기
                                # 한국어 OCR에서 아랍어가 아닌 부분 전체 추출 (숫자, 영문자, 기호 포함)
                                # 예: "10089-CP-104" 같은 전체 문자열 추출
                                non_arabic_ko = ''.join(c for c in text_ko if not ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF'))
                                # 아랍어 OCR에서 아랍어 부분만 추출
                                arabic_other = ''.join(c for c in text_other if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF' or c.isspace())
                                # 숫자/기호와 아랍어를 공백으로 합치기
                                combined_text = f"{non_arabic_ko.strip()} {arabic_other.strip()}".strip()
                                all_results.append((bbox_other, combined_text, conf_other))
                            else:
                                # 아랍어 OCR 결과에 아랍어만 있고, 한국어 OCR 결과에 숫자가 없으면 아랍어 OCR 결과 사용
                                all_results.append((bbox_other, text_other, conf_other))
                        elif has_arabic_ko and not has_korean_ko:
                            # 한국어 OCR 결과에 아랍어가 있으면 한국어 OCR 결과 사용 (혼합 텍스트)
                            all_results.append((bbox_ko, text_ko, conf_ko))
                        else:
                            # 둘 다 영어/숫자만인 경우 더 긴 텍스트 선택
                            if len(text_other) > len(text_ko):
                                all_results.append((bbox_other, text_other, conf_other))
                            else:
                                all_results.append((bbox_ko, text_ko, conf_ko))
                    elif text_other:
                        # 아랍어/중국어 OCR 결과만 있는 경우
                        all_results.append((bbox_other, text_other, conf_other))
                    elif text_ko:
                        # 한국어 OCR 결과만 있는 경우
                        all_results.append((bbox_ko, text_ko, conf_ko))
                
                result = all_results
            else:
                # 단일 OCR 사용
                result = ocr.readtext(tmp_path)
            
            # 모든 텍스트와 번역 대상 텍스트를 분리하여 저장
            all_texts = []
            all_bboxes = []
            texts_to_translate = []
            bboxes_to_translate = []
            
            # 디버깅 정보
            total_texts = 0
            skipped_texts = []
            low_confidence_texts = []
            
            if result:
                total_texts = len(result)
                
                # EasyOCR 결과 형식: [(bbox, text, confidence), ...]
                # bbox는 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형태
                for item in result:
                    if len(item) >= 3:
                        bbox_points = item[0]  # 바운딩 박스 좌표
                        text = item[1]  # 인식된 텍스트
                        confidence = item[2]  # 신뢰도
                        
                        if not text or not text.strip():
                            continue
                        
                        try:
                            # 바운딩 박스를 [x_min, y_min, x_max, y_max] 형태로 변환
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            
                            if x_coords and y_coords:
                                bbox_rect = [
                                    min(x_coords),
                                    min(y_coords),
                                    max(x_coords),
                                    max(y_coords)
                                ]
                                
                                # 모든 OCR 결과 저장
                                all_texts.append(text)
                                all_bboxes.append(bbox_rect)
                                
                                # 번역 대상인지 확인
                                if should_translate_text(text, translation_mode):
                                    # 영어인 경우 신뢰도 체크 없이 모두 번역 (숫자/기호만 제외)
                                    if is_english_text(text):
                                        texts_to_translate.append(text)
                                        bboxes_to_translate.append(bbox_rect)
                                    else:
                                        # 기타 언어(아랍어, 한자 등)는 신뢰도 체크
                                        confidence_threshold = 0.3
                                        if confidence > confidence_threshold:
                                            texts_to_translate.append(text)
                                            bboxes_to_translate.append(bbox_rect)
                                        else:
                                            # 신뢰도가 낮은 텍스트
                                            low_confidence_texts.append((text, confidence))
                                else:
                                    # 번역 대상이 아닌 텍스트 (한국어 등 - 그대로 유지)
                                    skipped_texts.append(text)
                        except Exception as e:
                            pass
        else:
            raise ValueError(f"지원하지 않는 OCR 타입: {ocr_type}")
        
        # 디버깅 정보 출력
        if debug and page_num is not None:
            print(f"\n[페이지 {page_num} 디버깅 정보]")
            print(f"총 인식된 텍스트: {total_texts}개")
            print(f"번역 대상 텍스트 (영어/아랍어/한자): {len(texts_to_translate)}개")
            print(f"번역 제외 텍스트 (한국어 등 - 그대로 유지): {len(skipped_texts)}개")
            if skipped_texts:
                print(f"  예시 (한국어는 번역하지 않음): {skipped_texts[:5]}")  # 처음 5개만 표시
            print(f"신뢰도 낮은 텍스트: {len(low_confidence_texts)}개")
            if low_confidence_texts:
                print(f"  예시: {[(t[:20], f'{c:.2f}') for t, c in low_confidence_texts[:5]]}")
        
        # 번역 결과 이미지 생성
        # 모든 텍스트를 처리: 번역 대상은 번역하고, 번역 대상이 아닌 것은 원문 그대로 표시
        translated_img = img.copy()
        translation_dict = {}
        
        if texts_to_translate:
            print(f"\n[번역 대상 확인] 페이지 {page_num if page_num is not None else '?'}: {len(texts_to_translate)}개 텍스트를 번역합니다.")
            # 번역 전 텍스트 전처리 (NO -> Number 등, 영어만 해당)
            preprocessed_texts = []
            for text in texts_to_translate:
                if is_english_text(text):
                    preprocessed_texts.append(preprocess_text_for_translation(text))
                else:
                    preprocessed_texts.append(text)  # 아랍어는 그대로
            
            # 디버깅: 번역 전 텍스트 출력
            if debug and page_num is not None:
                print(f"\n[번역 전 텍스트 예시 (처음 5개)]:")
                for i, text in enumerate(preprocessed_texts[:5]):
                    print(f"  {i+1}. {text}")
            
            # 한국어로 번역
            print(f"\n[번역 함수 호출] {len(preprocessed_texts)}개 텍스트를 번역 함수로 전달")
            translations = translate_to_korean(preprocessed_texts, debug=debug)
            print(f"[번역 함수 반환] {len(translations)}개 번역 결과 받음")
            
            # 디버깅: 번역 후 텍스트 출력
            if debug and page_num is not None:
                print(f"\n[번역 후 텍스트 예시 (처음 5개)]:")
                translated_count = 0
                not_translated_count = 0
                for i, trans in enumerate(translations[:5]):
                    print(f"  {i+1}. {trans}")
                    if i < len(preprocessed_texts):
                        original = preprocessed_texts[i]
                        if trans == original:
                            print(f"      ⚠️ 경고: 번역되지 않음 (원본과 동일)")
                            not_translated_count += 1
                        else:
                            translated_count += 1
                
                # 전체 번역 통계
                total_translated = sum(1 for i, trans in enumerate(translations) 
                                     if i < len(preprocessed_texts) and trans != preprocessed_texts[i])
                total_not_translated = len(translations) - total_translated
                print(f"\n[번역 통계]")
                print(f"  번역됨: {total_translated}개")
                print(f"  번역 안됨: {total_not_translated}개")
            
            # 번역 결과 개수 확인 및 조정
            if len(translations) != len(texts_to_translate):
                if debug and page_num is not None:
                    print(f"\n⚠️ 번역 결과 개수 불일치: 원본 {len(texts_to_translate)}개, 번역 {len(translations)}개")
                if len(translations) < len(texts_to_translate):
                    translations.extend(preprocessed_texts[len(translations):])
                else:
                    translations = translations[:len(texts_to_translate)]
            
            # 번역 결과를 딕셔너리로 저장 (bbox를 키로 사용)
            for translation, bbox in zip(translations, bboxes_to_translate):
                bbox_key = tuple(bbox)  # bbox를 튜플로 변환하여 키로 사용
                translation_dict[bbox_key] = translation
        
        # 모든 텍스트를 순회하면서 번역 결과 이미지에 그리기
        # 번역 대상이면 번역 결과를, 아니면 원문을 그대로 사용
        drawn_count = 0
        for text, bbox in zip(all_texts, all_bboxes):
            bbox_key = tuple(bbox)
            if bbox_key in translation_dict:
                # 번역된 텍스트 사용
                translated_img = draw_text_on_image(translated_img, translation_dict[bbox_key], bbox, show_highlight=show_highlight)
                drawn_count += 1
            else:
                # 번역되지 않은 텍스트는 원문 그대로 표시 (OCR 결과와 동일)
                translated_img = draw_ocr_text_on_image(translated_img, text, bbox, show_highlight=show_highlight)
                drawn_count += 1
        
        if debug and page_num is not None:
            print(f"\n[이미지에 그려진 텍스트]: {drawn_count}개 (번역: {len(translation_dict)}개, 원문: {len(all_texts) - len(translation_dict)}개)")
        elif not texts_to_translate:
            print(f"\n[번역 대상 없음] 페이지 {page_num if page_num is not None else '?'}: 번역할 텍스트가 없습니다. 원문 그대로 표시합니다.")
        
        # RGB 모드로 변환
        if translated_img.mode == 'RGBA':
            final_img = Image.new('RGB', translated_img.size, (255, 255, 255))
            final_img.paste(translated_img, mask=translated_img.split()[3] if len(translated_img.split()) == 4 else None)
            translated_img = final_img
        
        return translated_img
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# OCR 엔진 선택 (환경 변수 또는 명령줄 인자로 설정 가능)
# 'paddle' 또는 'easyocr' 중 선택
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").lower()  # 기본값: paddle

# EasyOCR GPU 사용 여부 (환경 변수 또는 명령줄 인자로 설정 가능)
# 'true', '1', 'yes' 등이면 GPU 사용, 그 외는 CPU 사용
EASYOCR_USE_GPU = os.getenv("EASYOCR_USE_GPU", "false").lower() in ['true', '1', 'yes', 'on']

# 번역 모드 선택 (환경 변수 또는 명령줄 인자로 설정 가능)
# 'eng_only': 영어만 번역
# 'eng_chi': 영어만 번역 (한자는 그대로 표시)
# 'eng_ar': 영어와 아랍어 번역
# 'all': 영어, 아랍어, 한자 모두 번역 (기본값)
TRANSLATION_MODE = os.getenv("TRANSLATION_MODE", "all").lower()  # 기본값: all
VALID_TRANSLATION_MODES = ["eng_only", "eng_chi", "eng_ar", "all"]

# PDF 파일 경로 (기본값, 명령줄 인자로 덮어쓸 수 있음)
PDF_PATH = "2.1.3 현재시공비 지출내역.pdf"

# 번역할 페이지 번호 지정 (1부터 시작, None이면 전체 페이지)
# 예: page_numbers = [1, 3, 5] 또는 page_numbers = [1] 또는 page_numbers = None (전체)
page_numbers = None  # 명령줄 인자로 받거나 여기서 수정

# OCR 결과만 보기 모드 (번역 없이)
OCR_VIEW_MODE = False
# 원본-OCR-번역 결과 모두 보기 모드
FULL_VIEW_MODE = False

# 명령줄 인자 파싱
def parse_arguments():
    """명령줄 인자를 파싱하여 OCR 엔진, OpenAI 모델, 번역 모드, 파일 경로, 페이지 번호를 반환"""
    global OCR_ENGINE, page_numbers, OPENAI_MODEL, TRANSLATION_MODE, PDF_PATH, OCR_VIEW_MODE, FULL_VIEW_MODE, EASYOCR_USE_GPU
    
    ocr_engine = OCR_ENGINE
    model = OPENAI_MODEL
    trans_mode = TRANSLATION_MODE
    file_path = PDF_PATH
    pages = None
    ocr_view = False
    full_view = False
    use_gpu = EASYOCR_USE_GPU
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--ocr-view' or arg == '--ocr-only':
            ocr_view = True
            i += 1
        elif arg == '--full-view' or arg == '--ocr-translate-view':
            full_view = True
            i += 1
        elif arg == '--gpu' or arg == '--use-gpu':
            use_gpu = True
            i += 1
        elif arg == '--no-gpu' or arg == '--cpu':
            use_gpu = False
            i += 1
        elif arg == '--ocr' or arg == '-o':
            if i + 1 < len(sys.argv):
                ocr_engine = sys.argv[i + 1].lower()
                if ocr_engine not in ['paddle', 'easyocr']:
                    print(f"경고: 지원하지 않는 OCR 엔진 '{ocr_engine}'. 'paddle' 또는 'easyocr'만 사용 가능합니다.")
                    print("기본값 'paddle'를 사용합니다.")
                    ocr_engine = 'paddle'
                i += 2
            else:
                print("경고: --ocr 옵션에 값이 없습니다. 기본값을 사용합니다.")
                i += 1
        elif arg == '--model' or arg == '-m':
            if i + 1 < len(sys.argv):
                model = sys.argv[i + 1].lower()
                if model not in VALID_MODELS:
                    print(f"경고: 지원하지 않는 모델 '{model}'. 사용 가능한 모델: {', '.join(VALID_MODELS)}")
                    print(f"기본값 '{OPENAI_MODEL}'를 사용합니다.")
                    model = OPENAI_MODEL
                i += 2
            else:
                print("경고: --model 옵션에 값이 없습니다. 기본값을 사용합니다.")
                i += 1
        elif arg == '--mode' or arg == '--translation-mode':
            if i + 1 < len(sys.argv):
                trans_mode = sys.argv[i + 1].lower()
                if trans_mode not in VALID_TRANSLATION_MODES:
                    print(f"경고: 지원하지 않는 번역 모드 '{trans_mode}'. 사용 가능한 모드: {', '.join(VALID_TRANSLATION_MODES)}")
                    print(f"기본값 '{TRANSLATION_MODE}'를 사용합니다.")
                    trans_mode = TRANSLATION_MODE
                i += 2
            else:
                print("경고: --mode 옵션에 값이 없습니다. 기본값을 사용합니다.")
                i += 1
        elif arg == '--file' or arg == '-f':
            if i + 1 < len(sys.argv):
                # 파일 경로가 .pdf로 끝날 때까지 다음 인자들을 합치기 (공백 포함 파일명 처리)
                file_path_parts = [sys.argv[i + 1]]
                j = i + 2
                # 현재까지 합친 경로가 .pdf로 끝나지 않으면 다음 인자들을 계속 합치기
                while j < len(sys.argv) and not ' '.join(file_path_parts).lower().endswith('.pdf'):
                    file_path_parts.append(sys.argv[j])
                    j += 1
                file_path = ' '.join(file_path_parts)
                i = j
            else:
                print("경고: --file 옵션에 값이 없습니다. 기본값을 사용합니다.")
                i += 1
        elif arg == '--help' or arg == '-h':
            print("사용법:")
            print("  python final.py [옵션] [페이지번호]")
            print("\n옵션:")
            print("  --ocr-view, --ocr-only        OCR 결과만 확인 (번역 없음)")
            print("                                 원본 이미지와 OCR 결과 이미지를 나란히 붙여서 PNG로 저장")
            print("  --full-view, --ocr-translate-view  원본-OCR-번역 결과 모두 확인")
            print("                                 원본 이미지, OCR 결과 이미지, 번역 결과 이미지를 나란히 붙여서 PNG로 저장")
            print("  --gpu, --use-gpu              EasyOCR에서 GPU 사용 (CUDA 자동 감지 시 기본값)")
            print("  --no-gpu, --cpu               EasyOCR에서 CPU 사용 (CUDA가 없거나 명시적으로 지정 시)")
            print("                                 CUDA가 사용 가능하면 자동으로 GPU 모드로 전환됩니다")
            print("  --ocr, -o <paddle|easyocr>    OCR 엔진 선택 (기본값: paddle)")
            print("  --model, -m <모델명>           OpenAI 모델 선택")
            print("                                 사용 가능한 모델:")
            print("                                 - gpt-3.5-turbo (빠르고 저렴)")
            print("                                 - gpt-4 (정확하지만 느리고 비쌈)")
            print("                                 - gpt-4-turbo (GPT-4 Turbo)")
            print("                                 - gpt-4o (최신 GPT-4)")
            print("                                 - gpt-4o-mini (경량 GPT-4, 기본값)")
            print("                                 - gpt-5 (베타, 접근 권한 필요할 수 있음)")
            print("  --mode, --translation-mode     번역 모드 선택")
            print("                                 - eng_only: 영어만 번역")
            print("                                 - eng_chi: 영어만 번역 (한자는 그대로 표시)")
            print("                                 - eng_ar: 영어와 아랍어 번역")
            print("                                 - all: 영어, 아랍어, 한자 모두 번역 (기본값)")
            print("  --file, -f <파일경로>          PDF 파일 경로 지정 (공백 포함 가능)")
            print("                                 기본값: 2.1.3 현재시공비 지출내역.pdf")
            print("  --help, -h                     도움말 표시")
            print("\n사용 예시:")
            print("  python final.py --ocr-view  # OCR 결과만 확인 (번역 없음)")
            print("  python final.py --ocr-view --file document.pdf 1  # 특정 페이지 OCR 결과 확인")
            print("  python final.py --full-view  # 원본-OCR-번역 결과 모두 확인")
            print("  python final.py --full-view --file document.pdf 1  # 특정 페이지 원본-OCR-번역 결과 확인")
            print("  python final.py --file document.pdf  # PDF 파일 지정")
            print("  python final.py --file \"2.2.4 의료비 지출내역.pdf\"  # 공백 포함 파일명 (따옴표 권장)")
            print("  python final.py --file 2.2.4 의료비 지출내역.pdf  # 공백 포함 파일명 (자동 처리)")
            print("  python final.py --file document.pdf 1  # 파일과 페이지 번호 지정")
            print("  python final.py 1              # 단일 페이지 (기본 파일 사용)")
            print("  python final.py 1,3,5          # 여러 페이지")
            print("  python final.py 1-10           # 연속 페이지 범위")
            print("  python final.py --ocr easyocr  # EasyOCR 사용")
            print("  python final.py --ocr easyocr --gpu  # EasyOCR GPU 사용")
            print("  python final.py --model gpt-4  # GPT-4 모델 사용")
            print("  python final.py --mode eng_only  # 영어만 번역")
            print("  python final.py --mode eng_chi  # 영어만 번역 (한자는 그대로 표시)")
            print("  python final.py --mode eng_ar  # 영어와 아랍어 번역")
            print("  python final.py --mode all  # 모든 언어 번역")
            print("  python final.py --file doc.pdf --ocr paddle --model gpt-3.5-turbo --mode eng_chi 1  # 모든 옵션 조합")
            exit(0)
        elif arg.startswith('-'):
            # 다른 옵션은 무시
            i += 1
        else:
            # 파일 경로 또는 페이지 번호로 간주
            # 먼저 파일 경로인지 확인 (확장자가 .pdf인 경우)
            if arg.lower().endswith('.pdf'):
                # PDF 파일 경로로 간주
                file_path = arg
                i += 1
            else:
                # 페이지 번호로 간주
                try:
                    pages = parse_page_numbers(arg)
                except ValueError as e:
                    print(f"페이지 번호 형식이 올바르지 않습니다: {e}")
                    print("예시:")
                    print("  python final.py --file document.pdf  # PDF 파일 지정")
                    print("  python final.py 1              # 단일 페이지")
                    print("  python final.py 1,3,5          # 여러 페이지")
                    print("  python final.py 1-10           # 연속 페이지 범위")
                    print("  python final.py --ocr easyocr  # EasyOCR 사용")
                    print("  python final.py --model gpt-4  # GPT-4 모델 사용")
                    print("  python final.py --ocr paddle 1 # PaddleOCR 사용, 1페이지")
                    exit(1)
                i += 1
    
    return ocr_engine, model, trans_mode, file_path, pages, ocr_view, full_view, use_gpu

# 명령줄 인자로 페이지 번호 받기
def parse_page_numbers(page_str):
    """페이지 번호 문자열을 파싱하여 리스트로 반환
    예: "1" -> [1]
        "1,3,5" -> [1, 3, 5]
        "1-10" -> [1, 2, 3, ..., 10]
        "1,3,5-8" -> [1, 3, 5, 6, 7, 8]
        "1-5,10-15" -> [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    """
    page_numbers = []
    parts = page_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 범위 처리 (예: "1-10")
            range_parts = part.split('-')
            if len(range_parts) == 2:
                try:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    if start > end:
                        raise ValueError(f"범위 시작({start})이 끝({end})보다 큽니다.")
                    page_numbers.extend(range(start, end + 1))
                except ValueError as e:
                    raise ValueError(f"범위 파싱 오류: {part} - {e}")
            else:
                raise ValueError(f"범위 형식이 올바르지 않습니다: {part}")
        else:
            # 단일 숫자 처리
            try:
                page_numbers.append(int(part))
            except ValueError:
                raise ValueError(f"페이지 번호가 올바르지 않습니다: {part}")
    
    # 중복 제거 및 정렬
    page_numbers = sorted(list(set(page_numbers)))
    return page_numbers

# 명령줄 인자 파싱
OCR_ENGINE, OPENAI_MODEL, TRANSLATION_MODE, pdf_path, page_numbers, OCR_VIEW_MODE, FULL_VIEW_MODE, EASYOCR_USE_GPU = parse_arguments()

# OCR 엔진 확인
if OCR_ENGINE == 'paddle' and not HAS_PADDLEOCR:
    print("PaddleOCR이 설치되지 않았습니다. 설치: pip install paddleocr")
    print("또는 EasyOCR을 사용하세요: python final.py --ocr easyocr")
    exit(1)

if OCR_ENGINE == 'easyocr' and not HAS_EASYOCR:
    print("EasyOCR이 설치되지 않았습니다. 설치: pip install easyocr")
    print("또는 PaddleOCR을 사용하세요: python final.py --ocr paddle")
    exit(1)

if not HAS_PYMUPDF:
    print("PyMuPDF가 필요합니다. 설치: pip install pymupdf")
    exit(1)

# CUDA 사용 가능 여부 확인 (EasyOCR 사용 시)
if OCR_ENGINE == 'easyocr':
    cuda_available, device_count, device_name = check_cuda_available()
    if cuda_available is True:
        print(f"\n[CUDA 확인] GPU 사용 가능: {device_count}개 디바이스 감지")
        if device_name:
            print(f"  → GPU: {device_name}")
        # 사용자가 명시적으로 --no-gpu 또는 --cpu를 지정하지 않은 경우 자동으로 GPU 사용
        # (단, --gpu 또는 --use-gpu를 명시적으로 지정한 경우는 이미 EASYOCR_USE_GPU가 True)
        if EASYOCR_USE_GPU is False and '--no-gpu' not in sys.argv and '--cpu' not in sys.argv:
            print("  → GPU가 감지되었습니다. 자동으로 GPU 모드를 활성화합니다.")
            EASYOCR_USE_GPU = True
        elif EASYOCR_USE_GPU is True:
            print("  → GPU 모드가 명시적으로 요청되었습니다.")
    elif cuda_available is False:
        print(f"\n[CUDA 확인] GPU 사용 불가능 (CPU 모드 사용)")
        if EASYOCR_USE_GPU:
            print("  → 경고: GPU 모드가 요청되었지만 CUDA를 사용할 수 없습니다. CPU 모드로 전환합니다.")
            EASYOCR_USE_GPU = False
    else:
        print(f"\n[CUDA 확인] CUDA 상태 확인 불가 (PyTorch 미설치 또는 기타 이유)")
        if EASYOCR_USE_GPU:
            print("  → GPU 모드로 시도합니다. 실패 시 자동으로 CPU 모드로 전환됩니다.")

if not os.path.exists(pdf_path):
    print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    exit(1)

# OCR 엔진 및 모델 정보 출력
print(f"PDF 파일: {pdf_path}")
print(f"OCR 엔진: {OCR_ENGINE.upper()}")
if FULL_VIEW_MODE:
    print("모드: 원본-OCR-번역 결과 확인")
    print("  → 원본 이미지, OCR 결과 이미지, 번역 결과 이미지를 나란히 붙여서 저장합니다.")
    print(f"OpenAI 모델: {OPENAI_MODEL}")
    print(f"번역 모드: {TRANSLATION_MODE.upper()}")
elif OCR_VIEW_MODE:
    print("모드: OCR 결과 확인 (번역 없음)")
    print("  → 원본 이미지와 OCR 결과 이미지를 나란히 붙여서 저장합니다.")
else:
    print(f"OpenAI 모델: {OPENAI_MODEL}")
    print(f"번역 모드: {TRANSLATION_MODE.upper()}")
    if TRANSLATION_MODE == 'eng_only':
        print("  → 영어만 한국어로 번역합니다.")
    elif TRANSLATION_MODE == 'eng_chi':
        print("  → 영어만 한국어로 번역합니다. (한자는 그대로 표시)")
    elif TRANSLATION_MODE == 'eng_ar':
        print("  → 영어와 아랍어를 한국어로 번역합니다.")
    else:
        print("  → 영어, 아랍어, 한자를 한국어로 번역합니다.")

# OCR 엔진 초기화 (OCR 전용 모드 또는 번역 모드에 따라 언어 설정)
if FULL_VIEW_MODE or OCR_VIEW_MODE:
    # OCR 전용 모드: 번역 모드에 따라 언어 설정
    if OCR_ENGINE == 'paddle':
        print("PaddleOCR 초기화 중...")
        if TRANSLATION_MODE == 'eng_only':
            # eng_only 모드: 한국어 OCR만 사용
            print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식)")
            ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='korean'  # 한국어 모드 (한국어, 영어 인식)
            )
            ocr_ar = None  # 중국어 OCR 사용 안 함
        else:
            # eng_chi, eng_ar, all 모드: 모든 언어 인식 가능하도록 설정
            print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식)")
            ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='korean'  # 한국어 모드 (한국어, 영어 인식)
            )
            print("  → 중국어 OCR 모델 사용 (한자, 영어 인식)")
            ocr_ch = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='ch'  # 중국어 모드 (한자, 영어 인식 가능)
            )
            print("  → 아랍어 OCR 모델 사용")
            ocr_ar = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='ar'  # 아랍어 모드
            )
    elif OCR_ENGINE == 'easyocr':
        gpu_status = "GPU" if EASYOCR_USE_GPU else "CPU"
        print(f"EasyOCR 초기화 중... ({gpu_status} 사용, 처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
        if TRANSLATION_MODE == 'eng_only':
            # eng_only 모드: 한국어 OCR만 사용
            print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
            ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어
            ocr_ar = None  # 중국어 OCR 사용 안 함
        else:
            # eng_chi, eng_ar, all 모드: 모든 언어 인식 가능하도록 설정
            if TRANSLATION_MODE == 'all':
                # all 모드: 한국어, 영어, 아랍어, 한자 모두 인식
                print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
                ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어
                print("  → 영어, 한자 OCR 모델 사용 (['en', 'ch_sim'])")
                ocr_ch = easyocr.Reader(['en', 'ch_sim'], gpu=EASYOCR_USE_GPU)  # 영어, 중국어 간체
                print("  → 영어, 아랍어 OCR 모델 사용 (['en', 'ar'])")
                ocr_ar = easyocr.Reader(['en', 'ar'], gpu=EASYOCR_USE_GPU)  # 영어, 아랍어
            elif TRANSLATION_MODE == 'eng_ar':
                # eng_ar 모드: 한국어, 영어, 아랍어 인식
                print("  → 한국어, 영어, 아랍어 OCR 모델 사용 (['ko', 'en', 'ar'])")
                ocr = easyocr.Reader(['ko', 'en', 'ar'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어, 아랍어
                ocr_ar = None
            else:
                # eng_chi 모드: 한국어, 영어, 한자 인식
                print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
                ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어
                print("  → 영어, 한자 OCR 모델 사용 (['en', 'ch_sim'])")
                ocr_ar = easyocr.Reader(['en', 'ch_sim'], gpu=EASYOCR_USE_GPU)  # 영어, 중국어 간체
    else:
        print(f"지원하지 않는 OCR 엔진: {OCR_ENGINE}")
        exit(1)
elif OCR_ENGINE == 'paddle':
    print("PaddleOCR 초기화 중...")
    # 번역 모드에 따라 언어 설정
    if TRANSLATION_MODE == 'eng_only':
        # 영어만 번역: 한국어 OCR 사용 (한국어도 인식 가능하도록)
        print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식 가능)")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='korean'  # 한국어 모드 (한국어, 영어 인식 가능)
        )
        ocr_ar = None  # 아랍어 OCR이 필요 없는 경우
    elif TRANSLATION_MODE == 'eng_chi':
        # 영어만 번역 (한자는 그대로 표시): 한국어 OCR과 중국어 OCR 모두 사용
        print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식)")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='korean'  # 한국어 모드 (한국어, 영어 인식)
        )
        print("  → 중국어 OCR 모델 사용 (한자, 영어 인식)")
        ocr_ar = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ch'  # 중국어 모드 (한자, 영어 인식 가능)
        )
    elif TRANSLATION_MODE == 'eng_ar':
        # 영어와 아랍어 번역: 한국어 OCR 사용 (한국어도 인식 가능하도록)
        print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식 가능)")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='korean'  # 한국어 모드 (한국어, 영어 인식 가능)
        )
        print("  → 아랍어 OCR 모델 사용")
        ocr_ar = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ar'  # 아랍어 모드
        )
    else:
        # 모든 언어 번역: 한국어 OCR, 중국어 OCR, 아랍어 OCR 모두 사용
        print("  → 한국어 OCR 모델 사용 (한국어, 영어 인식)")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='korean'  # 한국어 모드 (한국어, 영어 인식)
        )
        print("  → 중국어 OCR 모델 사용 (한자, 영어 인식)")
        ocr_ch = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ch'  # 중국어 모드 (한자, 영어 인식 가능)
        )
        print("  → 아랍어 OCR 모델 사용")
        ocr_ar = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ar'  # 아랍어 모드
        )
        # all 모드에서는 ocr_ar를 아랍어 OCR로 사용하고, ocr_ch를 중국어 OCR로 사용
        # ocr_ar 변수에 아랍어 OCR 저장 (기존 코드와의 호환성을 위해)
elif OCR_ENGINE == 'easyocr':
    gpu_status = "GPU" if EASYOCR_USE_GPU else "CPU"
    print(f"EasyOCR 초기화 중... ({gpu_status} 사용, 처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
    # 번역 모드에 따라 언어 설정
    if TRANSLATION_MODE == 'eng_only':
        # 영어만 번역: 한국어, 영어 (한국어도 인식 가능하도록)
        print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
        ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)
        ocr_ar = None
    elif TRANSLATION_MODE == 'eng_chi':
        # 영어만 번역 (한자는 그대로 표시): 한국어+영어와 중국어+영어를 별도 Reader로 사용
        # EasyOCR은 'ko'와 'ch_sim'을 동시에 사용할 수 없으므로 두 개의 Reader 생성
        print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
        ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어
        print("  → 영어, 한자 OCR 모델 사용 (['en', 'ch_sim'])")
        ocr_ar = easyocr.Reader(['en', 'ch_sim'], gpu=EASYOCR_USE_GPU)  # 영어, 중국어 간체
    elif TRANSLATION_MODE == 'eng_ar':
        # 영어와 아랍어 번역: 한국어, 영어, 아랍어 (한국어도 인식 가능하도록)
        print("  → 한국어, 영어, 아랍어 OCR 모델 사용 (['ko', 'en', 'ar'])")
        ocr = easyocr.Reader(['ko', 'en', 'ar'], gpu=EASYOCR_USE_GPU)
        ocr_ar = None
    else:
        # 모든 언어 번역: 한국어, 영어, 아랍어, 한자
        # EasyOCR은 'ko'와 'ch_sim', 'ar'와 'ch_sim'을 동시에 사용할 수 없으므로 여러 Reader 생성
        print("  → 한국어, 영어 OCR 모델 사용 (['ko', 'en'])")
        ocr = easyocr.Reader(['ko', 'en'], gpu=EASYOCR_USE_GPU)  # 한국어, 영어
        print("  → 영어, 한자 OCR 모델 사용 (['en', 'ch_sim'])")
        ocr_ch = easyocr.Reader(['en', 'ch_sim'], gpu=EASYOCR_USE_GPU)  # 영어, 중국어 간체
        print("  → 영어, 아랍어 OCR 모델 사용 (['en', 'ar'])")
        ocr_ar = easyocr.Reader(['en', 'ar'], gpu=EASYOCR_USE_GPU)  # 영어, 아랍어
        # all 모드에서는 ocr_ar를 아랍어 OCR로 사용하고, ocr_ch를 중국어 OCR로 사용
else:
    print(f"지원하지 않는 OCR 엔진: {OCR_ENGINE}")
    exit(1)

# PDF 열기
doc = fitz.open(pdf_path)
total_pages = len(doc)

# 처리할 페이지 번호 결정 (0-based index)
if page_numbers is None:
    # 전체 페이지 처리
    pages_to_process = list(range(total_pages))
    print(f"전체 {total_pages}페이지를 처리합니다.")
else:
    # 특정 페이지만 처리 (1-based를 0-based로 변환)
    pages_to_process = []
    for page_num in page_numbers:
        if 1 <= page_num <= total_pages:
            pages_to_process.append(page_num - 1)  # 0-based index
        else:
            print(f"경고: 페이지 {page_num}는 존재하지 않습니다. (총 {total_pages}페이지)")
    
    if not pages_to_process:
        print("처리할 페이지가 없습니다.")
        exit(1)
    
    print(f"페이지 {', '.join(map(str, page_numbers))}를 처리합니다. (총 {len(pages_to_process)}페이지)")

if OCR_VIEW_MODE:
    # OCR 전용 모드: 원본과 OCR 결과 이미지를 나란히 붙여서 이미지 파일로 저장
    print("\n[OCR 결과 확인 모드] 원본 이미지와 OCR 결과 이미지를 나란히 붙여서 저장합니다.")
    
    # results 폴더 생성
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"  → '{results_dir}' 폴더를 생성했습니다.")
    
    # tqdm으로 진행 상황 표시
    with tqdm(total=len(pages_to_process), desc="페이지 처리", unit="페이지") as pbar:
        for idx, page_num in enumerate(pages_to_process):
            page = doc[page_num]
            display_page_num = page_num + 1  # 1-based for display
            pbar.set_description(f"페이지 {display_page_num}/{total_pages} 처리 중")
            
            # 페이지 처리 (OCR만 수행)
            pbar.set_postfix({"단계": "OCR 수행 중"})
            original_img, ocr_img = process_pdf_page_ocr_only(page, ocr, ocr_type=OCR_ENGINE, ocr_ar=ocr_ar, debug=False, page_num=display_page_num)
            
            # 두 이미지를 나란히 붙이기 (왼쪽: 원본, 오른쪽: OCR 결과)
            pbar.set_postfix({"단계": "이미지 결합 중"})
            # OCR 이미지에 태그 추가
            ocr_img_tagged = add_color_tags_to_image(ocr_img, has_translation=False, has_ocr=True)
            combined_img = combine_images_side_by_side(original_img, ocr_img_tagged)
            
            # 이미지 파일로 저장 (results 폴더에 저장)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            if page_numbers is None:
                output_img_path = os.path.join(results_dir, f"ocr_result_{OCR_ENGINE}_{base_name}_page_{display_page_num}.png")
            else:
                pages_str = "_".join(map(str, page_numbers))
                output_img_path = os.path.join(results_dir, f"ocr_result_{OCR_ENGINE}_{base_name}_page_{display_page_num}.png")
            
            # 기존 파일이 있으면 삭제
            if os.path.exists(output_img_path):
                try:
                    os.remove(output_img_path)
                except:
                    pass
            
            combined_img.save(output_img_path, 'PNG')
            print(f"  → 저장됨: {output_img_path}")
            
            pbar.update(1)
    
    doc.close()
    print(f"\nOCR 결과 이미지가 저장되었습니다. (폴더: {results_dir})")
elif FULL_VIEW_MODE:
    # 원본-OCR-번역 결과 모두 보기 모드: 세 이미지를 나란히 붙여서 이미지 파일로 저장
    print("\n[원본-OCR-번역 결과 확인 모드] 원본 이미지, OCR 결과 이미지, 번역 결과 이미지를 나란히 붙여서 저장합니다.")
    
    # results 폴더 생성
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"  → '{results_dir}' 폴더를 생성했습니다.")
    
    # PDF 저장을 위한 문서 생성
    output_doc = fitz.open()
    
    # tqdm으로 진행 상황 표시
    with tqdm(total=len(pages_to_process), desc="페이지 처리", unit="페이지") as pbar:
        for idx, page_num in enumerate(pages_to_process):
            page = doc[page_num]
            display_page_num = page_num + 1  # 1-based for display
            pbar.set_description(f"페이지 {display_page_num}/{total_pages} 처리 중")
            
            # 페이지 처리 (OCR + 번역 수행) - 번역은 한 번만 수행
            pbar.set_postfix({"단계": "OCR 및 번역 수행 중"})
            original_img, ocr_img, translated_img, translation_dict = process_pdf_page_full_view(
                page, ocr, ocr_type=OCR_ENGINE, translation_mode=TRANSLATION_MODE, 
                ocr_ar=ocr_ar, debug=False, page_num=display_page_num, show_highlight=True
            )
            
            # PDF 저장용 이미지 생성 (색깔 없이) - 번역 결과 재사용
            pbar.set_postfix({"단계": "PDF 이미지 생성 중"})
            _, _, translated_img_for_pdf, _ = process_pdf_page_full_view(
                page, ocr, ocr_type=OCR_ENGINE, translation_mode=TRANSLATION_MODE, 
                ocr_ar=ocr_ar, debug=False, page_num=display_page_num, show_highlight=False,
                translations_dict=translation_dict  # 번역 결과 재사용
            )
            
            # 세 이미지를 나란히 붙이기 (왼쪽: 원본, 중간: OCR 결과, 오른쪽: 번역 결과)
            pbar.set_postfix({"단계": "이미지 결합 중"})
            combined_img, img1_width, img2_width, img3_width = combine_three_images_side_by_side(original_img, ocr_img, translated_img)
            # 결합된 이미지에 제목 추가
            combined_img = add_titles_to_combined_image(combined_img, img1_width, img2_width, img3_width)
            # 결합된 이미지 상단 오른쪽에 두 태그 모두 추가
            combined_img = add_color_tags_to_image(combined_img, has_translation=True, has_ocr=True)
            
            # 이미지 파일로 저장 (results 폴더에 저장)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            if page_numbers is None:
                output_img_path = os.path.join(results_dir, f"full_result_{OCR_ENGINE}_{base_name}_page_{display_page_num}.png")
            else:
                pages_str = "_".join(map(str, page_numbers))
                output_img_path = os.path.join(results_dir, f"full_result_{OCR_ENGINE}_{base_name}_page_{display_page_num}.png")
            
            # 기존 파일이 있으면 삭제
            if os.path.exists(output_img_path):
                try:
                    os.remove(output_img_path)
                except:
                    pass
            
            combined_img.save(output_img_path, 'PNG')
            print(f"  → 이미지 저장됨: {output_img_path}")
            
            # PDF에도 번역된 이미지 추가 (번역 결과만 PDF에 저장, 색깔 없이)
            pbar.set_postfix({"단계": "PDF 저장 중"})
            img_bytes = io.BytesIO()
            translated_img_for_pdf.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # 새 페이지 생성
            new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes.getvalue())
            
            pbar.update(1)
    
    # PDF 저장 (test.py와 동일한 방식)
    if page_numbers is None:
        output_path = "translated_" + os.path.basename(pdf_path)
    else:
        # 특정 페이지인 경우 페이지 번호 포함
        pages_str = "_".join(map(str, page_numbers))
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ext = os.path.splitext(pdf_path)[1]
        output_path = f"translated_{base_name}_page_{pages_str}{ext}"

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_new{ext}"

    output_doc.save(output_path)
    output_doc.close()
    doc.close()
    
    print(f"\n원본-OCR-번역 결과 이미지가 저장되었습니다. (폴더: {results_dir})")
    print(f"번역된 PDF가 저장되었습니다: {output_path}")
else:
    # 번역 모드: 기존 로직 사용
    # 새로운 PDF 생성
    output_doc = fitz.open()

    # 디버그 모드 활성화 여부 (10페이지 처리 시 자동으로 활성화)
    enable_debug = True  # 10페이지 디버깅을 위해 활성화

    # tqdm으로 진행 상황 표시
    with tqdm(total=len(pages_to_process), desc="페이지 처리", unit="페이지") as pbar:
        for idx, page_num in enumerate(pages_to_process):
            page = doc[page_num]
            display_page_num = page_num + 1  # 1-based for display
            pbar.set_description(f"페이지 {display_page_num}/{total_pages} 처리 중")
            
            # 페이지 처리 (10페이지는 항상 디버그 모드로)
            is_debug = enable_debug and display_page_num == 10
            pbar.set_postfix({"단계": "OCR 및 번역 중"})
            translated_img = process_pdf_page(page, ocr, ocr_type=OCR_ENGINE, translation_mode=TRANSLATION_MODE, ocr_ar=ocr_ar, debug=is_debug, page_num=display_page_num)
            
            # 이미지를 PDF 페이지로 변환
            pbar.set_postfix({"단계": "PDF 저장 중"})
            img_bytes = io.BytesIO()
            translated_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # 새 페이지 생성
            new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes.getvalue())
            
            pbar.update(1)

    # 결과 PDF 저장 (test.py와 동일한 방식)
    if page_numbers is None:
        output_path = "translated_" + os.path.basename(pdf_path)
    else:
        # 특정 페이지인 경우 페이지 번호 포함
        pages_str = "_".join(map(str, page_numbers))
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        ext = os.path.splitext(pdf_path)[1]
        output_path = f"translated_{base_name}_page_{pages_str}{ext}"

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_new{ext}"

    output_doc.save(output_path)
    output_doc.close()
    doc.close()

    print(f"\n번역된 PDF가 저장되었습니다: {output_path}")
    print(f"\n번역된 PDF가 저장되었습니다: {output_path}")