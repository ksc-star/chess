import os
import subprocess
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from openai import OpenAI

from fastapi.responses import RedirectResponse


# ============================================================
#  환경 변수 설정
# ============================================================

# Render 환경 변수에 OPENAI_API_KEY 넣어두었다고 가정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "./stockfish")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        "Render 대시보드 → Environment에서 OPENAI_API_KEY를 설정하세요."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
#  Stockfish 엔진 래퍼 클래스
# ============================================================

class StockfishEngine:
    """
    - Stockfish UCI 엔진을 subprocess로 실행
    - FEN을 넣어서 분석 결과를 받아오는 간단한 래퍼
    """

    def __init__(self, path: str, depth: int = 15):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stockfish 바이너리를 찾을 수 없습니다: {path}")

        self.path = path
        self.depth = depth

        # Stockfish 프로세스 실행
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # UCI 초기화
        self._send_command("uci")
        self._wait_for_token("uciok")
        self._send_command("isready")
        self._wait_for_token("readyok")

    # ---------- 내부 유틸 ----------

    def _send_command(self, cmd: str) -> None:
        """Stockfish 프로세스에 UCI 명령을 보냄."""
        if self.proc.stdin is None:
            raise RuntimeError("Stockfish stdin이 초기화되어 있지 않습니다.")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for_token(self, token: str) -> None:
        """stdout에서 특정 문자열이 나올 때까지 읽음."""
        if self.proc.stdout is None:
            raise RuntimeError("Stockfish stdout이 초기화되어 있지 않습니다.")
        for line in self.proc.stdout:
            line = line.strip()
            if token in line:
                break

    # ---------- 핵심 메서드 ----------

    def analyze_fen(self, fen: str, multi_pv: int = 1):
        """
        FEN을 입력 받아 Stockfish로 분석하고,
        - bestmove
        - eval_cp / eval_type
        - pv_moves
        를 dict로 리턴.
        """
        if self.proc.stdout is None:
            raise RuntimeError("Stockfish stdout이 초기화되어 있지 않습니다.")

        # 포지션 설정
        self._send_command(f"position fen {fen}")

        # MultiPV 설정 (여러 최선 수를 보고 싶을 때)
        self._send_command(f"setoption name MultiPV value {multi_pv}")

        # 분석 시작
        self._send_command(f"go depth {self.depth}")

        bestmove = None
        eval_cp = None
        eval_type = None
        pv_moves: List[str] = []

        # UCI 프로토콜 출력 파싱
        for line in self.proc.stdout:
            line = line.strip()

            # 점수 정보
            if line.startswith("info") and "score" in line:
                parts = line.split()
                if "score" in parts:
                    idx = parts.index("score")
                    if idx + 2 < len(parts):
                        eval_type = parts[idx + 1]  # "cp" or "mate"
                        eval_value_str = parts[idx + 2]
                        try:
                            eval_cp = int(eval_value_str)
                        except ValueError:
                            pass

                # PV(최선 수열) 추출
                if " pv " in line:
                    # " ... pv e2e4 e7e5 ..." 형태라서 " pv " 위치 기준 split
                    pv_index = line.index(" pv ")
                    pv_part = line[pv_index + 4 :].strip()
                    pv_moves = pv_part.split()

            # bestmove 정보
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break

        if bestmove is None:
            raise RuntimeError("Stockfish가 bestmove를 리턴하지 않았습니다.")

        return {
            "bestmove": bestmove,
            "eval_cp": eval_cp,
            "eval_type": eval_type,
            "pv_moves": pv_moves,
        }


# ============================================================
#  FastAPI 앱 & 스키마
# ============================================================

app = FastAPI(title="GPT + Stockfish Chess Tutor")

@app.get("/", include_in_schema=False)
def root():
    # / 로 들어오면 자동으로 /docs 로 리다이렉트
    return RedirectResponse(url="/docs")

# 서버 시작 시 한 번만 Stockfish 인스턴스 생성
stockfish_engine = StockfishEngine(path=STOCKFISH_PATH, depth=15)


class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="현재 체스 포지션의 FEN 문자열")
    last_move_uci: Optional[str] = Field(
        None,
        description="사용자가 방금 둔 수 (UCI 포맷, 예: 'e2e4')",
    )
    user_question: Optional[str] = Field(
        None,
        description="사용자의 질문 (예: '이 수 괜찮아요?', '대안은 뭐예요?')",
    )
    approx_elo: Optional[int] = Field(
        1200,
        description="사용자의 대략적인 Elo 레이팅 (설명 난이도 조절용)",
    )
    multi_pv: Optional[int] = Field(
        1,
        description="Stockfish MultiPV 값 (추천 수 여러 개 보고 싶으면 2 이상)",
    )


class StockfishResult(BaseModel):
    bestmove: str
    eval_cp: Optional[int]
    eval_type: Optional[str]
    pv_moves: List[str]


class AnalyzeResponse(BaseModel):
    stockfish: StockfishResult
    explanation: str


# ============================================================
#  GPT 프롬프트 생성 & 호출
# ============================================================

def build_gpt_prompt(
    fen: str,
    stockfish_result: dict,
    last_move_uci: Optional[str],
    user_question: Optional[str],
    approx_elo: Optional[int],
) -> str:
    """
    GPT에게 넘길 한국어 설명용 프롬프트 텍스트를 만든다.
    """
    eval_cp = stockfish_result.get("eval_cp")
    eval_type = stockfish_result.get("eval_type")
    bestmove = stockfish_result.get("bestmove")
    pv_moves = " ".join(stockfish_result.get("pv_moves", []))

    # 평가값 텍스트로 변환
    if eval_type == "mate" and eval_cp is not None:
        eval_text = f"mate in {eval_cp}"
    elif eval_cp is not None:
        eval_text = f"{eval_cp} centipawns (백 기준)"
    else:
        eval_text = "unknown"

    question_text = user_question or "이 포지션에서 어떻게 두면 좋은지 설명해줘."
    last_move_text = last_move_uci or "마지막 수 정보 없음"
    elo_text = approx_elo or 1200

    prompt = f"""
당신은 한국인 아마추어 체스 플레이어를 가르치는 체스 코치입니다.
플레이어의 대략적인 레이팅은 {elo_text} Elo 입니다.
설명은 한국어로, 가능한 한 이해하기 쉽게 해 주세요.

[포지션 정보]
- FEN: {fen}

[Stockfish 분석 결과]
- 최선의 수 (bestmove, UCI): {bestmove}
- 평가값 (score): {eval_text}
- 최선 수열 (PV, UCI): {pv_moves}

[사용자가 둔 마지막 수]
- {last_move_text}

[사용자 질문]
- {question_text}

다음 형식으로 설명해 주세요:
1. 현재 포지션에서 누가 유리한지 (백/흑, 어느 정도인지 직관적으로 설명).
2. Stockfish가 추천한 최선의 수가 왜 좋은지, 핵심 아이디어 2~3개.
3. 사용자가 둔 수가 최선 수와 비교해서 어떤 장점/단점이 있는지.
4. 비슷한 상황에서 기억해두면 좋은 간단한 원칙 1~2개.

전문 용어가 필요하면 괄호 안에 짧게 영어도 같이 적어 주세요
(예: '파일(file)', '디스커버드 체크(discovered check)').
"""
    return prompt.strip()


def ask_gpt_for_explanation(
    fen: str,
    stockfish_result: dict,
    last_move_uci: Optional[str],
    user_question: Optional[str],
    approx_elo: Optional[int],
) -> str:
    """
    OpenAI GPT 모델을 호출해서 한국어 해설을 생성한다.
    """
    prompt = build_gpt_prompt(
        fen=fen,
        stockfish_result=stockfish_result,
        last_move_uci=last_move_uci,
        user_question=user_question,
        approx_elo=approx_elo,
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",   # 필요하면 다른 모델 이름으로 변경 가능
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 친절한 체스 코치입니다. "
                    "Stockfish 엔진의 출력을 한국어로 쉽게 설명해 주는 역할입니다."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.4,
    )

    return completion.choices[0].message.content


# ============================================================
#  FastAPI 엔드포인트
# ============================================================

@app.get("/health")
def health_check():
    """Render 헬스 체크용 간단 엔드포인트."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_position(req: AnalyzeRequest):
    """
    FEN을 받아:
      1. Stockfish로 포지션 분석
      2. GPT로 한국어 설명 생성
    을 수행해서 함께 반환.
    """
    # 1) Stockfish 분석
    try:
        sf_result_dict = stockfish_engine.analyze_fen(
            fen=req.fen,
            multi_pv=req.multi_pv or 1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stockfish 분석 중 오류: {e}")

    # 2) GPT 설명
    try:
        explanation = ask_gpt_for_explanation(
            fen=req.fen,
            stockfish_result=sf_result_dict,
            last_move_uci=req.last_move_uci,
            user_question=req.user_question,
            approx_elo=req.approx_elo,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT 호출 중 오류: {e}")

    return AnalyzeResponse(
        stockfish=StockfishResult(**sf_result_dict),
        explanation=explanation,
    )

