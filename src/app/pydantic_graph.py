from textwrap import dedent
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()


class WorkflowState(BaseModel):
    original_input: str
    extracted_summary: str | None = None
    confidence: Literal["low", "medium", "high"] | None = None
    needs_user_input: bool = False
    pending_question: str | None = None
    user_followup: str | None = None
    final_report: str | None = None
    retry_count: int = 0


class ExtractResult(BaseModel):
    summary: str
    confidence: Literal["low", "medium", "high"]


extract_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=ExtractResult,
    instructions=dedent(
        """
    입력을 분석해서 핵심 요약과 confidence를 반환하라.
    입력 정보가 충분하지 않으면 confidence를 low로 반환하라.
    """
    ).strip(),
)


class ReportResult(BaseModel):
    report: str


report_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=ReportResult,
    instructions=dedent(
        """
    주어진 요약과 사용자 입력을 바탕으로 최종 리포트를 작성하라.
    추측하지 말고 제공된 정보만 사용하라
    """
    ).strip(),
)


def extract_node(state: WorkflowState) -> WorkflowState:
    text = state.original_input
    if state.user_followup:
        text += f"\n\n추가 사용자 정보:\n{state.user_followup}"

    result = extract_agent.run_sync(text)

    state.extracted_summary = result.output.summary
    state.confidence = result.output.confidence

    return state


def validate_node(state: WorkflowState) -> WorkflowState:
    if state.confidence == "low":
        state.needs_user_input = True
        state.pending_question = "정보가 부족합니다. 중요한 세부사항을 더 알려주세요."
    else:
        state.needs_user_input = False
        state.pending_question = None

    return state


def generate_report_node(state: WorkflowState) -> WorkflowState:
    result = report_agent.run_sync(
        f"""
        원본 입력:
        {state.original_input}
        
        요약:
        {state.extracted_summary}
        
        추가 정보:
        {state.user_followup or "없음"}
        """.strip()
    )

    state.final_report = result.output.report
    return state


def run_workflow_once(state: WorkflowState) -> WorkflowState:
    state = extract_node(state)
    state = validate_node(state)

    if state.needs_user_input:
        return state

    state = generate_report_node(state)
    return state


def main() -> None:
    state = WorkflowState(
        original_input="고객이 제품 사용 중 큰 불편을 겪었다고 말했지만 세부 정보는 부족한 상태"
    )

    state = run_workflow_once(state)

    if state.needs_user_input:
        print("추가 입력 필요:")
        print(state.pending_question)

        user_followup = input("사용자 추가 입력: ")
        state.user_followup = user_followup
        state.retry_count += 1

        state = run_workflow_once(state)

    print("\n===== 최종 상태 =====")
    print(state.model_dump_json(indent=2, ensure_ascii=False))

    if state.final_report:
        print("\n===== 최종 리포트 =====")
        print(state.final_report)


if __name__ == "__main__":
    main()
