from textwrap import dedent
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


class ExtractedIssue(BaseModel):
    customer_name: str | None = Field(default=None, description="고객 이름")
    product_name: str | None = Field(default=None, description="관련 상품명")
    issue_type: Literal["배송", "환불", "품질", "결제", "기타"]
    urgency: Literal["낮음", "보통", "높음"]
    key_points: list[str] = Field(description="핵심 포인트 목록")
    sentiment: Literal["긍정", "부정", "중립"]


extract_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=ExtractedIssue,
    instructions=dedent(
        """
    너는 고객 VOC 분석 담당자다.
    사용자의 원문에서 사실에 근거해 핵심 정보를 추출하라.
    모르는 값은 추측하지 말고 null로 둬라.
    issue_type, urgency, sentiment는 가장 적절한 값으로 분류하라.
    """
    ).strip(),
)


class ChecklistItem(BaseModel):
    item: str
    reason: str


class ReviewChecklist(BaseModel):
    checklist: list[ChecklistItem]
    recommended_team: Literal["CS", "물류", "재무", "품질관리", "운영"]


checklist_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=ReviewChecklist,
    instructions=dedent(
        """
    너는 VOC 처리 체크리스트를 만드는 운영 매니저다.
    입력으로 받은 분석 결과를 바탕으로 실제 후속 처리에 필요한 체크리스트를 작성하라.
    체크리스트는 구체적이고 실행 가능해야 한다.
    """
    ).strip(),
)


class FinalReport(BaseModel):
    summary: str = Field(description="한눈에 보는 요약")
    risk_level: Literal["낮음", "보통", "높음"]
    action_plan: list[str] = Field(description="실행 계획")
    customer_reply_draft: str = Field(description="고객에게 보낼 답변 초안")
    internal_report: str = Field(description="내부 공유용 종합 리포트")


report_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=FinalReport,
    instructions=dedent(
        """
    너는 VOC 최종 보고서를 작성하는 담당자다.
    반드시 입력으로 주어진 원문, 분석 결과, 체크리스트를 모두 반영하라.
    customer_reply_draft는 고객 친화적으로 작성하라.
    internal_report는 내부 공유용으로 명확하고 간결하게 작성하라.
    추측은 하지 말고 제공된 정보만 사용하라.
    """
    ).strip(),
)


def main() -> None:
    user_input = dedent(
        """
        안녕하세요. 김민기인데요.
        지난주에 주문한 무선 이어폰을 어제 받았는데, 오른쪽 이어버드가 충전이 안 됩니다.
        급하게 필요해서 샀는데 바로 써야 하는 상황이라 많이 곤란합니다.
        가능하면 빠른 교환이나 환불 절차를 안내받고 싶습니다.
        """
    ).strip()

    step1 = extract_agent.run_sync(user_input)
    extracted = step1.output

    print("===== STEP 1: 추출 결과 =====")
    print(extracted.model_dump_json(indent=2, ensure_ascii=False))

    step2_input = dedent(
        f"""
      [원문]
      {user_input}
      
      [분석 결과 JSON]
      {extracted.model_dump_json(indent=2, ensure_ascii=False)}
      """
    ).strip()

    step2 = checklist_agent.run_sync(step2_input)
    checklist = step2.output

    print("===== STEP 2: 체크리스트 결과 =====")
    print(checklist.model_dump_json(indent=2, ensure_ascii=False))

    step3_input = dedent(
        f"""
        [원문]
        {user_input}

        [1차 분석 결과]
        {extracted.model_dump_json(indent=2, ensure_ascii=False)}

        [처리 체크리스트]
        {checklist.model_dump_json(indent=2, ensure_ascii=False)}
        """
    ).strip()

    step3 = report_agent.run_sync(step3_input)
    final_report = step3.output

    print("\n===== STEP 3: 최종 리포트 =====")
    print(final_report.model_dump_json(indent=2, ensure_ascii=False))

    print("\n===== 사람이 읽기 쉬운 출력 =====")
    print("요약:", final_report.summary)
    print("위험도:", final_report.risk_level)

    print("\n[실행 계획]")
    for i, action in enumerate(final_report.action_plan, start=1):
        print(f"{i}. {action}")

    print("\n[고객 답변 초안]")
    print(final_report.customer_reply_draft)

    print("\n[내부 보고서]")
    print(final_report.internal_report)


if __name__ == "__main__":
    main()
