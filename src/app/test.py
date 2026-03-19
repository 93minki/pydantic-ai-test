from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from dotenv import load_dotenv

load_dotenv()


PRODUCTS = {
    "iphone 15": {"display_name": "아이폰 15", "price": 1250000, "stock": 3},
    "galaxy s24": {"display_name": "갤럭시 S24", "price": 1150000, "stock": 0},
    "airpods pro": {"display_name": "에어팟 프로", "price": 329000, "stock": 12},
}


@dataclass
class ShopDeps:
    user_name: str
    vip: bool = False


class QuoteItem(BaseModel):
    product_name: str = Field(description="상품 이름")
    quantity: int = Field(gt=0, description="상품 수량")
    unit_price: int = Field(gt=0, description="개당 가격(원)")
    line_total: int = Field(gt=0, description="해당 상품 총액(원)")
    in_stock: bool = Field(description="재고 보유 여부")


class QuoteResult(BaseModel):
    summary: str = Field(description="사용자에게 보여줄 최종 요약 문장")
    items: list[QuoteItem] = Field(description="견적에 포함된 상품 목록")
    subtotal: int = Field(ge=0, description="상품 금액 합계")
    shipping_fee: int = Field(get=0, description="배송비")
    total: int = Field(ge=0, description="최종 결제 예상 금액")
    unavailable_reasons: list[str] = Field(
        default_factory=list,
        description="재고 부족 등 구매 불가 사유 목록",
    )


agent = Agent[ShopDeps, QuoteResult](
    model="google-gla:gemini-2.5-flash-lite",
    deps_type=ShopDeps,
    output_type=QuoteResult,
    instructions="""
    너는 온라인 쇼핑몰 상담 에이전트다.
    반드시 필요한 경우에만 tool을 사용해 상품 정보, 재고, 배송비를 확인하라.
    가격이나 재고를 추측하지 마라.
    알 수 없는 상품은 없는 상품이라고 명확히 설명하라.
    최종 결과는 QuoteResult 구조에 맞춰 반환하라.
    """,
)


@agent.tool_plain
def normalize_product_name(raw_name: str) -> str:
    """
    사용자가 입력한 상품명을 내부 상품 키 형식으로 정규화한다.

    Args:
      raw_name: 사용자가 입력한 상품명
    """
    text = raw_name.strip().lower()

    alias_map = {
        "아이폰15": "iphone 15",
        "아이폰 15": "iphone 15",
        "iphone15": "iphone 15",
        "iphone 15": "iphone 15",
        "갤럭시s24": "galaxy s24",
        "갤럭시 s24": "galaxy s24",
        "galaxy s24": "galaxy s24",
        "에어팟프로": "airpods pro",
        "에어팟 프로": "airpods pro",
        "airpods pro": "airpods pro",
    }

    return alias_map.get(text, text)


@agent.tool_plain
def get_product_info(product_key: str) -> dict:
    """
    상품의 가격과 재고 정보를 조회한다.

    Args:
      product_key: 내부 상품 키
    """
    product = PRODUCTS.get(product_key)
    if product is None:
        return {
            "found": False,
            "display_name": product_key,
            "price": 0,
            "stock": 0,
        }

    return {
        "found": True,
        "display_name": product["display_name"],
        "price": product["price"],
        "stock": product["stock"],
    }


@agent.tool
def calculate_shipping_fee(ctx: RunContext[ShopDeps], subtotal: int) -> int:
    """
    주문 금액과 사용자 등급에 따라 배송비를 계산한다.

    Args:
      subtotal: 상품 금액 합계
    """
    if ctx.deps.vip:
        return 0

    if subtotal >= 50000:
        return 0

    return 3000


@agent.tool
def get_customer_tier_message(ctx: RunContext[ShopDeps]) -> str:
    """
    현재 사용자의 회원 등급 관련 안내 문구를 반환한다.
    """
    if ctx.deps.vip:
        return f"{ctx.deps.user_name}님은 VIP 회원으로 무료배송 대상입니다."
    return f"{ctx.deps.user_name}님은 일반 회원입니다."


def main() -> None:
    user_input = "아이폰 15 2개랑 에어팟 프로 1개 배송비 포함 견적 알려줘"

    result = agent.run_sync(user_input, deps=ShopDeps(user_name="민기", vip=True))

    print("===== 최종 구조화 결과 =====")
    print(result.output.model_dump_json(indent=2, ensure_ascii=False))

    print("\n===== 사람이 보기 쉬운 형태 =====")
    print("요약:", result.output.summary)
    print("소계:", result.output.subtotal)
    print("배송비:", result.output.shipping_fee)
    print("총액:", result.output.total)

    print("\n[상품 목록]")
    for item in result.output.items:
        print(
            f"- {item.product_name} / 수량 {item.quantity} / "
            f"개당 {item.unit_price}원 / 합계 {item.line_total}원 / "
            f"재고여부 {item.in_stock}"
        )

    if result.output.unavailable_reasons:
        print("\n[구매 불가 사유]")
        for reason in result.output.unavailable_reasons:
            print("-", reason)


if __name__ == "__main__":
    main()
