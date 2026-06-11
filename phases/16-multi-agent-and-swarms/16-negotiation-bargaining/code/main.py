"""协商：Contract Net (合同网) + OG-Narrator 演示，仅使用标准库。

对比 naive all-LLM 讨价还价与 OG-Narrator（确定性报价生成器 + LLM 叙述）。
测量 1000 次试验的成交率。末尾包含一个小型 Contract Net 任务市场演示。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class BargainState:
    buyer_max: int
    seller_min: int
    buyer_offer: int | None = None
    seller_offer: int | None = None
    rounds: int = 0
    max_rounds: int = 5


def naive_llm_bargain(state: BargainState, rng: random.Random) -> int:
    """模拟 naive LLM 讨价还价：以高方差选择价格，且经常超出 ZOPA
    （模仿 arXiv:2402.15813 中记录的战略错误）。"""
    r = rng.random()
    if state.seller_offer is None:
        candidate = rng.randint(state.buyer_max - 60, state.buyer_max + 30)
    elif r < 0.35:
        candidate = state.seller_offer + rng.randint(-8, 3)
    elif r < 0.65:
        candidate = rng.randint(state.seller_min - 30, state.buyer_max + 30)
    else:
        candidate = rng.randint(state.seller_min - 60, state.buyer_max + 60)
    return candidate


def og_narrator_bargain(state: BargainState, rng: random.Random,
                        concession: float = 0.35) -> int:
    """OG-Narrator 确定性报价：向中点进行 Zeuthen 风格让步。"""
    if state.seller_offer is None and state.buyer_offer is None:
        return state.buyer_max - max(1, int((state.buyer_max - state.seller_min) * 0.2))
    if state.seller_offer is None:
        return state.buyer_offer
    prior = state.buyer_offer if state.buyer_offer is not None else state.buyer_max
    move = max(1, int(concession * (state.seller_offer - prior)))
    candidate = prior + move
    candidate = min(candidate, state.buyer_max)
    return candidate


def seller_response(state: BargainState, rng: random.Random,
                    concession: float = 0.3) -> int:
    """卖方同样使用 OG-Narrator 风格的报价（对两种买方都适用）。"""
    if state.buyer_offer is None and state.seller_offer is None:
        return state.seller_min + max(1, int((state.buyer_max - state.seller_min) * 0.4))
    if state.buyer_offer is None:
        return state.seller_offer
    prior = state.seller_offer if state.seller_offer is not None else state.seller_min + 20
    move = max(1, int(concession * (prior - state.buyer_offer)))
    candidate = prior - move
    candidate = max(candidate, state.seller_min)
    return candidate


def simulate_bargain(buyer_fn, rng: random.Random, buyer_max: int = 100,
                     seller_min: int = 60) -> bool:
    state = BargainState(buyer_max=buyer_max, seller_min=seller_min)
    deal = False
    while state.rounds < state.max_rounds:
        state.buyer_offer = buyer_fn(state, rng)
        if state.seller_offer is not None and state.buyer_offer >= state.seller_offer:
            # 交易以卖方当前要价成交；仅当在双方保留价范围内时才可行
            if state.seller_offer >= state.seller_min and state.seller_offer <= state.buyer_max:
                deal = True
            break
        state.seller_offer = seller_response(state, rng)
        if state.buyer_offer is not None and state.seller_offer <= state.buyer_offer:
            # 交易以买方当前出价成交；仅当在双方保留价范围内时才可行
            if state.buyer_offer <= state.buyer_max and state.buyer_offer >= state.seller_min:
                deal = True
            break
        state.rounds += 1
    return deal


def bench_deal_rate(buyer_fn, label: str, trials: int = 1000) -> None:
    rng = random.Random(42)
    deals = 0
    for _ in range(trials):
        seller_min = rng.randint(50, 80)
        buyer_max = rng.randint(max(seller_min + 5, 75), 115)
        if simulate_bargain(buyer_fn, rng, buyer_max=buyer_max, seller_min=seller_min):
            deals += 1
    print(f"  {label:20s} deal rate: {deals / trials:.2%}  ({deals}/{trials})")


@dataclass
class Bid:
    bidder: str
    price: int
    eta_minutes: int
    confidence: float


@dataclass
class ContractNetTask:
    task_id: str
    description: str
    deadline_minutes: int
    budget: int


class ContractNetManager:
    def __init__(self, bidders: list[str]) -> None:
        self.bidders = bidders
        self.proposals: dict[str, list[Bid]] = {}

    def broadcast_cfp(self, task: ContractNetTask) -> None:
        self.proposals[task.task_id] = []
        print(f"  manager CFP -> {task.description} (deadline {task.deadline_minutes}m, budget {task.budget})")

    def receive_proposal(self, task_id: str, bid: Bid) -> None:
        self.proposals[task_id].append(bid)
        print(f"    propose from {bid.bidder}: price={bid.price} eta={bid.eta_minutes}m conf={bid.confidence:.2f}")

    def award(self, task: ContractNetTask) -> Bid | None:
        props = self.proposals.get(task.task_id, [])
        feasible = [b for b in props if b.price <= task.budget and b.eta_minutes <= task.deadline_minutes]
        if not feasible:
            print("    no feasible bid; awarding rejected")
            return None
        winner = max(feasible, key=lambda b: b.confidence / max(b.price, 1))
        print(f"  manager accept-proposal -> {winner.bidder} (score = conf/price)")
        for b in props:
            if b is not winner:
                print(f"  manager reject-proposal -> {b.bidder}")
        return winner


def demo_contract_net() -> None:
    print("\n" + "=" * 72)
    print("CONTRACT NET 任务市场 —— manager + 3 个 bidders")
    print("=" * 72)
    task = ContractNetTask(
        task_id="t-1",
        description="compress 10GB log bundle",
        deadline_minutes=30,
        budget=10,
    )
    mgr = ContractNetManager(bidders=["worker-a", "worker-b", "worker-c"])
    mgr.broadcast_cfp(task)
    mgr.receive_proposal(task.task_id, Bid("worker-a", price=3, eta_minutes=18, confidence=0.82))
    mgr.receive_proposal(task.task_id, Bid("worker-b", price=2, eta_minutes=25, confidence=0.77))
    mgr.receive_proposal(task.task_id, Bid("worker-c", price=4, eta_minutes=10, confidence=0.90))
    mgr.award(task)


def main() -> None:
    print("=" * 72)
    print("成交率 —— naive LLM 讨价还价 vs OG-Narrator")
    print("每次试验采样保留价：seller_min ∈ [50,80]，buyer_max ∈ [75,115]")
    print("=" * 72)
    bench_deal_rate(naive_llm_bargain, "naive LLM")
    bench_deal_rate(og_narrator_bargain, "OG-Narrator")
    demo_contract_net()

    print("\n要点:")
    print("  naive LLM-only 讨价还价具有过高的方差——波动超出 ZOPA (可能达成协议的区域)。")
    print("  OG-Narrator（确定性报价 + LLM 叙述）在每次试验中都能收敛，")
    print("  因为价格是算术计算，而非生成式的。")
    print("  原始论文 (arXiv:2402.15813) 报告在更严格的真实 LLM 基准上从 26.67% -> 88.88%。")
    print("  我们的模拟缩小了差距，因为对方卖方已经在使用确定性报价——结构模式是相同的。")
    print("  Contract Net 可扩展：广播 + 收集 + 授予；不需要同步聊天。")


if __name__ == "__main__":
    main()
