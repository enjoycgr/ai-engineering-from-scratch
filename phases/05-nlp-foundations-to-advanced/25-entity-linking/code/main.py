import re
from collections import Counter


ALIAS_INDEX = {
    "jordan": ["Q41421", "Q810", "Q254110", "Q3308285"],
    "paris":  ["Q90", "Q663094", "Q55411"],
    "apple":  ["Q312", "Q89"],
    "washington": ["Q23", "Q1223", "Q61"],
    "python": ["Q28865", "Q83320"],
}

# KB 描述：每个 QID 对应一段用于消歧的文本
KB_DESC = {
    "Q41421":  "Michael Jordan American basketball player Chicago Bulls six championships",
    "Q810":    "Jordan country Middle East kingdom capital Amman Arabic",
    "Q254110": "Michael B Jordan American actor Black Panther Creed film",
    "Q3308285": "Michael I Jordan Berkeley professor machine learning statistics",
    "Q90":     "Paris capital of France Eiffel Tower Seine river city",
    "Q663094": "Paris Texas city United States Lamar County",
    "Q55411":  "Paris Hilton American socialite hotel heiress television personality",
    "Q312":    "Apple Inc American technology company iPhone Mac Tim Cook Cupertino",
    "Q89":     "Apple fruit tree species Malus domestica red green grown worldwide",
    "Q23":     "George Washington American founding father first president",
    "Q1223":   "Washington state Pacific northwest United States Seattle capital Olympia",
    "Q61":     "Washington DC capital city of United States federal district",
    "Q28865":  "Python programming language Guido van Rossum interpreted dynamic",
    "Q83320":  "Python snake nonvenomous constrictor species Asia Africa large",
}

# 先验概率 (popularity prior)：在缺少上下文时各实体的流行度
PRIORS = {
    "Q41421":  0.50, "Q810":    0.20, "Q254110": 0.20, "Q3308285": 0.10,
    "Q90":     0.85, "Q663094": 0.05, "Q55411":  0.10,
    "Q312":    0.70, "Q89":     0.30,
    "Q23":     0.40, "Q1223":   0.30, "Q61":     0.30,
    "Q28865":  0.80, "Q83320":  0.20,
}


def tokenize(text):
    """将文本拆分为小写 token 集合。"""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def disambiguate(mention, context, use_prior=True):
    """
    基于 Jaccard 相似度 + 可选先验对候选实体进行消歧。
    返回 (最佳 QID, 得分)。
    """
    candidates = ALIAS_INDEX.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    ctx_tokens = tokenize(context)
    scored = []
    for cand in candidates:
        desc_tokens = tokenize(KB_DESC.get(cand, ""))
        if not desc_tokens:
            jac = 0.0
        else:
            overlap = len(ctx_tokens & desc_tokens)
            union = len(ctx_tokens | desc_tokens)
            jac = overlap / union if union else 0.0
        prior = PRIORS.get(cand, 0.0) if use_prior else 1.0
        score = jac + 0.1 * prior
        scored.append((cand, score, jac, prior))
    scored.sort(key=lambda x: -x[1])
    return scored[0][0], scored[0][1]


def run_eval(test_cases, use_prior):
    """在一组测试用例上运行消歧并打印结果。"""
    correct = 0
    print(f"=== 消歧 {'使用' if use_prior else '不使用'}先验 ===")
    for mention, context, gold in test_cases:
        pred, score = disambiguate(mention, context, use_prior=use_prior)
        ok = pred == gold
        correct += int(ok)
        tag = "  OK" if ok else "MISS"
        print(f"  [{tag}] mention={mention:<10} pred={pred:<7} gold={gold:<7} score={score:.3f}")
        print(f"         context: {context[:70]}...")
    print(f"  accuracy: {correct}/{len(test_cases)} ({100 * correct / len(test_cases):.1f}%)")
    print()
    return correct


def main():
    # 测试用例：(mention, 上下文, 黄金 QID)
    test_cases = [
        ("Jordan", "Jordan scored 45 points against the Lakers last night.", "Q41421"),
        ("Jordan", "Jordan borders Syria, Iraq, and Saudi Arabia in the Middle East.", "Q810"),
        ("Jordan", "Jordan starred in the superhero movie Black Panther.", "Q254110"),
        ("Jordan", "Jordan's work on variational inference shaped machine learning.", "Q3308285"),
        ("Paris",  "Paris is the capital of France and home to the Eiffel Tower.", "Q90"),
        ("Paris",  "Paris Texas is a small city in Lamar County.", "Q663094"),
        ("Paris",  "Paris Hilton is a television personality and heiress.", "Q55411"),
        ("Apple",  "Apple announced the new iPhone at their Cupertino event.", "Q312"),
        ("Apple",  "An apple a day keeps the doctor away; the fruit is rich in fiber.", "Q89"),
        ("Python", "Python is a popular programming language used for data science.", "Q28865"),
        ("Python", "The python is a large nonvenomous snake found in Asia.", "Q83320"),
    ]

    run_eval(test_cases, use_prior=True)
    run_eval(test_cases, use_prior=False)

    print("注意：这是仅 11 个用例的 toy 测试集。")
    print("生产级 EL 使用 Wikipedia 别名 dump（约 1800 万别名）和基于编码器的消歧。")


if __name__ == "__main__":
    main()
