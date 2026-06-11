import re


# 菜系同义词映射：canonical 值 → 同义词列表
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza"],
    "chinese": ["chinese", "chow mein", "dim sum"],
    "indian":  ["indian", "curry", "tandoori"],
    "thai":    ["thai", "pad thai"],
}

AREA_WORDS = {"north", "south", "east", "west", "center", "centre"}
PRICE_WORDS = {
    "cheap":     ["cheap", "inexpensive", "budget"],
    "moderate":  ["moderate", "mid-range", "mid range", "medium"],
    "expensive": ["expensive", "fancy", "high-end", "upscale"],
}
CORRECTION_CUES = ["actually", "no wait", "on second thought", "change that to", "instead"]
NEGATION_CUES = ["never mind", "forget about", "don't worry about"]


def extract_cuisine(utterance):
    """从用户话语中提取 canonical 菜系。"""
    low = utterance.lower()
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in low for syn in synonyms):
            return canonical
    if "any cuisine" in low or "any food" in low:
        return "any"
    return None


def extract_area(utterance):
    """从用户话语中提取区域（north/south/east/west/center）。"""
    low = utterance.lower()
    for w in AREA_WORDS:
        if re.search(rf"\b{w}\b", low):
            return "center" if w == "centre" else w
    return None


def extract_price(utterance):
    """从用户话语中提取价格档位。"""
    low = utterance.lower()
    for canonical, synonyms in PRICE_WORDS.items():
        if any(syn in low for syn in synonyms):
            return canonical
    return None


def extract_people(utterance):
    """从用户话语中提取用餐人数。"""
    m = re.search(r"\b(\d+|two|three|four|five|six|seven|eight)\s+(?:people|guests|persons|diners)", utterance.lower())
    if not m:
        return None
    word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    raw = m.group(1)
    return int(raw) if raw.isdigit() else word_to_num.get(raw)


SLOT_EXTRACTORS = {
    "cuisine": extract_cuisine,
    "area":    extract_area,
    "price":   extract_price,
    "people":  extract_people,
}


def is_correction(utterance):
    """检测用户话语是否包含修正信号。"""
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)


def is_negation(utterance, slot):
    """检测用户话语是否包含否定信号并提及指定 slot。"""
    low = utterance.lower()
    return any(cue in low for cue in NEGATION_CUES) and slot in low


def update_state(state, utterance):
    """基于当前状态和新的话语增量更新对话状态。"""
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
            continue
        if is_negation(utterance, slot):
            new_state[slot] = None
    return new_state


def render_dialog(turns):
    """将对话轮次渲染为缩进字符串。"""
    return "\n".join(f"  user: {u}" for u in turns)


def main():
    """运行基于规则的 DST 演示并计算 Joint Goal Accuracy (JGA)。"""
    dialogues = [
        {
            "turns": [
                "I want a cheap restaurant in the north.",
                "Italian food please.",
                "Actually make it moderate pricing.",
                "For four people.",
            ],
            "gold": {"cuisine": "italian", "area": "north", "price": "moderate", "people": 4},
        },
        {
            "turns": [
                "Find me an expensive Chinese place.",
                "In the center of town.",
                "Six guests.",
                "Never mind the cuisine, any food is fine.",
            ],
            "gold": {"cuisine": "any", "area": "center", "price": "expensive", "people": 6},
        },
        {
            "turns": [
                "Thai food in the south.",
                "For two people.",
                "Moderate price.",
            ],
            "gold": {"cuisine": "thai", "area": "south", "price": "moderate", "people": 2},
        },
    ]

    print("=== rule-based DST ===")
    jga_correct = 0
    for i, d in enumerate(dialogues):
        state = {"cuisine": None, "area": None, "price": None, "people": None}
        print(f"\ndialogue {i}:")
        for turn in d["turns"]:
            state = update_state(state, turn)
            print(f"  user: {turn}")
            print(f"  state: {state}")
        ok = state == d["gold"]
        jga_correct += int(ok)
        print(f"  gold:  {d['gold']}")
        print(f"  match: {'  OK' if ok else 'MISS'}")

    print()
    print(f"=== Joint Goal Accuracy: {jga_correct}/{len(dialogues)} ({100 * jga_correct / len(dialogues):.1f}%) ===")
    print()
    print("note: rule-based works in narrow domains with canonical vocabulary.")
    print("open-vocab slots (restaurant name, reservation time) need LLM-based DST.")
    print("production pattern: regenerate-whole-state with Instructor + Pydantic schema.")


if __name__ == "__main__":
    main()
