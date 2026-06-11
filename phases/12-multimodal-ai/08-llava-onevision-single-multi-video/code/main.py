"""LLaVA-OneVision token 预算 + 课程规划器 — stdlib。

给定每样本总视觉 token 预算和任务混合（单图像、多图像、视频比例），分配：
  - 单图像的 AnyRes 瓦片数和 pooling 因子
  - 多图像的每样本图像数和每图像分辨率
  - 视频的每样本帧数和每帧 pooling

打印带每样本预期 FLOPs 的逐阶段训练计划。
跨场景保持预算大致恒定，因此 LLM 不会爆炸上下文。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    single_image_tokens: int
    multi_image_tokens: int
    video_tokens: int

    def max(self) -> int:
        return max(self.single_image_tokens, self.multi_image_tokens, self.video_tokens)

    def min(self) -> int:
        return min(self.single_image_tokens, self.multi_image_tokens, self.video_tokens)


def anyres_tokens(tiles: int, per_tile: int) -> int:
    return (tiles + 1) * per_tile


def per_tile_tokens(resolution: int, patch: int, pool: int) -> int:
    g = resolution // patch
    pooled = g // pool
    return pooled * pooled


def plan_single_image(budget: int) -> dict:
    for tiles in [9, 4, 1]:
        for per_tile_size in [(384, 14, 2), (384, 14, 1), (336, 14, 2)]:
            res, patch, pool = per_tile_size
            per = per_tile_tokens(res, patch, pool)
            total = anyres_tokens(tiles, per)
            if total <= budget:
                return {
                    "scenario": "single-image",
                    "tiles": tiles,
                    "tile_res": res,
                    "pool": pool,
                    "per_tile": per,
                    "total": total,
                }
    return {"scenario": "single-image", "tiles": 1, "per_tile": 81, "total": 162}


def plan_multi_image(budget: int) -> dict:
    for n_images in [8, 6, 4, 2]:
        for res_pool in [(384, 2), (384, 1), (336, 2)]:
            res, pool = res_pool
            per = per_tile_tokens(res, 14, pool)
            total = n_images * per
            if total <= budget:
                return {
                    "scenario": "multi-image",
                    "n_images": n_images,
                    "resolution": res,
                    "pool": pool,
                    "per_image": per,
                    "total": total,
                }
    return {"scenario": "multi-image", "n_images": 2, "per_image": 81, "total": 162}


def plan_video(budget: int) -> dict:
    for n_frames in [32, 16, 8]:
        for res_pool in [(384, 3), (384, 2), (336, 2)]:
            res, pool = res_pool
            per = per_tile_tokens(res, 14, pool)
            total = n_frames * per
            if total <= budget:
                return {
                    "scenario": "video",
                    "n_frames": n_frames,
                    "resolution": res,
                    "pool": pool,
                    "per_frame": per,
                    "total": total,
                }
    return {"scenario": "video", "n_frames": 8, "per_frame": 64, "total": 512}


def print_plan(plan: dict, budget: int) -> None:
    pct = 100 * plan["total"] / budget
    print(f"\n{plan['scenario'].upper():<12} budget target {budget:>5}, used {plan['total']:>5}  ({pct:>5.1f}%)")
    for k, v in plan.items():
        if k in ("scenario", "total"):
            continue
        print(f"    {k:<12}: {v}")


def curriculum_stages(mix: dict) -> None:
    print("\nCURRICULUM SCHEDULE (three-stage)")
    print("-" * 60)
    stages = [
        ("Stage SI  ", 1.0, 0.0, 0.0, "single-image only, AnyRes high-res"),
        ("Stage OV  ", 0.5, 0.3, 0.2, "OneVision mix, unified budget"),
        ("Stage TT  ", mix["single"], mix["multi"], mix["video"],
         "target-task fine-tune"),
    ]
    print(f"{'stage':<12}{'single':>8}{'multi':>8}{'video':>8}   notes")
    for name, s, m, v, note in stages:
        print(f"{name:<12}{s:>8.2f}{m:>8.2f}{v:>8.2f}   {note}")
    print("\nordering matters: stages in reverse (video first) underperform by "
          "2-4 MMMU per LLaVA-OneVision ablation.")


def main() -> None:
    print("=" * 60)
    print("LLAVA-ONEVISION TOKEN BUDGET + CURRICULUM (Phase 12, Lesson 08)")
    print("=" * 60)

    budget = 4096

    si = plan_single_image(budget)
    mi = plan_multi_image(budget)
    vi = plan_video(budget)

    print(f"\nshared per-sample visual token budget: {budget}")
    for p in (si, mi, vi):
        print_plan(p, budget)

    spread = max(si["total"], mi["total"], vi["total"]) - min(si["total"], mi["total"], vi["total"])
    print(f"\nbudget spread across scenarios: {spread} tokens "
          f"({100*spread/budget:.1f}% of budget)")
    print("LLaVA-OneVision target: keep spread under 30% for predictable LLM cost.")

    mix = {"single": 0.4, "multi": 0.3, "video": 0.3}
    curriculum_stages(mix)

    print("\nEMERGENT CAPABILITIES (reported in LLaVA-OneVision Sec 4.3)")
    print("-" * 60)
    print("  multi-camera reasoning  : multi-image + video curriculum combine")
    print("  set-of-mark prompting   : spatial grounding + multi-image ref")
    print("  iPhone-screenshot agent : UI screenshots + video workflows transfer")
    print("  none of the three appears in stage-SI data; curriculum unlocks them.")


if __name__ == "__main__":
    main()
