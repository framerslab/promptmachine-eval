"""
Basic Usage Examples for promptmachine-eval

This script demonstrates the core features of the package.
Run with: python examples/basic_usage.py
"""

import asyncio
import os
from promptmachine_eval import (
    EloCalculator,
    EloConfig,
    MatchmakingService,
    ModelInfo,
    CostTracker,
    PromptTester,
    BattleRunner,
)


def demo_elo_calculations():
    """Demonstrate ELO rating calculations."""
    print("\n" + "=" * 50)
    print("ELO RATING SYSTEM")
    print("=" * 50)

    # Create calculator with default config
    elo = EloCalculator()

    # Example 1: Expected scores
    print("\n📊 Expected Win Probabilities:")
    for rating_diff in [0, 100, 200, 400]:
        prob = elo.expected_score(1200, 1200 - rating_diff)
        print(f"  1200 vs {1200 - rating_diff}: {prob:.1%}")

    # Example 2: Rating updates
    print("\n🎯 Rating Updates After Battles:")

    # Equal players, A wins
    new_a, new_b = elo.update_ratings(1000, 1000, score_a=1.0)
    print(f"  Equal (1000 vs 1000), A wins: A→{new_a:.0f}, B→{new_b:.0f}")

    # Upset: underdog wins
    new_a, new_b = elo.update_ratings(1000, 1200, score_a=1.0)
    print(f"  Upset (1000 vs 1200), underdog wins: A→{new_a:.0f}, B→{new_b:.0f}")

    # Draw
    new_a, new_b = elo.update_ratings(1100, 1000, score_a=0.5)
    print(f"  Draw (1100 vs 1000): A→{new_a:.0f}, B→{new_b:.0f}")

    # Example 3: Full update with uncertainty
    print("\n📈 Full Update with Uncertainty:")
    result = elo.update_after_battle(
        rating_a=1200,
        rating_b=1000,
        sd_a=150,
        sd_b=250,
        score_a=1.0,
    )
    print(f"  A: {result.new_rating_a:.0f} ({result.rating_change_a:+d}), SD: {result.new_sd_a:.0f}")
    print(f"  B: {result.new_rating_b:.0f} ({result.rating_change_b:+d}), SD: {result.new_sd_b:.0f}")
    print(f"  Expected score for A was: {result.expected_score_a:.1%}")


def demo_matchmaking():
    """Demonstrate matchmaking for battles."""
    print("\n" + "=" * 50)
    print("MATCHMAKING SERVICE")
    print("=" * 50)

    service = MatchmakingService()

    # Create a roster of models
    models = [
        ModelInfo(id="gpt-4o", rating=1250, sd=80, battles_count=100, display_name="GPT-4o"),
        ModelInfo(id="claude-3.5-sonnet", rating=1230, sd=90, battles_count=80, display_name="Claude 3.5 Sonnet"),
        ModelInfo(id="gemini-pro", rating=1150, sd=150, battles_count=30, display_name="Gemini Pro"),
        ModelInfo(id="llama-3.1-70b", rating=1100, sd=200, battles_count=15, display_name="Llama 3.1 70B"),
        ModelInfo(id="mistral-large", rating=1050, sd=300, battles_count=5, display_name="Mistral Large"),
    ]

    print("\n🎮 Model Roster:")
    for m in models:
        print(f"  {m.display_name}: ELO {m.rating:.0f} ±{m.sd:.0f} ({m.battles_count} battles)")

    # Select optimal pairing
    print("\n🎯 Optimal Pairing Selected:")
    model_a, model_b = service.select_pair_for_battle(models)
    print(f"  {model_a.display_name} vs {model_b.display_name}")

    # Score analysis
    score = service.score_match(model_a, model_b)
    print(f"  Competitiveness: {score.competitiveness_score:.2f}")
    print(f"  Uncertainty value: {score.uncertainty_score:.2f}")
    print(f"  Total score: {score.total_score:.2f}")

    # Multiple matches
    print("\n📋 Multiple Non-Overlapping Matches:")
    matches = service.select_matches(models, num_matches=2, avoid_repeats=True)
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m.model_a.display_name} vs {m.model_b.display_name} (score: {m.total_score:.2f})")


def demo_cost_tracking():
    """Demonstrate cost tracking and estimation."""
    print("\n" + "=" * 50)
    print("COST TRACKING")
    print("=" * 50)

    tracker = CostTracker(session_budget=1.00)

    prompt = "Explain the theory of relativity in simple terms for a high school student."

    print("\n💰 Cost Estimates for Different Models:")
    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "claude-3-5-haiku"]

    for model in models:
        estimate = tracker.estimate(prompt, model, expected_output_tokens=500)
        print(f"  {model}:")
        print(f"    Input: {estimate.input_tokens} tokens (${estimate.input_cost:.5f})")
        print(f"    Output: ~{estimate.output_tokens} tokens (${estimate.output_cost:.5f})")
        print(f"    Total: ${estimate.total:.5f}")

    # Simulate tracking
    print("\n📊 Simulated Session Tracking:")
    tracker.track_usage("gpt-4o-mini", input_tokens=50, output_tokens=200)
    tracker.track_usage("gpt-4o-mini", input_tokens=100, output_tokens=300)

    summary = tracker.get_summary()
    print(f"  Calls: {summary['num_calls']}")
    print(f"  Tokens: {summary['total_tokens']}")
    print(f"  Cost: ${summary['total_cost']:.4f}")
    print(f"  Budget remaining: ${summary['budget_remaining']:.4f}")


async def demo_prompt_testing():
    """Demonstrate prompt testing (requires API keys)."""
    print("\n" + "=" * 50)
    print("PROMPT TESTING")
    print("=" * 50)

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("\n⚠️  Set OPENAI_API_KEY to run this demo")
        print("   Example: export OPENAI_API_KEY=sk-...")
        return

    tester = PromptTester(openai_api_key=openai_key)

    print("\n🧪 Testing prompt across models...")
    results = await tester.test(
        prompt="What is 2 + 2? Reply with just the number.",
        models=["gpt-4o-mini", "gpt-6-luna"],
        temperature=0.0,
        max_tokens=10,
    )

    for r in results:
        if r.error:
            print(f"  {r.model}: ERROR - {r.error}")
        else:
            print(f"  {r.model}: '{r.response.strip()}' ({r.latency_ms}ms, ${r.cost:.5f})")


async def demo_arena_battle():
    """Demonstrate arena battle (requires API keys)."""
    print("\n" + "=" * 50)
    print("ARENA BATTLE")
    print("=" * 50)

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("\n⚠️  Set OPENAI_API_KEY to run this demo")
        return

    runner = BattleRunner(openai_api_key=openai_key)

    print("\n⚔️  Running battle: gpt-4o-mini vs gpt-6-luna")
    result = await runner.battle(
        prompt="Write a one-sentence definition of recursion.",
        model_a="gpt-4o-mini",
        model_b="gpt-6-luna",
        judge_model="gpt-4o-mini",
    )

    print(f"\n🏆 Winner: {result.winner}")
    print(f"📝 Judge reasoning: {result.judgement.reasoning[:200]}...")
    print(f"💰 Total cost: ${result.total_cost:.4f}")


def main():
    """Run all demos."""
    print("\n" + "🚀 " * 20)
    print("promptmachine-eval Demo")
    print("🚀 " * 20)

    # Sync demos (no API needed)
    demo_elo_calculations()
    demo_matchmaking()
    demo_cost_tracking()

    # Async demos (need API keys)
    asyncio.run(demo_prompt_testing())
    asyncio.run(demo_arena_battle())

    print("\n" + "=" * 50)
    print("Demo complete! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    main()

