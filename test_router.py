"""Quick test: make a single LLM call through the router."""
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from engine.llm.router import RoutedLLMBackend

router = RoutedLLMBackend()
print(f"Router initialized with {len(router.chain())} models")

# Test with a simple JSON request
messages = [{"role": "user", "content": 'Return only this JSON, no other text: {"test": true, "status": "ok"}'}]

print("\nMaking test call...")
result = router.route(messages)

if result:
    print(f"SUCCESS!")
    print(f"  Model: {result.model.name} ({result.model.provider})")
    print(f"  Attempts: {result.attempts}")
    print(f"  Elapsed: {result.elapsed:.2f}s")
    print(f"  Fell back: {result.fell_back}")
    print(f"  Response: {result.text[:300]}")
    if result.notes:
        print(f"  Notes: {result.notes}")
else:
    print("FAILED - all providers exhausted")
    print(f"  Summary: {router.failure_summary()}")
