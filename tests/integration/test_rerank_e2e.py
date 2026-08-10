"""
Rerank Converter End-to-End Integration Test

Tests the full conversion pipeline with real rerank API calls:
- response_from_provider: real API response → IR
- Cross-provider: Provider A response → IR → Provider B format
- Request round-trip: IR → provider request → IR

Requires API keys in .env:
- JINA_API_KEY
- COHERE_API_KEY
- VOYAGE_API_KEY
- SILICONFLOW_API_KEY + SILICONFLOW_BASE_URL

Usage:
    conda activate llm-rosetta
    python tests/integration/test_rerank_e2e.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import dotenv
import requests

from llm_rosetta.converters.rerank import (
    CohereRerankConverter,
    JinaRerankConverter,
    VoyageRerankConverter,
)
from llm_rosetta.types.ir.rerank import IRRerankRequest, RerankDocument

dotenv.load_dotenv()

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is in Paris.",
]
TOP_N = 2


def _call_jina() -> dict:
    resp = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": QUERY,
            "documents": DOCUMENTS,
            "top_n": TOP_N,
            "return_documents": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call_cohere() -> dict:
    resp = requests.post(
        "https://api.cohere.com/v2/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['COHERE_API_KEY']}",
        },
        json={
            "model": "rerank-v3.5",
            "query": QUERY,
            "documents": DOCUMENTS,
            "top_n": TOP_N,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call_voyage() -> dict:
    resp = requests.post(
        "https://api.voyageai.com/v1/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}",
        },
        json={
            "model": "rerank-2-lite",
            "query": QUERY,
            "documents": DOCUMENTS,
            "top_k": TOP_N,
            "return_documents": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call_siliconflow() -> dict:
    base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn")
    resp = requests.post(
        f"{base_url}/v1/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['SILICONFLOW_API_KEY']}",
        },
        json={
            "model": "BAAI/bge-reranker-v2-m3",
            "query": QUERY,
            "documents": DOCUMENTS,
            "top_n": TOP_N,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def test_jina_response_to_ir():
    print("\n=== Jina → IR ===")
    raw = _call_jina()
    converter = JinaRerankConverter()
    ir = converter.response_from_provider(raw)
    assert ir["object"] == "rerank"
    assert len(ir["results"]) == TOP_N
    assert ir["results"][0]["index"] == 0  # Paris doc should rank first
    assert ir["results"][0]["relevance_score"] > ir["results"][1]["relevance_score"]
    assert "document" in ir["results"][0]
    assert "usage" in ir
    print(f"  model: {ir['model']}")
    print(f"  top result: index={ir['results'][0]['index']}, score={ir['results'][0]['relevance_score']:.4f}")
    print(f"  usage: {ir.get('usage')}")
    print("  ✓ PASS")


def test_cohere_response_to_ir():
    print("\n=== Cohere → IR ===")
    raw = _call_cohere()
    converter = CohereRerankConverter()
    ir = converter.response_from_provider(raw)
    assert ir["object"] == "rerank"
    assert len(ir["results"]) == TOP_N
    assert ir["results"][0]["index"] == 0
    assert "id" in ir
    print(f"  id: {ir['id']}")
    print(f"  top result: index={ir['results'][0]['index']}, score={ir['results'][0]['relevance_score']:.4f}")
    print("  ✓ PASS")


def test_voyage_response_to_ir():
    print("\n=== Voyage → IR ===")
    raw = _call_voyage()
    converter = VoyageRerankConverter()
    ir = converter.response_from_provider(raw)
    assert ir["object"] == "rerank"
    assert ir["model"] == "rerank-2-lite"
    assert len(ir["results"]) == TOP_N
    assert ir["results"][0]["index"] == 0
    assert "document" in ir["results"][0]
    assert "usage" in ir
    print(f"  model: {ir['model']}")
    print(f"  top result: index={ir['results'][0]['index']}, score={ir['results'][0]['relevance_score']:.4f}")
    print(f"  usage: {ir.get('usage')}")
    print("  ✓ PASS")


def test_siliconflow_response_to_ir():
    print("\n=== Siliconflow → IR (via Cohere converter) ===")
    raw = _call_siliconflow()
    converter = CohereRerankConverter()
    ir = converter.response_from_provider(raw)
    assert ir["object"] == "rerank"
    assert len(ir["results"]) == TOP_N
    assert ir["results"][0]["index"] == 0
    assert "document" not in ir["results"][0]  # Siliconflow sends null
    assert "usage" in ir
    assert ir["usage"]["total_tokens"] > 0
    print(f"  id: {ir.get('id')}")
    print(f"  top result: index={ir['results'][0]['index']}, score={ir['results'][0]['relevance_score']:.4f}")
    print(f"  usage: {ir.get('usage')}")
    print("  ✓ PASS")


def test_cross_provider_jina_to_voyage():
    print("\n=== Cross-provider: Jina → IR → Voyage ===")
    jina_raw = _call_jina()
    jina = JinaRerankConverter()
    voyage = VoyageRerankConverter()
    ir = jina.response_from_provider(jina_raw)
    voyage_resp = voyage.response_to_provider(ir)
    assert voyage_resp["object"] == "list"
    assert "data" in voyage_resp
    assert len(voyage_resp["data"]) == TOP_N
    assert voyage_resp["data"][0]["document"] == "Paris is the capital of France."
    print(f"  Voyage format: {json.dumps(voyage_resp, indent=2)[:200]}...")
    print("  ✓ PASS")


def test_cross_provider_cohere_to_jina():
    print("\n=== Cross-provider: Cohere → IR → Jina ===")
    cohere_raw = _call_cohere()
    cohere = CohereRerankConverter()
    jina = JinaRerankConverter()
    ir = cohere.response_from_provider(cohere_raw)
    jina_resp = jina.response_to_provider(ir)
    assert jina_resp["object"] == "list"
    assert "results" in jina_resp
    assert len(jina_resp["results"]) == TOP_N
    print(f"  Jina format: {json.dumps(jina_resp, indent=2)[:200]}...")
    print("  ✓ PASS")


def test_request_roundtrip_all_providers():
    print("\n=== Request round-trip: IR → Provider → IR (all 3) ===")
    ir_request = IRRerankRequest(
        model="test-model",
        query=QUERY,
        documents=[RerankDocument(text=d) for d in DOCUMENTS],
        top_n=TOP_N,
        return_documents=True,
    )

    for name, converter in [
        ("Jina", JinaRerankConverter()),
        ("Cohere", CohereRerankConverter()),
        ("Voyage", VoyageRerankConverter()),
    ]:
        provider_req, warnings = converter.request_to_provider(ir_request)
        ir_back = converter.request_from_provider(provider_req)
        assert ir_back["model"] == ir_request["model"]
        assert ir_back["query"] == ir_request["query"]
        assert len(ir_back["documents"]) == len(ir_request["documents"])
        for orig, back in zip(ir_request["documents"], ir_back["documents"], strict=True):
            assert orig["text"] == back["text"]
        print(f"  {name}: ✓")
    print("  ✓ ALL PASS")


if __name__ == "__main__":
    tests = [
        test_jina_response_to_ir,
        test_cohere_response_to_ir,
        test_voyage_response_to_ir,
        test_siliconflow_response_to_ir,
        test_cross_provider_jina_to_voyage,
        test_cross_provider_cohere_to_jina,
        test_request_roundtrip_all_providers,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
