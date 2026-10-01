"""Build a real-retrieval RAG workload: BM25 top-k over a BEIR corpus.

Downloads a BEIR dataset (default NFCorpus: 3.6k documents, 3.2k queries),
retrieves the top-k documents for every query with BM25 over title + text, and
measures document and query lengths with the Qwen2.5 tokenizer. Only the
structure is saved (lengths and ranked document ids per query); token contents
do not matter for prefix caching, only identity and length.

    pip install rank_bm25 pyarrow
    python benchmarks/build_real_rag.py [--dataset BeIR/nfcorpus] [--k 5]
Writes docs/real_rag_<name>.json.
"""

import argparse
import json
import re

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

TOKEN = re.compile(r"[a-z0-9]+")


def words(text):
    return TOKEN.findall(text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BeIR/nfcorpus")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    def table(name):
        path = hf_hub_download(args.dataset, f"{name}/{name}-00000-of-00001.parquet",
                               repo_type="dataset")
        return pq.read_table(path).to_pylist()

    corpus = table("corpus")
    queries = table("queries")
    print(f"{len(corpus)} documents, {len(queries)} queries")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    doc_texts = [f"{d['title']}\n{d['text']}" for d in corpus]
    doc_lens = [len(tok.encode(t)) for t in doc_texts]
    query_lens = [len(tok.encode(q["text"])) for q in queries]

    bm25 = BM25Okapi([words(t) for t in doc_texts])
    retrievals = []
    for i, q in enumerate(queries):
        scores = bm25.get_scores(words(q["text"]))
        top = sorted(range(len(scores)), key=lambda d: -scores[d])[: args.k]
        retrievals.append(top)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(queries)} queries retrieved", flush=True)

    name = args.dataset.split("/")[-1]
    with open(f"docs/real_rag_{name}.json", "w") as f:
        json.dump(dict(dataset=args.dataset, retriever="BM25Okapi(title+text)", k=args.k,
                       tokenizer=args.tokenizer, doc_lens=doc_lens, query_lens=query_lens,
                       retrievals=retrievals), f)
    print(f"wrote docs/real_rag_{name}.json")


if __name__ == "__main__":
    main()
