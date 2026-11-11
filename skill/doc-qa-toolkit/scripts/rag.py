#!/usr/bin/env python3
"""Document Q&A CLI: index a folder of text/markdown files, then ask
questions with cited sources. Works fully offline.

Subcommands:
  index PATH [PATH...]   chunk and index .md/.txt files into the local store
  ask QUESTION           answer from the index; sources on stderr
  ask --prompt-only Q    print the retrieval prompt instead of an answer
  status                 show index location and document count
  reset                  delete the local index

Exit codes: 0 ok, 2 usage/config error.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragkit import BM25Retriever, EchoLLM, RAGPipeline, sentence_chunks

STATE_DIR = os.path.expanduser(os.environ.get("DOC_QA_DIR", "~/.doc-qa-toolkit"))
INDEX_PATH = os.path.join(STATE_DIR, "index.json")
TEXT_SUFFIXES = (".md", ".markdown", ".txt")


def _fail(message):
    print("error: %s" % message, file=sys.stderr)
    return 2


def _iter_files(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if name.lower().endswith(TEXT_SUFFIXES):
                        yield os.path.join(root, name)
        elif os.path.isfile(path):
            yield path
        else:
            print("warning: not found, skipped: %s" % path, file=sys.stderr)


def cmd_index(paths):
    os.makedirs(STATE_DIR, exist_ok=True)
    retriever = BM25Retriever()
    if os.path.exists(INDEX_PATH):
        retriever = BM25Retriever.load(INDEX_PATH)
    pipe = RAGPipeline(retriever, EchoLLM())
    files = list(_iter_files(paths))
    if not files:
        return _fail("no .md/.txt files found under: %s" % ", ".join(paths))
    total_chunks = 0
    for file_path in files:
        with open(file_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        chunks = sentence_chunks(text)
        if not chunks:
            continue
        metadatas = [{"source": os.path.abspath(file_path)}] * len(chunks)
        pipe.ingest(chunks, metadatas)
        total_chunks += len(chunks)
    retriever.save(INDEX_PATH)
    print("indexed %d chunks from %d files -> %s" % (total_chunks, len(files), INDEX_PATH))
    return 0


def _load_pipe():
    if not os.path.exists(INDEX_PATH):
        return None
    return RAGPipeline(BM25Retriever.load(INDEX_PATH), EchoLLM())


def _print_sources(sources):
    print("--- sources ---", file=sys.stderr)
    for i, src in enumerate(sources, 1):
        origin = src["metadata"].get("source", "?")
        print("[%d] score=%s %s" % (i, src["score"], origin), file=sys.stderr)
        print("    %s" % src["text"][:160].replace("\n", " "), file=sys.stderr)


def cmd_ask(args):
    prompt_only = "--prompt-only" in args
    k = 4
    rest = []
    it = iter(args)
    for arg in it:
        if arg == "--prompt-only":
            continue
        if arg == "-k":
            try:
                k = int(next(it))
            except (StopIteration, ValueError):
                return _fail("-k needs an integer")
            continue
        rest.append(arg)
    question = " ".join(rest).strip()
    if not question:
        return _fail("ask needs a question")
    pipe = _load_pipe()
    if pipe is None:
        return _fail("no index yet - run: rag.py index <folder>")
    result = pipe.query(question, k=k)
    if not result["sources"]:
        print("no matching passages in the index", file=sys.stderr)
        return 0
    if prompt_only:
        print(result["prompt"])
    else:
        print(result["answer"])
    _print_sources(result["sources"])
    return 0


def cmd_status():
    if not os.path.exists(INDEX_PATH):
        print("no index (%s does not exist)" % INDEX_PATH)
        return 0
    retriever = BM25Retriever.load(INDEX_PATH)
    print("index: %s" % INDEX_PATH)
    print("chunks: %d" % len(retriever))
    return 0


def cmd_reset():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        print("removed %s" % INDEX_PATH)
    else:
        print("no index to remove")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    command, args = argv[1], argv[2:]
    if command == "index":
        if not args:
            return _fail("index needs at least one file or folder")
        return cmd_index(args)
    if command == "ask":
        return cmd_ask(args)
    if command == "status":
        return cmd_status()
    if command == "reset":
        return cmd_reset()
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
