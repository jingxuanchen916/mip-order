"""Run one model on one setting of the MIP data and append one JSON line per prompt.

    python run.py --model meta-llama/llama-4-scout --setting core
    python run.py --model meta-llama/llama-4-scout --setting core --prompt self_prompt
    python run.py --model Qwen/Qwen3-4B-Instruct-2507 --setting generalisability --base-url http://localhost:8000/v1

Re-running the same command resumes: rows already in the output file are skipped.
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from string import Template
from urllib.parse import urlparse

import datasets
import openai
import yaml

HERE = Path(__file__).resolve().parent
SETTINGS = ("core", "generalisability")
TASKS = ("math", "ner", "occurrence", "sentiment", "wsd")
DATA = "cjxhwyt/mip-order"  # on the Hugging Face Hub
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_ATTEMPTS = 3  # per call, as in the paper: a response that fails its check is retried
# Bad key, no credit, no access, unknown model: no retry or later prompt can succeed,
# so the run stops instead of recording every row as a failure.
FATAL_STATUS = {401, 402, 403, 404}


# --- Data -------------------------------------------------------------------

def load(setting, tasks=None, source=DATA):
    """The rows of one setting, optionally only some tasks."""
    ds = datasets.load_dataset(source, setting, split="test")
    if tasks:
        ds = ds.filter(lambda row: row["task"] in tasks)
    return ds


# --- Prompts ----------------------------------------------------------------
# Single-call templates: standard (the paper's main prompt), hint and
# suggested_strategy. self_prompt takes two calls: build_prime_prompt, then
# build_solve_prompt with the instruction the model wrote.

TEMPLATES = yaml.safe_load((HERE / "prompts.yaml").read_text(encoding="utf-8"))
PROMPTS = tuple(TEMPLATES)
SELF_PROMPT = "self_prompt"


def format_instances(instances):
    return "\n\n".join(f"Instance {i}: {text}" for i, text in enumerate(instances, 1))


def build_prompt(task, instances, prompt="standard"):
    """Template, a blank line, then "Instance i: <text>" blocks separated by blank lines."""
    if prompt == SELF_PROMPT:
        raise ValueError("self_prompt takes two calls: use build_prime_prompt and build_solve_prompt")
    # Strip the joined block as a whole, not each instance: this is what the
    # models in the paper saw (it only trims the end of the last instance).
    return TEMPLATES[prompt][task].strip() + "\n\n" + format_instances(instances).strip()


def build_prime_prompt(task, n_instances):
    """self_prompt, call 1: the model writes the solver's instruction without seeing the instances."""
    template = TEMPLATES[SELF_PROMPT]
    text = template["prime"][task].strip() + "\n\n" + template["handoff"].strip()
    return Template(text).safe_substitute(num_instances=n_instances)


def extract_instruction(prime_response):
    """The last <<<NEXT>>> ... <<<END>>> block of call 1, or the whole response if there is none."""
    if not prime_response:
        return ""
    blocks = re.findall(r"<<<NEXT>>>(.*?)<<<END>>>", prime_response, re.DOTALL)
    return blocks[-1].strip() if blocks else prime_response.strip()


def build_solve_prompt(instruction, instances):
    """self_prompt, call 2: the model's instruction, the answer format, then the instances."""
    answer_format = TEMPLATES[SELF_PROMPT]["answer_format"].strip()
    head = Template("$instruction\n\n" + answer_format).safe_substitute(instruction=instruction)
    # Unlike build_prompt, the paper's code did not strip the instance block here.
    return head.strip() + "\n\n" + format_instances(instances)


# --- Answers ----------------------------------------------------------------
# As in the paper: the answer is the "answer" field of the last JSON object in
# the response, compared to the majority label ignoring case and whitespace.

def parse_answer(text):
    """The answer as a string, or None if no JSON object carries one."""
    if not text:
        return None
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates += re.findall(r"\{[^{}]*\}", text, re.DOTALL)  # flat objects only
    candidates.append(text.strip())

    # Latest candidate first: models put the final answer at the end.
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(data, dict) or data.get("answer") is None:
            continue
        answer = str(data["answer"]).strip()
        if (answer.startswith('"') and answer.endswith('"')) or (
            answer.startswith("'") and answer.endswith("'")
        ):
            answer = answer[1:-1]
        return answer
    return None


def is_correct(prediction, majority_label):
    if prediction is None:
        return False
    return prediction.strip().lower() == majority_label.strip().lower()


# --- Model client -----------------------------------------------------------

class Client:
    """Any OpenAI-compatible endpoint: OpenRouter by default, or e.g. a local vLLM server."""

    def __init__(self, base_url=OPENROUTER_URL, api_key=None, timeout=1200, transient_retries=5):
        if api_key is None:
            # The OpenRouter key only ever goes to OpenRouter; other endpoints read
            # API_KEY, and a local vLLM server accepts any key.
            env = "OPENROUTER_API_KEY" if urlparse(base_url).hostname == "openrouter.ai" else "API_KEY"
            api_key = os.getenv(env) or "EMPTY"
        # The SDK's own retries are off: after a timeout it would silently resend a
        # long generation from scratch. Timeouts surface as failed attempts instead.
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key,
                                          timeout=timeout, max_retries=0)
        self.transient_retries = transient_retries

    async def complete(self, model, prompt, *, max_tokens=20000, temperature=0, extra_body=None):
        """One chat completion; rate limits, 5xx and dropped connections are retried with backoff."""
        for k in range(self.transient_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                break
            except openai.APITimeoutError:
                raise
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
                if k == self.transient_retries:
                    raise
                await asyncio.sleep(min(5 * 2 ** k, 120))

        if not resp.choices:  # OpenRouter can return 200 with an error body
            raise RuntimeError(f"empty response: {(resp.model_extra or {}).get('error')}")
        choice = resp.choices[0]
        extra = choice.message.model_extra or {}
        return {
            "content": choice.message.content,
            "reasoning": extra.get("reasoning") or extra.get("reasoning_content"),
            "finish_reason": choice.finish_reason,
            # Which upstream served the call matters: on OpenRouter the provider
            # decides whether a model reasons when no reasoning setting is sent.
            "provider": (resp.model_extra or {}).get("provider"),
            "usage": resp.usage.model_dump() if resp.usage else None,
            "response_id": resp.id,
        }


# --- Running ----------------------------------------------------------------

def has_answer(text):
    return parse_answer(text) is not None


def non_empty(text):
    return bool(text and text.strip())


async def call(client, prompt, args, accept):
    """Up to MAX_ATTEMPTS requests, until accept(response text) holds."""
    attempts, response = [], None
    for _ in range(MAX_ATTEMPTS):
        try:
            response = await client.complete(args.model, prompt, max_tokens=args.max_tokens,
                                             temperature=args.temperature, extra_body=args.extra_body)
        except Exception as e:
            if isinstance(e, openai.APIStatusError) and e.status_code in FATAL_STATUS:
                raise
            attempts.append({"error": f"{type(e).__name__}: {e}"})
            if isinstance(e, openai.BadRequestError):  # e.g. prompt too long; retrying won't help
                break
            continue
        attempts.append({k: response[k] for k in ("finish_reason", "provider", "usage", "response_id")})
        if accept(response["content"]):
            return response, attempts
    return response, attempts  # the last response received, which failed the check


class Instructions:
    """self_prompt's first call. As in the paper, the model writes one instruction per
    task and instance count, and every prompt of that size reuses it."""

    def __init__(self, done_records):
        self.cache = {}
        self.locks = defaultdict(asyncio.Lock)
        for record in done_records:  # on resume, keep the instructions already written
            prime = record.get("prime")
            if prime and prime.get("instruction") is not None:
                self.cache.setdefault((record["task"], record["n_instances"]), prime)

    async def get(self, client, task, n_instances, args):
        key = (task, n_instances)
        async with self.locks[key]:
            if key in self.cache:  # the full first call is stored with the prompt that made it
                return {"instruction": self.cache[key]["instruction"], "reused": True}
            response, attempts = await call(client, build_prime_prompt(task, n_instances), args, non_empty)
            ok = response is not None and non_empty(response["content"])
            prime = {
                "instruction": extract_instruction(response["content"]) if ok else None,
                "content": response["content"] if response else None,
                "reasoning": response["reasoning"] if response else None,
                "attempts": attempts,
                "reused": False,
            }
            if ok:  # a failed first call is tried again by the next prompt
                self.cache[key] = prime
            return prime


async def run_row(client, row, args, instructions):
    record = {key: row[key] for key in ("id", "setting", "task", "n_instances",
                                        "label_distribution", "order_pattern", "majority_label")}
    record.update(model=args.model, prompt=args.prompt)

    response, attempts = None, []
    if args.prompt == SELF_PROMPT:
        prime = await instructions.get(client, row["task"], row["n_instances"], args)
        record["prime"] = prime
        if prime["instruction"] is not None:
            solve_prompt = build_solve_prompt(prime["instruction"], row["instances"])
            response, attempts = await call(client, solve_prompt, args, has_answer)
    else:
        prompt = build_prompt(row["task"], row["instances"], args.prompt)
        response, attempts = await call(client, prompt, args, has_answer)

    prediction = parse_answer(response["content"]) if response else None
    record.update(
        prediction=prediction,
        valid=prediction is not None,
        correct=is_correct(prediction, row["majority_label"]),
        n_attempts=len(attempts),
        # The last response received (the accepted one if any).
        content=response["content"] if response else None,
        reasoning=response["reasoning"] if response else None,
        attempts=attempts,
        request={"base_url": args.base_url, "max_tokens": args.max_tokens,
                 "temperature": args.temperature, "extra_body": args.extra_body},
    )
    return record


async def main(args):
    rows = load(args.setting, args.task, source=args.data)
    model_file = args.model.replace("/", "__") + ".jsonl"
    out = args.out or Path("results") / args.setting / args.prompt / model_file
    out.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if out.exists():
        text = out.read_text()
        for line in text.splitlines():
            try:
                done.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off when a run was killed: that row runs again
        if text and not text.endswith("\n"):
            with open(out, "a") as f:
                f.write("\n")
    done_ids = {record["id"] for record in done}
    todo = [row for row in rows if row["id"] not in done_ids]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{args.model} | {args.setting} | {args.prompt} | "
          f"{len(todo)} to run, {len(done_ids)} already in {out}")
    if not todo:
        return

    client = Client(base_url=args.base_url, timeout=args.timeout)
    try:  # a typo in the model name or a server that is not up would fail every row
        await client.complete(args.model, "Reply with OK.", max_tokens=args.max_tokens,
                              temperature=args.temperature, extra_body=args.extra_body)
    except openai.OpenAIError as e:
        raise SystemExit(f"{args.model} at {args.base_url} does not respond: {e}")

    instructions = Instructions(done)
    semaphore = asyncio.Semaphore(args.concurrency)
    finished = 0
    start = time.time()

    async def worker(row):
        nonlocal finished
        async with semaphore:
            result = await run_row(client, row, args, instructions)
        with open(out, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        finished += 1
        if result["valid"]:
            status = "correct" if result["correct"] else "wrong"
        else:  # say why: the last attempt's error, or how the response ended
            attempts = result["attempts"] or result.get("prime", {}).get("attempts") or [{}]
            last = attempts[-1]
            why = last.get("error") or f"no answer, finish_reason={last.get('finish_reason')}"
            status = f"INVALID ({why[:150]})"
        print(f"[{finished}/{len(todo)} {time.time() - start:.0f}s] {row['id']} "
              f"-> {result['prediction']!r} (truth {row['majority_label']!r}) {status}", flush=True)

    try:
        await asyncio.gather(*(worker(row) for row in todo))
    except openai.APIStatusError as e:
        raise SystemExit(f"Stopped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--prompt", default="standard", choices=PROMPTS,
                    help="prompt template, see prompts.yaml (default: standard)")
    ap.add_argument("--task", nargs="+", choices=TASKS, help="default: all five")
    ap.add_argument("--limit", type=int, help="run at most this many rows (for a quick test)")
    ap.add_argument("--base-url", default=OPENROUTER_URL,
                    help="any OpenAI-compatible endpoint (default: OpenRouter, with OPENROUTER_API_KEY; "
                         "other endpoints read API_KEY)")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--extra-body", type=json.loads,
                    help='JSON passed through to the API, e.g. \'{"provider": {"only": ["parasail"]}}\'')
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200, help="seconds per request")
    ap.add_argument("--data", default=DATA, help="Hub id or local folder of the data")
    ap.add_argument("--out", type=Path, help="default: results/<setting>/<prompt>/<model>.jsonl")
    asyncio.run(main(ap.parse_args()))
