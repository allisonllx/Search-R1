import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests


DEFAULT_MODEL = os.environ.get("VLLM_JUDGE_MODEL", "qwen-judge")
DEFAULT_BASE_URL = os.environ.get("VLLM_JUDGE_BASE_URL", "http://127.0.0.1:8001/v1")


def extract_solution(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_question(solution_str: str) -> str:
    prompt = solution_str.split("<|im_start|>assistant", 1)[0]
    match = re.search(r"Question:\s*(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _as_answer_list(ground_truth: Any) -> list[str]:
    target = ground_truth.get("target", ground_truth) if isinstance(ground_truth, dict) else ground_truth
    if hasattr(target, "tolist"):
        target = target.tolist()
    if isinstance(target, str):
        return [target]
    return [str(item) for item in target]


def build_judge_prompt(question: str, model_answer: str, gold_answers: list[str]) -> str:
    gold_answer = ", ".join(gold_answers)
    return f"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer_list] below. Each answer in the [correct_answer_list] is separated by a comma.
[question]: {question}
[response]: {model_answer}
Your judgment must be in the format and criteria specified below:
extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.
[correct_answer_list]: {gold_answer}
reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer_list], focusing only on if there are meaningful differences between answer in the [correct_answer_list] and the extracted_final_answer. Focus on recall, i.e. if the extracted_final_answer covers all the points in the answer in the [correct_answer_list]. It is ok if it provides more details. It is also ok if the extracted_final_answer misses minor point from the correct_answer, as long as it is evident that they are referring to the same thing. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer_list], focus only on whether the answers match. Ignore capitalization.
correct: Answer 'yes' if extracted_final_answer matches any of the answers in [correct_answer_list] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def _completion_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _cache_path(cache_dir: Optional[str], key: str) -> Optional[Path]:
    if not cache_dir:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(cache_dir).expanduser() / f"{digest}.json"


def _call_openai_compatible(
    prompt: str,
    model: str,
    base_url: str,
    api_key: Optional[str],
    temperature: float,
    timeout: float,
    max_retries: int,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    url = _completion_url(base_url)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"LLM judge request failed after {max_retries + 1} attempts: {last_error!r}")


def _parse_correct(judge_text: str) -> bool:
    match = re.search(r"correct\s*:\s*(yes|no)\b", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return bool(re.search(r"\byes\b", judge_text, re.IGNORECASE)) and not bool(
        re.search(r"\bno\b", judge_text, re.IGNORECASE)
    )


def compute_score_llm_judge(
    solution_str,
    ground_truth,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 3,
    cache_dir: Optional[str] = None,
    score: float = 1.0,
) -> float:
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0

    api_key = api_key or os.environ.get("VLLM_JUDGE_API_KEY")

    question = extract_question(solution_str)
    gold_answers = _as_answer_list(ground_truth)
    prompt = build_judge_prompt(question, answer, gold_answers)

    cache_key = json.dumps(
        {"model": model, "question": question, "answer": answer, "gold": gold_answers},
        ensure_ascii=False,
        sort_keys=True,
    )
    path = _cache_path(cache_dir, cache_key)
    if path and path.exists():
        judge_text = path.read_text(encoding="utf-8")
    else:
        judge_text = _call_openai_compatible(
            prompt=prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(judge_text, encoding="utf-8")

    return score if _parse_correct(judge_text) else 0.0
