from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from copy import copy

from openai import OpenAI



@dataclass
class SESConfig:
    # conversation control
    continue_turn: int = 2
    begin_turn: int = 1
    vote_num: int = 10
    sample_times: int = 3
    po_sample_times: int = 3
    po_vote_num: int = 10


    # model & decoding
    model: str = "xxx" # using the deployed model
    rec_temp_first: float = 0.7
    rec_temp_normal: float = 0.1
    user_temp_normal: float = 0.1
    user_temp_vote: float = 0.7
    max_tokens: int = 2048

    # retries
    retries: int = 3
    backoff_sec: float = 1.0



def create_client() -> Any:
    """Create an OpenAI-compatible client from environment variables.

    Required env vars:
      - OPENAI_API_KEY
    Optional:
      - OPENAI_BASE_URL
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please `pip install openai`." )

    api_key="xxxxxxx"
    base_url="xxxxx"
    
    

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it in your environment before running."
        )

    # base_url can be None for official API; keep if user wants a custom endpoint
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client



def load_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_replace(template: str, mapping: Dict[str, str]) -> str:
    out = template
    if "profile" in mapping and "{proflie}" in out and "{profile}" not in out:
        out = out.replace("{proflie}", "{profile}")
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out



def history2messages(history: List[str]) -> List[Dict[str, str]]:
    """Convert alternating seeker/recommender utterances to chat messages.

    Index even -> seeker (user), index odd -> recommender (assistant).
    If the history length is odd (ends with seeker), we keep it as-is.
    """
    messages: List[Dict[str, str]] = []
    for i, utter in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utter})
    return messages


def _chat_with_retry(client: Any, *, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, retries: int, backoff_sec: float) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = getattr(resp.choices[0].message, "content", None)
            if not content:
                raise RuntimeError("Empty content from model response")
            return content
        except Exception as e:  # noqa: PERF203
            last_err = e
            # exponential backoff
            time.sleep(backoff_sec * (2 ** attempt))
    # if here, all attempts failed
    raise RuntimeError(f"Model call failed after {retries} retries: {last_err}")


def get_preference(client: Any, history: List[str], config: SESConfig) -> str:
    pref_template = load_prompt("prompt_template/user_prof_sum_prompt.md")

    history_string = "\n"
    for idx, utt in enumerate(history):
        if idx % 2 == 0:
            history_string += f"Seeker: {utt}\n"
        else:
            history_string += f"Recommender: {utt}\n"

    llm_input = pref_template.replace("{chat_log}", history_string)

    content = _chat_with_retry(
        client,
        model=config.model,
        messages=[{"role": "user", "content": llm_input}],
        temperature=config.user_temp_vote,  # use a bit higher temp for summarization
        max_tokens=config.max_tokens,
        retries=config.retries,
        backoff_sec=config.backoff_sec,
    )
    return content


_num_re = re.compile(r"\b([0-2])\b")


def _parse_vote_score(text: str) -> Optional[int]:
    # extract a single integer 0/1/2 from text; return None if not present
    if text is None:
        return None
    m = _num_re.search(text.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _simulate_internal_and_score(
    client: Any,
    history: List[str],
    query: str,
    pref: str,
    rec_prompt: str,
    user_internal_prompt: str,
    rec_simu_last_turn_prompt: str,
    user_last_turn_prompt: str,
    continue_utter: int,
    utter_i: int,
    config: SESConfig,
    tree: bool = False,
    is_root: bool = True,
    sample_score_list: List[float] = [],
    depth = 0,
) -> Tuple[str, float]:
    """Run internal simulations and return best first response and its score."""
    internal_results: List[Dict[str, Any]] = []
    query_org = copy(query)
    
    for _sample in range(config.sample_times):
        internal_history = copy(history)
        first_resp: Optional[str] = None
        simulate_utters = continue_utter - utter_i
        query = query_org
        if is_root:
            
            sample_score_list: List[float] = []
            depth = 0
        print('Sample', _sample, '-depth', depth)
        for inter_utter_i in range(simulate_utters):
            # recommender turn
            if inter_utter_i % 2 == 0:
                
                if tree and inter_utter_i>0:
                    # recursive call for tree-based SES
                    sample_score_list = _simulate_internal_and_score(
                        client,
                        history=internal_history,
                        query=query,
                        pref=pref,
                        rec_prompt=rec_prompt,
                        user_internal_prompt=user_internal_prompt,
                        rec_simu_last_turn_prompt=rec_simu_last_turn_prompt,
                        user_last_turn_prompt=user_last_turn_prompt,
                        continue_utter=continue_utter,
                        utter_i=inter_utter_i+utter_i,
                        config=config,
                        tree=tree,
                        is_root=False,
                        sample_score_list=sample_score_list,
                        depth=depth+1,
                    )
                    break
                    
                else:
                    temperature = config.rec_temp_first if inter_utter_i == 0 else config.rec_temp_normal
                    messages = (
                        [{"role": "system", "content": rec_prompt}]
                        + history2messages(internal_history)
                        + [{"role": "user", "content": query}]
                    )
                    resp_rec = _chat_with_retry(
                        client,
                        model=config.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=config.max_tokens,
                        retries=config.retries,
                        backoff_sec=config.backoff_sec,
                    )
                    
                    if inter_utter_i == 0:
                        first_resp = resp_rec
                    if inter_utter_i == simulate_utters - 2:
                        resp_rec = resp_rec + rec_simu_last_turn_prompt
                    internal_history.append(query)
                    query = resp_rec 
            else:
                # internal seeker turn
                is_last_scoring_turn = (inter_utter_i == simulate_utters-1)
                if not is_last_scoring_turn:
                    messages = (
                        [{"role": "system", "content": user_internal_prompt}]
                        + history2messages(internal_history)
                        + [{"role": "user", "content": query}]
                    )
                    resp_user = _chat_with_retry(
                        client,
                        model=config.model,
                        messages=messages,
                        temperature=config.user_temp_normal,
                        max_tokens=config.max_tokens,
                        retries=config.retries,
                        backoff_sec=config.backoff_sec,
                    )
                    if inter_utter_i == continue_utter - utter_i - 3:
                        resp_user = resp_user + user_last_turn_prompt
                    internal_history.append(query)
                    query = resp_user
                else:
                    # the last turn to score
                    votes: List[int] = []
                    vote_messages = (
                        [{"role": "system", "content": user_internal_prompt}]
                        + history2messages(internal_history)
                        + [{"role": "user", "content": query}]
                    )
                    text_ls = []
                    for _ in range(config.vote_num):
                        text = _chat_with_retry(
                            client,
                            model=config.model,
                            messages=vote_messages,
                            temperature=config.user_temp_vote,
                            max_tokens=config.max_tokens,
                            retries=config.retries,
                            backoff_sec=config.backoff_sec,
                        )
                        text_ls.append(text)                    
                        score = _parse_vote_score(text)
                        if score is not None:
                            votes.append(score)
                    print('    score: ', votes)
                    avg_score_i = (sum(votes) / len(votes)) if votes else 0.0
                    internal_history.append(query)
                    internal_history.append(str(avg_score_i))
                    sample_score_list.append(avg_score_i)

                        
        if is_root:        
            print('  sample_score_list:', sample_score_list)
            internal_results.append({
                "response": first_resp or "",
                "score": sum(sample_score_list) / len(sample_score_list) if sample_score_list else 0.0,
            })


    if is_root:
        if not internal_results:
            return "", 0.0
        best = max(internal_results, key=lambda x: x["score"])  # type: ignore[arg-type]
        return best["response"], float(best["score"])  # type: ignore[index]
    else:
        return sample_score_list
        


def eval_conversations(client: Any, data: List[Dict[str, Any]], config: SESConfig, ses: bool, tree:bool) -> Dict[str, Any]:
    """Run SES evaluation for a dataset.

    Returns a dict with scores per conversation and aggregated info.
    """
    print('Starting evaluation with SES=', ses)
    score_log: Dict[str, Any] = {}
    continue_utter = 2 * config.continue_turn 

    # pre-load templates
    user_prompt_tpl = load_prompt("prompt_template/user_simu_eval_prompt.md")
    user_last_turn_prompt = load_prompt("prompt_template/user_simu_eval_last_turn_prompt.md")
    rec_prompt = load_prompt("prompt_template/rec_sys_prompt.md")
    rec_eval_last_turn_tpl = load_prompt("prompt_template/rec_sys_eval_last_turn.md")
    rec_simu_last_turn_prompt = load_prompt("prompt_template/rec_sys_internal_last_turn_prompt.md")
    user_internal_prompt_tpl = load_prompt("prompt_template/user_simu_internal_prompt.md")

    for data_i in range(len(data)):
        print(f"Evaluating conversation {data_i + 1}/{len(data)}")
        item = data[data_i]
        history = item["context"][:-1]  # remove the last recommender utterance
        query = item["context"][-1]
        label = item.get("resp")
        rec_items = item.get("rec", [])

        # render prompts that depend on rec items
        user_prompt = safe_replace(user_prompt_tpl, {"rec": ", ".join(rec_items)})
        rec_eval_last_turn_prompt = safe_replace(rec_eval_last_turn_tpl, {"rec": ", ".join(rec_items)})

        # compute user preference summary for internal simulation


        for utter_i in range(continue_utter):
            print('utter_i of data_i:', utter_i, data_i)
            is_rec_turn = (utter_i % 2 == 0)
            if is_rec_turn:
                if ses and (utter_i // 2 >= config.begin_turn):
                    try:
                        pref = get_preference(client, history, config)
                    except Exception as e:
                        print(f"Preference summarization failed: {e}")
                        pref = ""
                    user_internal_prompt = safe_replace(user_internal_prompt_tpl, {"profile": pref})
                    try:
                        best_resp, best_score = _simulate_internal_and_score(
                            client,
                            history=history,
                            query=query,
                            pref=pref,
                            rec_prompt=rec_prompt,
                            user_internal_prompt=user_internal_prompt,
                            rec_simu_last_turn_prompt=rec_simu_last_turn_prompt,
                            user_last_turn_prompt=user_last_turn_prompt,
                            continue_utter=continue_utter,
                            utter_i=utter_i,
                            config=config,
                            tree=tree,
                        )
                    except Exception as e:
                        print(f"Internal simulation failed: {e}")
                        best_resp, best_score = "", 0.0
                else:
                    # if not SES, just generate once with normal rec pipeline
                    messages = (
                        [{"role": "system", "content": rec_prompt}]
                        + history2messages(history)
                        + [{"role": "user", "content": query}]
                    )
                    try:
                        best_resp = _chat_with_retry(
                            client,
                            model=config.model,
                            messages=messages,
                            temperature=config.rec_temp_normal,
                            max_tokens=config.max_tokens,
                            retries=config.retries,
                            backoff_sec=config.backoff_sec,
                        )

                    except Exception as e:
                        print(f"Recommender turn generation failed: {e}")


                if utter_i == continue_utter - 2:
                    best_resp = best_resp + rec_eval_last_turn_prompt
                history.append(query)
                query = best_resp

            else:
                # seeker turn
                messages = (
                    [{"role": "system", "content": user_prompt}]
                    + history2messages(history)
                    + [{"role": "user", "content": query}]
                )
                try:
                    resp_user = _chat_with_retry(
                        client,
                        model=config.model,
                        messages=messages,
                        temperature=config.user_temp_normal,
                        max_tokens=config.max_tokens,
                        retries=config.retries,
                        backoff_sec=config.backoff_sec,
                    )
                    # print('resp_user:', resp_user)
                except Exception as e:
                    print(f"User turn generation failed: {e}")
                    resp_user = ""
                if utter_i == continue_utter - 3:
                    resp_user = resp_user + user_last_turn_prompt
                history.append(query)
                if utter_i == continue_utter - 1:
                    conv_score = _parse_vote_score(resp_user)
                    history.append(str(conv_score))
                query = resp_user

        score_log[str(data_i)] = {
            "resp_label": label,
            "rec_label": rec_items,
            "score": conv_score,
            "history": history,
        }
        print('score_log:', score_log[str(data_i)])
        

    return score_log

def eval_conversations_with_voting(client: Any, data: List[Dict[str, Any]], config: SESConfig) -> Dict[str, Any]:
    """Run SES evaluation for a dataset.

    Returns a dict with scores per conversation and aggregated info.
    """
    score_log: Dict[str, Any] = {}
    continue_utter = 2 * config.continue_turn 

    # pre-load templates
    user_prompt_tpl = load_prompt("prompt_template/user_simu_eval_prompt.md")
    user_last_turn_prompt = load_prompt("prompt_template/user_simu_eval_last_turn_prompt.md")
    rec_prompt = load_prompt("prompt_template/rec_sys_prompt.md")
    rec_eval_last_turn_tpl = load_prompt("prompt_template/rec_sys_eval_last_turn.md")
    rec_simu_last_turn_prompt = load_prompt("prompt_template/rec_sys_internal_last_turn_prompt.md")
    user_internal_prompt_tpl = load_prompt("prompt_template/user_simu_internal_prompt.md")

    for data_i in range(len(data)):
        print(f"Evaluating conversation {data_i + 1}/{len(data)}")
        item = data[data_i]
        history_org = item["context"][:-1]  # remove the last recommender utterance
        query_org = item["context"][-1]
        label = item.get("resp")
        rec_items = item.get("rec", [])
        resp_label = item.get("resp")
        # render prompts that depend on rec items
        user_prompt = safe_replace(user_prompt_tpl, {"rec": ", ".join(rec_items)})
        rec_eval_last_turn_prompt = safe_replace(rec_eval_last_turn_tpl, {"rec": ", ".join(rec_items)})

        # compute user preference summary for internal simulation

        for _ in range(config.po_sample_times):
            sample_results: List[Dict[str, Any]] = []
            query = copy(query_org)
            history = copy(history_org)
            first_resp: Optional[str] = None
            for utter_i in range(continue_utter):
                print('utter_i of data_i:', utter_i, data_i)
                is_rec_turn = (utter_i % 2 == 0)
                if is_rec_turn:

                    messages = (
                        [{"role": "system", "content": rec_prompt}]
                        + history2messages(history)
                        + [{"role": "user", "content": query}]
                    )
                    try:
                        rec_resp = _chat_with_retry(
                            client,
                            model=config.model,
                            messages=messages,
                            temperature=config.rec_temp_normal,
                            max_tokens=config.max_tokens,
                            retries=config.retries,
                            backoff_sec=config.backoff_sec,
                        )

                    except Exception as e:
                        print(f"Recommender turn generation failed: {e}")

                    if utter_i == 0:
                        first_resp = rec_resp
                    if utter_i == continue_utter - 2:
                        rec_resp = rec_resp + rec_eval_last_turn_prompt
                    history.append(query)
                    query = rec_resp

                else:
                    # seeker turn
                    messages = (
                        [{"role": "system", "content": user_prompt}]
                        + history2messages(history)
                        + [{"role": "user", "content": query}]
                    )
                    if utter_i == continue_utter - 1:
                        # voting for the last seeker turn
                        votes: List[int] = []
                        vote_messages = copy(messages)
                        for _ in range(config.po_vote_num):
                            text = _chat_with_retry(
                                client,
                                model=config.model,
                                messages=vote_messages,
                                temperature=config.user_temp_vote,
                                max_tokens=config.max_tokens,
                                retries=config.retries,
                                backoff_sec=config.backoff_sec,
                            )
                            score = _parse_vote_score(text)
                            if score is not None:
                                votes.append(score)
                        print('    score: ', votes)
                        avg_score_i = (sum(votes) / len(votes)) if votes else 0.0

                    else:
                        try:
                            resp_user = _chat_with_retry(
                                client,
                                model=config.model,
                                messages=messages,
                                temperature=config.user_temp_normal,
                                max_tokens=config.max_tokens,
                                retries=config.retries,
                                backoff_sec=config.backoff_sec,
                            )
                            # print('resp_user:', resp_user)
                        except Exception as e:
                            print(f"User turn generation failed: {e}")
                            resp_user = ""
                            
                        if utter_i == continue_utter - 3:
                            resp_user = resp_user + user_last_turn_prompt
                    history.append(query)
                    query = resp_user
                        
            sample_results.append({
                "response": first_resp or "",
                "avg_score": avg_score_i,
            })
        
        #select the highest scored response among samples
        win_resp = resp_label
        win_sample_candid = max(sample_results, key=lambda x: x["avg_score"])
        win_resp_candid = win_sample_candid["response"]
        win_score_candid = win_sample_candid["avg_score"]
        if win_score_candid >= 2.0:
            win_resp = win_resp_candid
        # select the loest scored response among samples
        loose_resp = resp_label
        loose_sample_candid = min(sample_results, key=lambda x: x["avg_score"])
        loose_resp_candid = loose_sample_candid["response"]
        loose_score_candid = loose_sample_candid["avg_score"]
        if loose_score_candid < 2.0:
            loose_resp = loose_resp_candid
        score_log[str(data_i)] = {
            "resp_label": label,
            "rec_label": rec_items,
            "win_resp": win_resp,
            "win_score": win_score_candid,
            "loose_resp": loose_resp,
            "loose_score": loose_score_candid,
            # "history": history,
        }
        print('score_log:', score_log[str(data_i)])
        

    return score_log
__all__ = [
    "SESConfig",
    "create_client",
    "load_data",
    "load_prompt",
    "history2messages",
    "eval_conversations",
]
