from typing import List, Dict, Optional


VLM_EVAL_PROMPT_TEMPLATE = """You are an expert AI assistant tasked with evaluating the quality of a response for a given image and instruction.
Please assess the response based on the following holistic criteria:
1. Visual Grounding: Is the response accurately and faithfully describing the visual content of the image?
2. Instruction Following: Does the response directly and completely address the user's instruction?
3. Helpfulness and Coherence: Is the response helpful, logical, and well-written?
Based on your overall assessment, provide a single integer score from 1 (very poor) to 5 (excellent).
Your output should ONLY be this single integer number and nothing else.

Instruction:
{instruction}

Response to Evaluate:
{response}

Score (1-5):
"""


def build_eval_prompt_from_conversations(conversations: List[Dict]) -> Optional[str]:
    """
    从 conversations 里找到最后一个 human -> gpt 对。
    """
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    instruction = None
    response = None

    for i in range(len(conversations) - 1, 0, -1):
        cur = conversations[i]
        prev = conversations[i - 1]

        if cur.get("from") == "gpt" and prev.get("from") == "human":
            instruction = prev.get("value")
            response = cur.get("value")
            break

    if not instruction or not response:
        return None

    return VLM_EVAL_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response=response,
    )