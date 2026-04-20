ERROR_CATALOG = [
    # A. Consistency Errors
    {
        "error_code": "consistency_attribute",
        "type": "consistency",
        "prompt_instruction": "Rewrite the response by changing an attribute (like color, count, or size) of one key object."
    },
    {
        "error_code": "consistency_spatial",
        "type": "consistency",
        "prompt_instruction": "Rewrite the response by incorrectly describing the spatial relationship between two objects (e.g., change 'on the table' to 'under the table')."
    },
    {
        "error_code": "consistency_action",
        "type": "consistency",
        "prompt_instruction": "Rewrite the response by describing an incorrect action or state for a subject (e.g., change 'a man is sitting' to 'a man is running')."
    },
    {
        "error_code": "consistency_fake",
        "type": "consistency",
        "prompt_instruction": "Rewrite the response to include a mention of a plausible but non-existent object."
    },
    {
        "error_code": "consistency_misidentification",
        "type": "consistency",
        "prompt_instruction": "Rewrite the response by misidentifying an existing object (e.g., call a 'cup' a 'bowl')."
    },

    # B. Reasoning Errors
    {
        "error_code": "reasoning_conclusion",
        "type": "reasoning",
        "prompt_instruction": "Rewrite the text by making a hasty generalization. Use one small visual detail to derive an overly broad conclusion. Use reasoning words like 'so' or 'therefore'."
    },
    {
        "error_code": "reasoning_causal",
        "type": "reasoning",
        "prompt_instruction": "Confuse correlation with causation. Force a cause-and-effect relationship between two loosely related things using words like 'because' or 'leading to'."
    },
    {
        "error_code": "reasoning_prediction",
        "type": "reasoning",
        "prompt_instruction": "Make an overly confident prediction based on extremely limited information. The prediction should sound possible but be a large logical leap."
    },
    {
        "error_code": "reasoning_procedural",
        "type": "reasoning",
        "prompt_instruction": "Insert a step into a process description that seems plausible but is actually unnecessary or pseudoscientific."
    },
    {
        "error_code": "reasoning_comparison",
        "type": "reasoning",
        "prompt_instruction": "Construct a faulty analogy between two things with only superficial similarity, and draw a misleading conclusion from it."
    },

    # C. External Knowledge Errors
    {
        "error_code": "knowledge_entity",
        "type": "knowledge",
        "prompt_instruction": "If the response mentions a real-world named entity, rewrite it by corrupting that entity (e.g., 'Eiffel Tower in London')."
    },
    {
        "error_code": "knowledge_context",
        "type": "knowledge",
        "prompt_instruction": "Rewrite the response to place an object or scene in a wrong historical or technological context."
    },
    {
        "error_code": "knowledge_definition",
        "type": "knowledge",
        "prompt_instruction": "If the response defines a concept, rewrite it to provide an incorrect definition."
    },
    {
        "error_code": "knowledge_attribution",
        "type": "knowledge",
        "prompt_instruction": "If the response mentions a creation or quote, misattribute it to the wrong source."
    },
]

PROMPT_ANALYZE_TEXT = """
You are a text analysis expert.
Analyze the following text and determine whether it contains:
1. logical reasoning, inference, or conclusion
2. specific external knowledge (people, places, brands, historical facts, etc.)

Respond ONLY with a JSON object:
{{"contains_reasoning": boolean, "contains_knowledge": boolean}}

Text:
"{text}"
""".strip()

PROMPT_CHOOSE_ERROR_SUBTYPE = """
You are a text analysis expert. Your task is to select the single best error-injection strategy.

Available Options:
{options}

Original Text:
"{text}"

Respond ONLY with a JSON object:
{{"best_choice": "one_error_code"}}
""".strip()

PROMPT_GENERATE_LOW_QUALITY = """
Instruction: {instruction}

Original Text:
"{text}"

You MUST output ONLY the corrupted / rewritten text itself.
Do not add explanations, wrappers, or prefixes.
""".strip()

PROMPT_STEP1_MARK_NON_VISUAL = """
Response: {response}

Your task is to precisely insert <INFER> for subjective judgments and <KNOW> for external knowledge.

Critical Guidelines:
1. Tag the shortest complete phrase that expresses the full idea.
2. Tag interpretations of effect, purpose, or reason.
3. Do NOT tag objective visible facts.
4. Do NOT change original wording.

Output Format:
Marked Response: <your_text>

Examples:
Input: The lighting in the room is soft, creating a cozy atmosphere. The design suggests it is from the Victorian era.
Output: Marked Response: The lighting in the room is soft, <INFER>creating a cozy atmosphere</INFER>. <INFER>The design suggests it is from the Victorian era</INFER>.

Input: This is a 1976 postage stamp from Hungary, a country in Central Europe.
Output: Marked Response: This is a 1976 postage stamp from Hungary, <KNOW>a country in Central Europe</KNOW>.

Input: The image shows a can of Coca-Cola.
Output: Marked Response: The image shows a can of Coca-Cola.
""".strip()

PROMPT_STEP2_REMOVE_NON_VISUAL = """
Instruction: {instruction}
Annotated Response: {marked_response}

Task: Process the "Annotated Response" by modifying ONLY the segments wrapped in <INFER>...</INFER> or <KNOW>...</KNOW> tags.

Rules:
1. Rewrite tagged segments into neutral objective visual statements when possible.
2. Delete tagged content if it cannot be grounded visually.
3. Do NOT modify untagged content.
4. Do NOT add new information.
5. Output must start with "Cleaned Response:".

Example:
Cleaned Response: A person wearing sunglasses stands under a tree. Leaves are scattered on the ground.
""".strip()

PROMPT_STEP3_REPHRASE_VISUAL = """
Instruction: {instruction}
Cleaned Response: {cleaned_response}

Task: Rephrase the cleaned response into one fluent, purely visual description.

Rules:
1. Use only information already present.
2. Preserve all visual details.
3. Do NOT add inference or subjective wording.
4. Improve flow and grammar.
5. Output must start with "Visual Summary:".
""".strip()

PROMPT_VISUAL_CONSISTENCY = """
Input Text: {text_input}

Task: Evaluate whether the text description is visually consistent with the image.
Judge consistency, not completeness.

Output Format:
Score: integer 1-5
Explanation: Brief justification.

Rubric:
1 = severely inconsistent
2 = largely inconsistent
3 = partially consistent
4 = mostly consistent with minor issues
5 = fully consistent and accurate
""".strip()

PROMPT_INFERENCE_CORRECTNESS = """
Input Text for Evaluation: {text_to_evaluate}

Task: Evaluate ONLY the statements inside <INFER>...</INFER> tags.
Judge whether the inference is logically plausible from the image.

Output Format:
Score: integer 1-5
Explanation: Brief evaluation.

Rubric:
1 = grossly illogical
2 = major logical gaps
3 = plausible but unprovable
4 = logically sound
5 = logically airtight
""".strip()

PROMPT_EXTERNAL_KNOWLEDGE_CORRECTNESS = """
Input Text for Evaluation: {text_to_evaluate}

Task: Evaluate ONLY the statements inside <KNOW>...</KNOW> tags.
Judge factual correctness using general knowledge.

Output Format:
Score: integer 1-5
Explanation: Brief justification.

Rubric:
1 = entirely incorrect
2 = largely incorrect
3 = partially correct but misleading
4 = mostly correct
5 = fully correct
""".strip()