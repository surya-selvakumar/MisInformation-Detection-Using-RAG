from typing import List

SYSTEM_TASK = (
    "Task: Determine whether the claim is True, False, or Not Enough Information "
    "based on the evidence. Then provide a brief explanation citing the evidence."
)

def build_prompt(claim: str, evid_spans: List[str], restrict_binary: bool=False) -> str:
    header = SYSTEM_TASK
    if restrict_binary:
        header = header.replace(" or Not Enough Information", "")
    lines = [header, f"Claim: {claim}"]
    for i, ev in enumerate(evid_spans, 1):
        lines.append(f"Evidence [{i}]: {ev}")
    lines.append("Format your answer as: Prediction: <True/False/Not Enough Information>."
                 " Explanation: <2--3 sentences with citations like [1], [2]>")
    return "\n".join(lines)

# label tagging spans to weight label tokens in loss
def tag_label(answer_label: str, explanation: str) -> str:
    return f"Prediction: <LABEL>{answer_label}</LABEL>. Explanation: {explanation}"
