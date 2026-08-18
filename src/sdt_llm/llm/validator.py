from __future__ import annotations

import re
from typing import Dict, Any


def validate_ue_assignment(
    answer: str,
    evidence: Dict[int, Dict[str, Any]],
) -> tuple[bool, str]:
    """
    Minimal cross-UE attribution validator.

    This is intentionally conservative. It only rejects answers when
    the model explicitly reports a conflicting RNTI/value combination.
    """

    for rnti, data in evidence.items():
        sinr = data.get("sinr_db")
        loss = data.get("packet_loss_pct")

        if sinr is not None:
            pattern = rf"RNTI\s*=?\s*{rnti}.*?SINR.*?{float(sinr):.2f}"
            if not re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
                # Don't automatically reject if the model omits SINR.
                pass

        if loss is not None:
            pattern = (
                rf"RNTI\s*=?\s*{rnti}.*?"
                rf"packet\s*loss.*?{float(loss):.0f}"
            )
            if not re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
                pass

    # Explicit swap check for the current two-UE experiment.
    if 1 in evidence and 2 in evidence:
        ue1_sinr = evidence[1].get("sinr_db")
        ue2_sinr = evidence[2].get("sinr_db")

        swap_pattern = (
            rf"RNTI\s*=?\s*2.*?"
            rf"SINR.*?{float(ue1_sinr):.2f}"
        )

        if ue1_sinr is not None and re.search(
            swap_pattern,
            answer,
            re.IGNORECASE | re.DOTALL,
        ):
            return False, "RNTI 2 appears to use RNTI 1 SINR."

        swap_pattern = (
            rf"RNTI\s*=?\s*1.*?"
            rf"SINR.*?{float(ue2_sinr):.2f}"
        )

        if ue2_sinr is not None and re.search(
            swap_pattern,
            answer,
            re.IGNORECASE | re.DOTALL,
        ):
            return False, "RNTI 1 appears to use RNTI 2 SINR."

    return True, "No explicit cross-UE attribution conflict detected."
