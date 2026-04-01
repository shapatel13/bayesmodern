from __future__ import annotations

import streamlit as st

from security.secrets import validate_live_llm_config
from utils.config import get_settings


def main() -> None:
    settings = get_settings()
    secret_status = validate_live_llm_config(settings)

    st.set_page_config(page_title="PRIORI-X Research Console", page_icon="PX", layout="wide")
    st.title("PRIORI-X")
    st.caption("Clinician-facing Bayesian reasoning lab for offline evaluation and decision-science research.")

    st.warning(
        "Research only. PRIORI-X is not a clinical autopilot and must not be used for autonomous "
        "patient care decisions."
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Operational Posture")
        st.markdown(
            "\n".join(
                [
                    f"- Safety mode: `{settings.safety_mode}`",
                    f"- Live external LLMs enabled: `{settings.allow_live_llm}`",
                    f"- Provider ready: `{secret_status.provider_ready}`",
                    f"- Trace redaction: `{settings.redact_traces}`",
                    f"- Experiment namespace: `{settings.experiment_namespace}`",
                ]
            )
        )
    with right:
        st.subheader("Current Focus")
        st.markdown(
            "\n".join(
                [
                    "- Bayesian differential ranking",
                    "- next-best-test reasoning",
                    "- uncertainty and calibration",
                    "- offline benchmark rollouts",
                    "- safety and stewardship analysis",
                ]
            )
        )


if __name__ == "__main__":
    main()

