"""
sdt_llm — Semantic Digital Twin + LLM Inference
=================================================

A reference implementation of the pipeline described in:

  Chen, Ge, Zhang, Shi, Wei (Huawei Wireless Advanced System Competency
  Centre), "Semantic Digital Twins: Enhancing Performance in Wireless
  Communication and LLM Inference."
  https://www.huawei.com/en/huaweitech/future-technologies/semantic-digital-twins-wireless-communication-llm-inference

Paper pipeline:      vision  -> SDT -> LLM inference
This project adds:   6G radio (ISAC/CSI, Sionna RT-shaped) -> SDT -> LLM inference

See README.md for the full mapping between paper sections/equations and code.
"""

__version__ = "0.1.0"
