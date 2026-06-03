# S-MAPE Project Positioning

## Final Landing Point

S-MAPE is no longer framed as a generic automatic prompt rewriting system. Its final landing point is:

> Red-blue self-play for adversarial reasoning robustness, where LLM failures are converted into structured, reusable, and prunable prompt patch memory under a clean-accuracy guard.

In code terms, the system should follow this chain:

```text
clean sample
-> red-team adversarial mutation
-> target-model failure
-> blue-team failure diagnosis
-> structured candidate prompt patch
-> adversarial replay + clean guard
-> meta-controller decision
-> patch memory update / merge / prune
-> final prompt assembled from base prompt + active patches
```

The key shift is from:

```text
old_prompt -> rewritten_prompt
```

to:

```text
failure -> diagnosis -> candidate_patch -> evaluated_patch -> patch_memory -> assembled_prompt
```

## Research Question

Chinese:

如何通过红蓝对抗自博弈，将大语言模型在对抗性推理样本上的失败转化为可解释、可复用、可剪枝的 prompt patch memory，从而在不更新模型参数的情况下提升鲁棒性，并控制 clean accuracy 与 prompt 长度成本？

English:

How can red-blue self-play transform LLM failures on adversarial reasoning samples into interpretable, reusable, and prunable prompt patch memory, improving robustness without updating model parameters while preserving clean accuracy and prompt compactness?

One-sentence abstract version:

We propose S-MAPE, a red-blue self-play framework that converts adversarial reasoning failures into structured prompt patches and selectively merges them into memory under clean-accuracy, redundancy, conflict, and token-budget constraints.

Introduction-end problem statement:

This work studies patch-level prompt evolution for adversarial reasoning robustness: given a base prompt, a clean dataset, a target LLM, and adaptive red-team perturbations, the goal is to learn a compact set of local defense patches that improve adversarial accuracy while preserving clean-task performance and interpretability.

## Core Innovations

1. Red-Blue Self-Play for Prompt Robustness

   The red team actively generates adversarial reasoning samples, while the blue team reacts only to observed failures. This turns prompt optimization into an adversarial robustness process rather than a one-shot rewrite process.

2. Failure-Conditioned Prompt Patch Memory

   Each failure is mapped to a structured patch with fields such as failure type, trigger condition, defense rule, source examples, estimated adversarial gain, clean risk, redundancy, conflict, token cost, and status. The memory is reusable, auditable, and prunable.

3. Multi-Agent Blue-Team Diagnosis and Patch Critique

   The blue team is decomposed into failure diagnosis, patch generation, patch criticism, and prompt editing roles. This gives the system a defensible mechanism for moving from observed errors to localized defense rules.

4. Meta-Controller with Clean-Accuracy Guard

   A deterministic controller decides whether to accept, merge, reject, or prune patches using adversarial replay gain, clean accuracy drop, token cost, redundancy, conflict risk, specificity, generality, and interpretability.

## What This Project Should Claim

The strongest paper claim is not "we automatically optimize prompts." The stronger claim is:

> We introduce patch-level prompt evolution, where robustness improvements are represented as local, typed, and testable prompt patches rather than unconstrained prompt rewrites.

This makes the project closer to adversarial robustness, prompt memory, and reasoning failure analysis than to ordinary prompt engineering.
