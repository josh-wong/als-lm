# SFT failure analysis: instruction tuning a from-scratch 1B model

This document analyzes the failure of supervised fine-tuning (SFT) applied to the ALS-LM 1B model trained from scratch. The analysis connects quantitative evaluation results to the project's research hypotheses, diagnoses root causes, and extracts lessons learned for future domain-specific language model work.

## Hypothesis 3 verdict: REJECTED

The v2 white paper proposed Hypothesis 3: "Instruction tuning can help the from-scratch model surface internalized ALS knowledge in structured Q&A format." This hypothesis is **rejected**.

The 1B model trained from scratch on approximately 153M ALS-domain tokens did not internalize sufficient factual knowledge for SFT to surface. The instruction tuning dataset of 970 validated Q&A pairs was insufficient to teach structured question-answering behavior to a model that lacked foundational language understanding. Rather than producing structured responses, the model collapsed into degenerate token repetition, generating "TheTheThe..." output for all 160 benchmark questions.

The v1 research paper (Section 9.2) explicitly recommended "a more modern instruction-capable base model" as the principled next step for instruction tuning. This SFT failure validates that recommendation: instruction tuning a from-scratch model with limited data and limited pre-existing knowledge produces worse output than the base model, not better.

## Quantitative results

The instruction-tuned model was evaluated through the same 6-stage hallucination evaluation pipeline used for all other models. The results are unambiguous.

| Model                 | Accuracy | Non-degenerate | Degenerate    | Capability gap |
| --------------------- | -------: | -------------: | ------------- | -------------: |
| 500M (from-scratch)   |    0.21% |          67.5% | 52/160        |          67.3% |
| GPT-2 large (tuned)   |    3.12% |           2.5% | 156/160       |          -0.6% |
| 1B base (from-scratch)|    0.00% |          35.0% | 104/160       |          35.0% |
| 1B instruct (SFT)     |    0.00% |           0.0% | 160/160       |           0.0% |

The 1B instruction-tuned model achieves 0.00% mean accuracy with 160/160 degenerate responses. The qualitative spot-check during Phase 45 confirmed 0/8 coherent responses across all 8 ALS knowledge categories (clinical trials, diagnostic criteria, disease mechanisms, drug treatment, epidemiology, gene mutation, patient care, temporal accuracy).

**Training metrics:** 970 instruction pairs (776 train, 194 val), 2 epochs over 290 steps, learning rate 2e-5 with 50-step warmup, effective batch size 8. Training loss decreased from 9.0 to 4.7 and validation loss reached 4.8782 at the best checkpoint (epoch 1, step 290). The model optimized the loss objective but the learned behavior is degenerate.

## Failure mode analysis

The dominant failure mode is concatenated token repetition: the model generates "TheTheThe..." for every input, regardless of the question content or category. This is not gradual degradation where the model starts with partial coherence and deteriorates; it is immediate collapse where the very first generated token is "The" and every subsequent token is also "The".

The word "The" is the most common English word and the most frequent token in the training corpus. The model has essentially collapsed to a maximum-likelihood single-token distribution, repeating the highest-probability token indefinitely. This pattern is consistent across all 8 ALS knowledge categories, all difficulty levels (easy, medium, hard), and all question types.

The is_coherent filter in the evaluation pipeline correctly classifies all 160 responses as non-coherent. The taxonomy classifier assigns all 160 responses to the "degenerate" category with "low" severity, indicating that while the output is useless, it does not pose the ethical risk of confident-but-wrong medical information.

Critically, the 1B base model without SFT produces 56/160 non-degenerate responses (35.0%). While none of these achieve factual accuracy (0.00% mean accuracy), the base model at least generates diverse text patterns including responses classified as confident fabrication (19), plausible blending (29), and outdated information (8). The SFT process destroyed this diversity rather than redirecting it toward accurate answers.

## Root cause analysis

Three compounding factors explain the failure.

**Data scale.** The 970 instruction pairs are one to two orders of magnitude below published SFT dataset sizes. Stanford Alpaca used 52,000 instruction pairs. Databricks Dolly used 15,000 human-written pairs. Even smaller published SFT datasets like LIMA (1,000 pairs) relied on a pre-trained 65B parameter model with extensive general knowledge. The combination of 970 pairs with a model that has no general language understanding is far below the minimum viable threshold.

**Base model quality.** The 1B model was trained from scratch on approximately 153M domain-specific tokens. It has no general English understanding, no exposure to WebText or any broad internet corpus, and no pre-existing ability to follow instructions. Pre-trained models like GPT-2 or LLaMA bring billions of tokens of general knowledge that SFT can redirect toward specific tasks. Without this foundation, SFT must simultaneously teach language understanding, task format, and domain knowledge from only 970 examples.

**Compounding knowledge deficit.** The base model's 0.00% benchmark accuracy (not merely an instruction-following deficit but a complete knowledge deficit) means there is no internalized knowledge for SFT to surface. The hypothesis assumed the model had learned factual content during pre-training that it simply could not express in Q&A format. The 4-model comparison disproves this: the 1B base model produces diverse but universally inaccurate responses, indicating it learned statistical patterns in ALS text but not extractable medical facts. SFT cannot create knowledge that does not exist in the base model.

## Lessons learned

**SFT cannot create knowledge; it can only format existing knowledge.** This is the most important takeaway. The instruction tuning process is designed to teach a model how to respond to instructions by reformatting knowledge it already possesses. When the base model has no factual knowledge to reformat, SFT has nothing to work with and collapses to degenerate output.

**From-scratch SFT requires orders of magnitude more data than fine-tuning pre-trained models.** A pre-trained model can be instruction-tuned with as few as 1,000 high-quality examples (LIMA) because it already understands language. A from-scratch model would need to learn language, domain knowledge, and instruction following simultaneously, requiring vastly more data.

**Token repetition is a known collapse mode for undertrained SFT.** When the loss landscape offers no gradient signal toward meaningful responses (because the model lacks the representational capacity to produce them), the model converges to the maximum-likelihood single-token distribution. This is a well-documented failure mode in the SFT literature.

**The 4-model comparison tells a consistent story.** Pre-trained knowledge (GPT-2 large fine-tuned, 3.12% accuracy) helps far more than model scale (500M to 1B, both near 0%) or instruction tuning on a from-scratch model (0.00% with complete output collapse). The hierarchy is clear: pre-trained knowledge > model scale > instruction tuning without pre-trained knowledge.

**Training loss reduction does not guarantee useful behavior.** The SFT training loss decreased from 9.0 to 4.7, a substantial reduction that would normally indicate successful learning. However, the model "learned" to produce the same degenerate output for every input, which minimizes cross-entropy loss by concentrating probability mass on the most common token. Loss curves alone cannot distinguish between meaningful learning and degenerate convergence.

## Perceived capability gap interpretation

The capability gap metric (coherence% minus accuracy%) is designed to quantify ethical risk: models that produce coherent-sounding but factually incorrect output are potentially dangerous because users may trust their responses. The 1B instruction-tuned model has a capability gap of 0.0% because it produces 0% coherent output and 0% accurate output. The ethical risk of confident-but-wrong medical information did not materialize because the output is obviously broken.

This is a null result rather than a positive safety finding. The ethical framework is designed for models that achieve partial coherence, where the gap between appearing knowledgeable and being accurate creates real risk. The 500M from-scratch model illustrates this concern with a 67.3% capability gap (67.5% coherent output but only 0.21% accuracy). The GPT-2 large model has a negative gap (-0.6%) because its accuracy slightly exceeds its coherence rate, though both values are very low.

For the 1B instruct model, the absence of capability gap does not indicate safety; it indicates that the model failed too completely for the ethical risk framework to apply. A model that cannot produce coherent sentences cannot mislead users, but it also cannot serve any useful purpose.

## Implications for future work

**SFT of the GPT-2 large fine-tuned model.** This would be the principled next experiment. The GPT-2 large model has pre-trained general knowledge from WebText, domain-specific knowledge from ALS fine-tuning, and achieves 3.12% accuracy, indicating some internalized knowledge that SFT could potentially surface in Q&A format. The same 970 instruction pairs might be sufficient for this model because SFT would be redirecting existing knowledge rather than creating it.

**Larger instruction datasets.** For from-scratch models, instruction datasets of 5,000 to 50,000 pairs would be needed. These could be generated through more diverse corpus extraction strategies, including extracting Q&A pairs from different sections of research papers, clinical guidelines, and educational materials rather than relying on a single extraction pipeline.

**Modern instruction-capable architectures.** The v1 research paper's recommendation of a "more modern instruction-capable base model" remains the strongest path forward. Models trained on diverse internet text with instruction-following capabilities built into the pre-training objective (e.g., models from the LLaMA or Mistral families) would provide the foundational language understanding that this project's from-scratch models lack.

## Source data references

The quantitative data in this analysis is drawn from these evaluation artifacts.

- `results/sft/verify_results.json` -- Phase 45 spot-check (0/8 coherent, FAIL verdict)
- `results/sft/sft_summary.md` -- Training metrics (970 pairs, 2 epochs, loss 9.0 to 4.4)
- `eval/results/als-lm-1b-instruct_q8_0/scores.json` -- Benchmark accuracy (0.00% overall)
- `eval/results/als-lm-1b-instruct_q8_0/taxonomy.json` -- Failure taxonomy (160/160 degenerate)
- `reports/model_comparison_report.json` -- 4-model comparison with capability gap

---

*This is a research analysis document. The model analyzed here is a research artifact and should never be used for medical decision-making.*
