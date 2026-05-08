# **Combining Boundary-Aware Architectural Supervision and Losses in Transformer Segmentation**

## Abstract

Boundary-aware loss functions and architectures have shown to improve semantic segmentation at object boundaries; however, they have developed largely in parallel, leaving an open question regarding whether these interventions are complements or substitutes. We answer this question through a controlled comparison, holding the backbone Segformer-B0 and dataset ADE20K fixed while introducing two boundary-aware interventions: modifying segmentation loss with Boundary Difference-over-Union (B-DoU) and architecture with the presence of a lightweight auxiliary boundary prediction head. Across five training configurations, we evaluate on mean Intersection-over-Union (mIoU) as well as Boundary F1 (BF1) at four tolerances and Boundary IoU (BIoU). We find that our two boundary-aware interventions are complements: their combined configuration leads on every metric, improving over a cross-entropy fine-tuned baseline by +2.33 on BF1 at tolerance τ = 3 pixels, +0.55 on BIoU, and +0.52 on mIoU. The gains on BF1 are largest at the smallest boundary tolerance and shrink monotonically as tolerance increases , indicating improvement at the pixel-level. Our results also support reporting boundary-aware metrics alongside mIoU to have a more comprehensive view of segmentation performance.

## Acceptable Use Policy

This code and any model checkpoints derived from it are released for research purposes. **We explicitly prohibit use of this work in surveillance applications.** We also do not recommend deployment in high-stakes settings such as medical imaging or autonomous driving without revalidation on domain-appropriate data.

## License

Released under the MIT License. See `LICENSE` for details.
