<h2>Introduction</h2>
<p>
Large Vision-Language Models (VLMs) often generate fluent but inaccurate descriptions of images,
a phenomenon known as <b>hallucination</b>. These hallucinations typically appear as incorrect
object counts, colors, or relationships that are not grounded in the visual content. This project
focuses on detecting such hallucinations in VLM-generated image descriptions by combining
linguistic signals (e.g., logits and POS tags), multimodal features, and different modeling
approaches.
</p>

<p>
The project explores three complementary directions for hallucination detection:
</p>
<ul>
  <li>Heuristic methods based on logits and subsequence analysis</li>
  <li>Classical machine learning models using engineered multimodal features</li>
  <li>LLM-based classification with a two-phase prompting strategy that highlights risky words before final prediction</li>
</ul>

<p>
The dataset consists of natural images with VLM-generated descriptions that were manually
annotated for hallucinations, enabling systematic evaluation of detection methods. Overall,
the work aims to better understand when and why VLMs hallucinate and how auxiliary signals
(logits, POS, embeddings, and similarity measures) can help identify unreliable outputs.
</p>

<h2>Project Overview</h2>
<p>
This repository contains the full pipeline for:
</p>
<ul>
  <li>Data collection and preprocessing of VLM descriptions</li>
  <li>Feature extraction from text and images (logits, POS, embeddings, similarity, etc.)</li>
  <li>Training and evaluation of heuristic, ML, and LLM-based hallucination detectors</li>
</ul>

<p>
For a concise visual summary of the methodology, experiments, and results, please refer to the
project poster included in the repository (<code>Poster.pdf</code>).
</p>
