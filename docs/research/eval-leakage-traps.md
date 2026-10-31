# Deep Research: eval-leakage-traps

- **Agent**: `deep-research-max-preview-04-2026`
- **Interaction ID**: `v1_ChduZDdsYXFuNk9adWp2ZElQbDRlUGtRRRIXbmQ3bGFxbjZPWnVqdmRJUGw0ZVBrUUU`
- **Query**: Evaluation methodology and data leakage traps when evaluating zero-human-label VLM pseudo-labeling, selective prediction (abstention / risk-coverage curves / AURC), and student model distillation on surgical video datasets such as CholecSeg8k and Cholec80.

---

# Evaluation Methodology and Data Leakage Traps in Surgical Vision-Language Models

**MEDICAL AND CLINICAL SAFETY DISCLAIMER:** *The methodologies, metrics, and technical evaluations discussed in this report are for academic, research, and informational purposes only. This analysis does not constitute validated clinical guidelines, and the artificial intelligence systems described herein have not necessarily been cleared by regulatory bodies (e.g., the FDA) for direct surgical deployment or independent diagnostic use in a live operating room environment.*

## Executive Summary

*   **Temporal Autocorrelation and Splitting:** Evaluating surgical videos at the frame level introduces severe temporal autocorrelation leakage. Because adjacent frames are highly correlated, models memorize patient-specific artifacts rather than learning generalized surgical features, necessitating strict video-level or patient-level dataset partitioning.
*   **CholecSeg8k Structural Traps:** The CholecSeg8k dataset contains undocumented watershed mask pixel-mapping artifacts (RGB channel repetition), phantom classes caused by anti-aliasing software interpolation, and specific RGBA formatting errors. Navigating this requires explicit data cleansing and Nearest-Neighbor Interpolation, alongside minimum area fraction thresholds to prevent dominant background classes from swallowing minority tool classes.
*   **Class Imbalance and AUROC/AURC Metrics:** Standard accuracy and the Area Under the Receiver Operating Characteristic curve (AUROC) dangerously obscure class prevalence imbalances by allowing massive True Negative counts from majority classes to artificially inflate scores. Comprehensive evaluation demands Area Under the Risk-Coverage curve (AURC) and macro-F1 scores to capture the realities of selective prediction and rare-event detection.
*   **Calibration Leakage in Zero-Shot Claims:** Claiming "zero-shot" or "zero-human-label" capabilities while retrospectively selecting decision thresholds or temperature scaling parameters on test data constitutes calibration leakage. This artificially inflates apparent utility and misrepresents the model's true clinical safety.
*   **Evaluation Protocol Divergence:** Current leading surgical vision-language benchmarks adopt divergent evaluation protocols. SurgVLP utilizes zero-shot hierarchical textual matching; SurgSLOT emphasizes cross-dataset semantic long-term tracking; and Med-Gemini focuses on long-context multimodal reasoning utilizing complex, proprietary benchmark datasets. This underscores the urgent need for a standardized evaluation rubric across the field.

Vision-Language Models (VLMs) and large multimodal foundation models are rapidly transforming surgical data science, promising systems capable of parsing complex intraoperative video, recognizing surgical phases, and identifying fine-grained anatomical structures without the need for exhaustive human annotations. The integration of self-supervised learning, zero-human-label pseudo-labeling, and student model distillation holds immense potential for democratizing surgical artificial intelligence. By leveraging vast, unannotated surgical video repositories, these methodologies aim to capture the nuanced, high-stakes reality of the operating room.

However, the transition from theoretical model architectures to robust, clinically reliable systems is fraught with profound methodological hazards. The evaluation of these models routinely falls victim to insidious forms of data leakage and statistical misrepresentation. Because surgical videos represent continuous, highly correlated data streams, traditional computer vision evaluation paradigms are frequently inadequate. When models are allowed to selectively predict—abstaining when uncertain—the metrics used to evaluate them must evolve simultaneously to prevent the masking of critical minority-class failures. 

This comprehensive report systematically investigates the critical evaluation methodologies and pervasive data leakage traps inherent in assessing surgical VLMs, pseudo-labeling pipelines, and student model distillation. By examining temporal splitting strategies, structural dataset anomalies in CholecSeg8k, class imbalance pitfalls in risk-coverage evaluation, calibration leakage in zero-shot claims, and the diverse protocols of prior benchmarks, this analysis provides a rigorous framework for navigating the complexities of modern surgical artificial intelligence evaluation.

## Temporal Autocorrelation and Splitting Strategies in Surgical Video Data

Surgical procedures are recorded as continuous video streams, resulting in a dataset structure that fundamentally differs from collections of independent natural images. The methodological choice of how to partition this data into training, validation, and testing sets—whether at the frame, clip, video, or patient level—determines whether an evaluation measures true algorithmic generalization or merely short-term memorization. 

### The Pitfall of Frame-Level and Clip-Level Splitting

In standard computer vision tasks, data is often shuffled and split randomly. Applying this paradigm to surgical video datasets, such as the widely used Cholec80 dataset, introduces a severe methodological flaw known as temporal autocorrelation leakage. This phenomenon is akin to evaluating a model's ability to recognize a surgical action, but discovering the model has simply memorized the unique "wallpaper" (e.g., specific lighting, distinct patient liver texture, or specific bleeding patterns) of the operating room where that action took place.

The fundamental issue stems from the high visual similarity between adjacent and near-adjacent frames within a single surgical procedure. A laparoscope moves relatively slowly (e.g., at average velocities typically ranging between 0.5 cm/s (5 mm/s) and 5.0 cm/s (50 mm/s) during low-to-medium motion phases [cite: 1]); therefore, consecutive frames share identical background anatomy, lighting conditions, and instrument characteristics.
When datasets are split at the frame level, frames from the exact same surgical procedure—and often the exact same surgical action—are distributed across both the training and validation sets. Research indicates that this allows models to achieve high performance by recognizing patient-specific textures or background artifacts rather than learning the generalized semantic features of a surgical phase [cite: 2]. 





The impact of this leakage is highly quantifiable. Frame-level evaluations consistently yield overestimations of model robustness. For example, some studies utilize an undersampling strategy, drawing frames randomly across all 80 videos of the Cholec80 dataset and partitioning them into stratified folds [cite: 2]. While this is sometimes done to maintain consistency with early, legacy benchmarks [cite: 2], it explicitly violates independence assumptions. The risk of performance overestimation due to visual correlation remains high, even when temporal redundancy is partially mitigated by downsampling the video to 1 frame per second (fps) [cite: 2, 3]. 

### The Necessity of Video-Level and Patient-Level Partitioning

To rigorously evaluate zero-human-label VLM pseudo-labeling and student model distillation, the evaluation methodology must strictly enforce independence between the training and testing distributions.

The most robust defense against temporal leakage is patient-level or video-level splitting. Under this paradigm, all frames belonging to a specific video (and thereby a specific patient and unique anatomical environment) are sequestered entirely into a single partition [cite: 3, 4, 5, 6]. 

While video-level splitting is methodologically superior, it introduces significant logistical and computational overhead. Processing long-temporal contexts or entire 3D volumetric video clips demands massive GPU memory compared to independently processing 2D frames, often forcing researchers to aggressively downsample frame rates or rely on specialized, resource-heavy architectures [cite: 7].

*Data Partitioning Standard:*
1. **Training Set**: Videos 01–40 (or similar subsets) [cite: 3, 4].
2. **Validation/Calibration Set**: Videos 41–60 (used for threshold selection and hyperparameter tuning) [cite: 3, 4].
3. **Held-out Test Set**: Videos 61–80 (strictly unseen during any phase of model development) [cite: 3, 4].

By grouping datasets by video using stratified group K-fold cross-validation, evaluators minimize leakage and force the model to prove its generalization capabilities on entirely unseen clinical environments [cite: 8]. This distinction is critical when transferring models to out-of-domain datasets. Studies demonstrate that natural-domain methods utilizing video-level pre-training consistently outperform image-level counterparts, with video-level models achieving substantial gains of up to 19.6% in phase-level Jaccard scores on out-of-domain datasets [cite: 5]. If a student model distilled from a VLM only achieves high metrics on a frame-level split, it is highly probable that the student has merely memorized the teacher's responses to specific, recurring visual textures rather than acquiring genuine surgical reasoning. 

Furthermore, when evaluating foundation models across multiple datasets, researchers must be vigilant regarding dataset overlap. For instance, because the CholecSeg8k dataset is a derivative subset of Cholec80, any evaluation utilizing CholecSeg8k must strictly exclude its originating videos from the pretraining corpora of the foundation model to prevent representation leakage [cite: 8, 9, 10].

Furthermore, datasets like CholecSeg8k contain an exceptionally small pool of distinct procedures—only 17 videos. The immediate, unanswered follow-up question is: *How do you practically implement robust video-level partitioning on a dataset with an 'N' as small as 17 without destroying statistical significance?* To solve this, evaluators must deploy a "leave-one-video-out" cross-validation (LOOCV) strategy [cite: 7, 11, 12]. By sequentially training on 16 videos and testing on the single held-out video across 17 distinct, rotating folds, researchers can maximize training volume while guaranteeing that the test distribution remains structurally unseen, ensuring true generalizability [cite: 7].

## Structural Quirks and Masking Traps in CholecSeg8k

While high-level video splitting governs temporal generalization, the granular mechanics of dataset utilization dictate the success of pixel-level tasks such as semantic segmentation. CholecSeg8k, a critical dataset comprising 8,080 frames extracted from 17 Cholec80 video clips, provides pixel-level annotations for 13 common classes in laparoscopic cholecystectomy [cite: 10, 13, 14, 15, 16]. However, evaluating VLM-generated pseudo-labels against this dataset requires navigating severe, undocumented structural traps.

### Watershed Mask Pixel-Value Mapping and Data Cleansing

CholecSeg8k distributes ground truth annotations in three formats per frame: a color mask for visualization, an annotation mask, and a watershed mask designed for algorithmic processing [cite: 10, 13, 16, 17, 18]. Relying blindly on these masks without preprocessing leads directly to pipeline failures.

The primary trap lies in the pixel-value mapping of the watershed masks. Unlike standard single-channel integer masks, the watershed masks are saved as RGB PNG images where the class ID is repeated across all three color channels. For instance, a pixel belonging to class 5 (Grasper) is encoded with the RGB value `[5, 5, 5]` [cite: 17, 18]. During preprocessing, models must specifically target this three-channel structure to extract a one-dimensional class tensor [cite: 18].

Furthermore, the dataset contains specific, localized data corruption that must be handled manually to prevent loss-function explosions during student model distillation:

*   **The Phantom 14th Class**: While the dataset officially defines 13 classes (including class 0 for background), exploratory data analysis reveals the presence of a 14th class color artifact. This artifact typically exists as an anti-aliasing or boundary interpolation error generated by the original annotation software, where discrete class edges are erroneously blended into an unmapped color. These anomalous pixels appear black in the watershed masks and white in the color masks [cite: 18]. If fed into a standard Cross-Entropy loss function expecting 13 classes, this artifact will crash the training loop. Evaluators must explicitly map this phantom class to an `ignore_index` within the loss function to exclude these corrupted pixels from metric calculations and backpropagation [cite: 18].
*   **Format Inconsistencies**: A small subset of exactly four specific frames—namely `video18_01139/frame_1216_endo.png`, `video35_00780/frame_858_endo.png`, `video37_00848/frame_865_endo.png`, and `video37_00848/frame_926_endo.png`—were color-corrected by the dataset authors but inadvertently saved in RGBA format rather than the standard RGB format used by the rest of the dataset [cite: 18]. Automated dataloaders will fail on these specific images unless an explicit RGB conversion step is hardcoded into the preprocessing pipeline [cite: 18].

These silent errors demonstrate the danger of treating public medical datasets as plug-and-play components. Methodologically sound evaluations must explicitly document their data cleansing pipelines to ensure reproducibility.

### Interpolation Traps and Minimum Area Fraction Thresholds

A secondary, yet equally devastating trap in CholecSeg8k involves image resizing and class prevalence. Medical VLMs often require specific input resolutions (e.g., 224x224 or 299x299 for models pretrained on natural images) [cite: 13]. 

When resizing the input RGB frames, standard bicubic or bilinear interpolation is appropriate. However, applying these standard interpolation methods to the *segmentation masks* alters the discrete integer values that represent class IDs. Bicubic interpolation blends neighboring pixels, transforming a sharp boundary between class 2 (Liver) and class 10 (Gallbladder) into fractional, non-existent class IDs [cite: 13]. Evaluators must strictly utilize Nearest-Neighbor Interpolation (NNI) or advanced Nearest-Exact Interpolation (NEI) when resizing masks to preserve the absolute integrity of the class ID integers [cite: 13, 19, 20].

Finally, CholecSeg8k suffers from extreme spatial class imbalance. Classes such as the liver and the background abdominal wall each occupy approximately 30% of the total annotated pixels [cite: 17]. Conversely, critical surgical instruments like the L-hook electrocautery and grasper occupy less than 1% of the total pixel area due to their small physical size [cite: 15, 17]. When evaluating pseudo-label generation, small area fractions are easily swallowed by evaluation noise or ignored by models optimizing for global pixel accuracy. Effective methodology requires establishing minimum area fraction thresholds. To mitigate this, evaluators routinely combine rarely appearing classes (such as unifying the dataset's two distinct instrument classes into a single collective "instrument" class), effectively reducing the 13 original classes down to 8 clinically distinct categories to ensure that evaluation metrics reflect surgical relevance rather than raw pixel dominance [cite: 18].

## Navigating Class Imbalance in Selective Prediction

As surgical Vision-Language Models and Large Language Models (LLMs) are increasingly designed to act as assistive tools rather than fully autonomous agents, the evaluation paradigm has shifted toward selective prediction. In this framework, a model is not forced to classify every frame; instead, it provides a confidence score and is permitted to abstain (defer to a human expert) when its confidence falls below a specific threshold [cite: 21, 22, 23]. While clinically vital, selective prediction introduces highly nuanced evaluation traps, particularly regarding class prevalence imbalance.

### The Pitfalls of Accuracy and AUROC in Skewed Distributions

In highly imbalanced datasets, standard accuracy is a mathematically deceptive metric. A model can achieve over 90% accuracy simply by collapsing into a majority-class predictor (e.g., always predicting "Abdominal Wall" or "Liver"), completely failing to detect rare but critical events like vessel clipping or thermal injuries. 

Similarly, the Area Under the Receiver Operating Characteristic curve (AUROC) falls victim to severe class imbalance. AUROC mathematically integrates both True Positives and True Negatives. In surgical videos, massive, continuously present majority classes (like background tissue) generate overwhelming quantities of True Negatives. These easy, passive predictions artificially inflate the AUROC score, producing an overwhelmingly positive metric that completely masks a model's total failure to detect small, rare instruments or brief adverse events. 

To counteract this, rigorous evaluations completely discard both standard accuracy and standard AUROC as primary endpoints. Instead, the literature mandates the use of macro-averaged F1 (macro-F1), balanced accuracy, or the Matthews Correlation Coefficient (MCC) [cite: 2, 21, 24, 25, 26]. Macro-F1 calculates the F1 score independently for each class and takes the unweighted mean, ensuring that a performance gain on a small class (e.g., L-hook electrocautery) contributes equally to the final score as a gain on a massive class [cite: 25]. By requiring class-wise metrics, evaluators ensure that apparent improvements are not merely artifacts of exploiting support asymmetry [cite: 25]. 

### Risk-Coverage Curves and AURC

When introducing selective prediction, models are evaluated on the trade-off between coverage (the proportion of total predictions the model chooses to make) and selective risk (the error rate on that retained subset) [cite: 21, 27]. 

The primary endpoint for selective prediction is the Area Under the Risk-Coverage curve (AURC) [cite: 21, 24, 27]. At every confidence threshold, the model's risk is plotted against its coverage, and the AURC integrates this risk across all thresholds [cite: 27]. 

*Understanding AURC Mechanics:*
1. **The Ranking Objective**: AURC does not measure whether a model's absolute confidence values are perfectly calibrated; it measures whether the model's confidence accurately *ranks* its predictions [cite: 24]. A model with excellent AURC successfully pushes all its incorrect predictions to the bottom of the confidence ranking, allowing them to be cleanly abstained from at lower coverages [cite: 24, 28].
2. **Tie-Aware Formulations**: Because LLMs and VLMs frequently output deterministic or highly quantized softmax scores, predictions can share identical confidence values (ties). Standard AURC calculations break down here. Methodologies must use "tie-aware" AURC. Much like a tie-breaker in a sports tournament where all players with the same score must share the identical penalty or outcome, tie-aware AURC adds equal-score groups as a block and integrates the right-continuous risk step, avoiding arbitrary and misleading orderings within tied scores [cite: 27, 28]. 

A major pitfall in selective prediction occurs when abstention disproportionately removes minority classes. If a model consistently flags the hardest, rarest surgical phases as "uncertain," the retained subset becomes heavily biased toward the easy, majority classes. The selective accuracy will artificially skyrocket, but the model's clinical utility is compromised because it systematically refuses to act on the most critical parts of the surgery. To expose this trap, evaluations must jointly report coverage alongside the specific class distributions of the rejected predictions, proving that abstentions are successfully targeting genuine errors rather than merely avoiding minority classes [cite: 21].

Furthermore, research explicitly demonstrates that the mechanisms improving a model's ranking ability (AURC) can simultaneously degrade its absolute probability accuracy (Expected Calibration Error, or ECE) [cite: 24]. For example, ensembling multiple models may halve the AURC (a massive improvement in ranking) while worsening the ECE [cite: 24]. Methodologies must disentangle these properties, reporting ECE for probability interpretation and Excess AURC (E-AURC) to isolate the quality of the confidence ranking from the baseline error rate [cite: 24, 25].

## The Illusion of Zero-Shot: Calibration and Threshold Leakage

The most egregious data leakage trap in modern VLM evaluation involves the definitions of "zero-shot" and "zero-human-label" learning. True zero-shot evaluation implies that a model generalizes to a task without ever updating its weights or tuning its hyperparameters on the target distribution. However, when evaluating pseudo-labeling and selective prediction, researchers often inadvertently leak target domain information through threshold selection and calibration.

### Threshold Selection Leakage

In a selective prediction pipeline, the model requires a confidence threshold (τ) to decide whether to predict or abstain [cite: 23]. If an evaluator processes the entire test set, plots the risk-coverage curve, and then explicitly claims a specific macro-F1 score at an "optimal" threshold found on that exact curve, they have committed threshold selection leakage [cite: 25, 28].

Post-hoc operating points computed on pooled out-of-fold test predictions represent an upper-bound, theoretically perfect deployment scenario, not a generalizable metric. A valid methodology requires a strict "freeze-and-test" protocol [cite: 22]. 

*The Freeze-and-Test Protocol:*
1. **Isolation**: A dedicated validation (or calibration) split must be strictly separated from the final test split [cite: 22, 28, 29].
2. **Calibration**: The optimal threshold (τ*) is calculated solely on the validation split based on predefined utility constraints (e.g., balancing the cost of human review against the cost of an error) [cite: 28, 30].
3. **Execution**: This threshold is frozen. The model is run exactly once on the held-out test split using the frozen threshold [cite: 22, 28, 29].

Failing to adhere to this protocol transforms a zero-shot evaluation into a supervised, few-shot tuning exercise. Claiming zero human labels while optimizing thresholds on the test data is mathematically disingenuous and severely overestimates real-world deployment safety. The downstream regulatory consequences of this are profound: submitting evaluation metrics polluted by threshold leakage to agencies like the FDA misrepresents the algorithm's genuine autonomy, potentially accelerating the clearance of brittle, unsafe devices that will fail unpredictably during live surgery.

### Calibration Leakage and Temperature Scaling

A corollary trap exists when calibrating the confidence scores of the VLM itself. Because neural networks are notoriously overconfident, their raw logits are often calibrated using post-hoc techniques like Temperature Scaling, which divides all logits by a single positive scalar to align predicted confidence with observed accuracy [cite: 23, 24].

If the temperature scalar is fit using the test dataset, calibration leakage occurs [cite: 26, 31, 32, 33]. A proper zero-shot calibration methodology must either use fixed, a priori thresholds derived from Bayesian principles (e.g., maximum entropy thresholds based on the number of classes) [cite: 31], or it must explicitly acknowledge that it is operating in a "calibrated split" paradigm where a small amount of annotated target data was consumed to align the probabilities [cite: 24, 28]. 

In the context of student model distillation (where a VLM teacher generates pseudo-labels for a smaller, specialized student network), adaptive thresholding is often utilized to filter noisy pseudo-labels [cite: 30]. Rather than using a fixed top-10% rule, advanced pipelines treat pseudo-label selection as a utility-based decision problem, balancing label quality against dataset coverage [cite: 30]. Even here, the threshold algorithms must be designed on validation subsets. Allowing the test set to influence the pseudo-label acceptance criteria contaminates the student model, invalidating any claims of zero-shot transferability.

## Evaluation Protocols of Prior Surgical VLM Benchmarks

To contextualize these methodological requirements, it is instructive to examine how recent, state-of-the-art surgical VLMs and self-training benchmarks handle evaluation. The architectures of SurgVLP, SurgSLOT, and Med-Gemini reveal fundamentally different approaches to defining and measuring surgical intelligence.

### SurgVLP: Text-Guided Prototype Matching

SurgVLP (Surgical Vision-Language Pretraining) focuses on aligning surgical video frames with text transcripts generated from automated speech recognition of surgical lectures [cite: 34, 35, 36]. Its core philosophy relies on mapping visual embeddings and textual descriptions into a shared joint latent space via contrastive learning [cite: 34, 35, 36].

*Evaluation Protocol:* 
SurgVLP is evaluated under a strict zero-shot prototype matching protocol [cite: 37, 38]. The visual encoder is frozen, and test frames (or temporal clip-level representations aggregated via a lightweight adapter) are embedded [cite: 35, 38]. Simultaneously, a set of candidate textual prompts (representing surgical phases or tool presence) are embedded by a frozen text encoder. Classification is performed simply by computing the cosine similarity between the visual embedding and the text prototypes [cite: 35, 37]. 
By avoiding task-specific fine-tuning or linear probing (which would require learning rates, hyperparameter tuning, and target-domain labels), SurgVLP's methodology provides a highly transparent, leakage-free assessment of raw representation quality [cite: 35]. Performance is typically reported on the Cholec80 and CholecT50 datasets using macro-averaged F1 and mean average precision (mAP) for fine-grained triplet recognition [cite: 37, 38].

### SurgSLOT: Cross-Dataset Semantic Long-Term Tracking

SurgSLOT pivots away from high-level phase recognition toward Interactive Video Object Segmentation (IVOS), aiming to track prompted surgical instruments or tissues across extensive temporal horizons [cite: 39, 40]. Built atop the Segment Anything Model (SAM) architectures, SurgSLOT tackles the specific challenge of target re-identification when an object disappears from the field of view for long periods [cite: 39, 40].

*Evaluation Protocol:*
SurgSLOT operates on the iSurg data foundation and emphasizes cross-dataset generalization [cite: 39, 40]. The evaluation protocol is exceptionally stringent: every test source is held out entirely from the training distribution [cite: 39, 40]. Furthermore, its zero-shot capabilities are explicitly tested on completely unseen procedure types and previously unencountered object categories across videos lasting up to 30 minutes [cite: 39, 40, 41]. 
By defining a stable object-level semantic anchor in the first frame and tracking it via a semantic-driven long-term memory buffer, SurgSLOT suppresses identity drift [cite: 39]. The reliance on Macro Average J&F (Jaccard index and F-measure) under strict cross-dataset holding effectively neutralizes the temporal autocorrelation traps that plague standard frame-level segmentation [cite: 39, 41].

### Med-Gemini: Long-Context Multimodal Reasoning

Med-Gemini represents the frontier of large, multimodal LLM application in medicine, scaling up parameters and context windows (e.g., reaching up to 200 billion parameters in Med-Gemini-L 1.0 [cite: 42]) to process entire electronic health records and continuous surgical videos, establishing a staggering 91.1% baseline accuracy on the general MedQA (USMLE) benchmark [cite: 43, 44].

*Evaluation Protocol:*
Med-Gemini diverges from specific pixel-tracking (SurgSLOT) or latent space similarity (SurgVLP), instead relying on zero-shot prompting with task-specific natural language instructions [cite: 44, 45]. The model evaluates entire video contexts to perform temporal localization of medical concepts (e.g., Medical Visual Answer Localization) and high-level safety assessments, most notably evaluating the Critical View of Safety (CVS) in cholecystectomies using private datasets sourced from Intermountain Health (IM) and Surgical Safety Technologies (SST) [cite: 46, 47]. 

Med-Gemini employs a "chain-of-reasoning" protocol, identifying relevant video segments before determining surgical phase compliance [cite: 45, 48]. Med-Gemini is evaluated on specific quantitative baseline metrics, testing its surgical scene comprehension and progression understanding via exact match accuracy, macro-F1 score, mean Average Precision (mAP) for detection tasks, and mean Intersection over Union (mIoU) for localization and segmentation tasks [cite: 47, 49]. Its performance is directly benchmarked against leading open models like SurgVLP (surgery-specific) and proprietary systems like GPT-4o (which serves as the best overall baseline for generalist autoregressive VLMs) [cite: 46, 47]. 

While its benchmark performance is staggering (outperforming prior models on numerous multimodal tasks), evaluating models of this scale introduces novel leakage concerns [cite: 44, 48, 50]. Because Med-Gemini is trained on massive datasets, guaranteeing that test videos from specific surgical databases were completely excluded from the pretraining corpus is highly complex [cite: 51]. Consequently, while it operates without task-specific fine-tuning (zero-shot prompting), the distinction between deep clinical reasoning and the memorization of in-distribution medical lectures remains a subject of intense academic scrutiny [cite: 46, 51].

### Benchmark Protocols Overview Table

| Model | Core Architecture/Backbone | Evaluation Protocol/Task | Primary Evaluation Metrics | Zero-Shot Handling Method |
| :--- | :--- | :--- | :--- | :--- |
| **SurgVLP** | Contrastive Vision-Language Encoders | Phase & Tool Recognition (Prototype Matching) | macro-averaged F1, mAP | Frozen visual/text encoders; cosine similarity to text prompts |
| **SurgSLOT** | Segment Anything Model (SAM) | Interactive Video Object Segmentation (IVOS) | Macro Average J&F | Strict cross-dataset holding; anchor-based memory tracking |
| **Med-Gemini** | Multimodal LLM (up to 200B params) | Contextual Reasoning & Critical View of Safety (CVS) | exact match accuracy, F1, mAP, mIoU | Zero-shot chain-of-reasoning prompting via natural language |

## Synthesis and Methodological Recommendations

The evaluation of zero-human-label VLM pseudo-labeling and student distillation in surgical data science requires a paradigm shift away from the relaxed protocols of traditional natural image benchmarks. To ensure that surgical AI models are safe, robust, and clinically translatable, researchers must adhere to a rigid methodological framework.

First, temporal autocorrelation must be respected. Frame-level dataset splitting is fundamentally invalid for surgical video evaluation; patient-level or video-level cross-validation (utilizing leave-one-video-out methods when 'N' is critically small) is mandatory to prevent representation leakage. 

Second, the structural nuances of the underlying dataset cannot be ignored. Utilizing CholecSeg8k requires explicit data cleansing pipelines to manage RGB watershed mask mappings, address phantom class anti-aliasing artifacts via `ignore_index` parameters, correct specific RGBA frame formatting bugs, enforce nearest-neighbor interpolations, and establish minimum area fraction thresholds to protect minority classes.

Third, class imbalance must dictate the choice of evaluation metrics. Standard accuracy and AUROC are inherently deceptive in surgical workflows. Models must be graded on macro-F1, balanced accuracy, and tie-aware AURC to isolate genuine predictive and ranking power from majority-class biases. When analyzing selective prediction, the distribution of abstained classes must be reported to ensure the model is not simply avoiding difficult pathologies.

Finally, claims of zero-shot capabilities must be guarded fiercely against calibration and threshold leakage. Operating points, confidence thresholds, and temperature scalars must be locked down on isolated validation splits using a strict freeze-and-test protocol. Only through an uncompromising adherence to these evaluation methodologies can the field accurately measure the progress of surgical vision-language models and their distilled student counterparts.

**Sources:**
1. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFxLjPkcF-b_8KcFzRvVBi7ZzHj8qFZbXok0nfOzmcTv60UlH4AXbYLfIC2GbbQuljYRp-vma5iCJDY7Zo6NZBVJvKHOuCDQIX8kVCel37e4_hVEosdSEIeZruUAy9f2Yn5Ki4e-jte3Q==)
2. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFV2ihmAGGWu9BOthQatYGy9j6y4kKEMyfQA1f91_wkHvtfmbtzdvg7YzjYw7YMvbxDWc3Bm19V9dq_WpWrSsM3FZE4zDVQynGqDBEtohEoGtXUDJ8IatTm27_bdg==)
3. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEBQ97tzD3ghXCUyGoL-tWGreiAu2v1aTfOwawAAKdLFlPWJSWIsdU0lilQXKxYWTbptENgkV-3vsGzXxfqMpOJavP3qdnDlJyol1seSFWEz-mZwuLQ9D_mQA==)
4. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH-ZU25cvd6K-gvpwE4omsOizaZliNEmGRmUSBk_m4sb92uJHcCtJ1obPeWVBMApVOKRrzpm7O-shBnQ0nkHa0XuDUgMINENejyukct6oxQc3WDvpDoIIcTzw==)
5. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEW3hovD34peNUGABXFUuX39PUMF8xpm5CRWOy14L9_MSabfD_BP1dhItitpKkaB7XAdLvzlGLO4LjyVDg2_Sh9VJ6phBmaJWBBVroqxstqbdG6_Fr1MAr91wdS2ZlhJu0mmdpJT5O-5g==)
6. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvnH5ZePxjkcqyHrrXjn6S18XfUQwvFZI3n9zmArOk4zLrfkAYSjee-yOA5uR-bfkms85HG09qKiuMNrPZYlDdhZLyi9mZcGkIAjOqW2J2XXGGnPP42B19OgB6ne6Dr9LwLHp31pFjJA==)
7. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHSZCCfcWbDEFCPT9NQnIHVrHSTVYx95qoyA_wwxm2utU0Uns7pgz1d7yFqVyanJeCt-FWYACYm_ZGBiESzNjdtWIacATduiiSNJb07zIGrtLK9EegW3QHrv-TChuAb5UE=)
8. [openreview.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHOcL-1E9-Ht5l-g_wg2YKj3EguhGtL63spr68Y1tSjQ7C2erq4i2vxT04jIzEGYBuHfd6MNe6xMj6OQ01TxjTZkcEQ5gyaBeuyCuhwnmVmPDsaK6mM9FkG8THlBXFlTzERwgCWQrmsXGX5qZlUbRFRvmEkWmlpE7GL84vMLQ==)
9. [miccai.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHbygE88HpTnxTyA1Pfncn7935due4oEGlik7qYtLvU2CFcmvqoSm8kiRzu1--HQU8JEeUKDgOV-_HpCn7wNxkurhWX-CGiI965Rhwhkd6TND6dKO2gU1F-s0kYxjirOTCHD0JMo2SNyu8EJm7AdMjcFsmmYw==)
10. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpN2PUgYvJ5QkZiYRfGjkQH48nDvcFVbFepF6ARtJhappWUWz1X-0AqS_EX8SUd5GWAW45G2E6gmZOnu4nrKy6S-k3IIrws1J6OzvleNC-Sd1jCPX9ZQx-gw==)
11. [emergentmind.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGuVAjqrtrdN9XQQasDbZxGqXiwJkhnR08jx8NETJIEilx0pZs0Sbxjs-T1LTK99JzJgrTOYfQBBsKORXn9xI89Nozn9FZiyZ4cl_7wkfKQSvkU1PJkThr9a4lFrs5ENGPUYSNIKkxxd-SzOOWoMZ_RF6pXTbSSXgUaj3jG5xa-wUdrPQK7J8RbA4tBe61ljI3xAud2795L13hIIvDXIZhnNMsvH7Y=)
12. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEELpLkwRVC7GjMIYVA8SrnosUeoyhcgUX8WfRe0VFMFKX8hQK4BEMseW3pkzofxuSKL61Ewe4kE4hRmaEvcIg3D5ki35JWiz93jNW05O7R7qISkG18XdvBqFv0sqa0tCtp3V-u4oYsRsB3TvAbVYIawVjjl4ydZEdBYVjw_oAUrkdTCQpitTPJ5lWXK6E6xYKB20Jap0aR3MFb3C-Ly6jUFe1tpN6FxNwrIUW5hyPNqgR4ukaKGR2W6xhwgX5eh9-2uF_B)
13. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFyMykhaTwf1N7xmAbens3rWWsYfoplJpadOYgBs835-MOaCpnz7VXwwsZk4HWBJ-TVPgpwpcFGoQO4Ehq5EK0UYHv0oz9MHcBWj9XCcSjibxGovuVNq-upzctlOpScfp-9VKMCgOzCxg==)
14. [acm.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGPc4Fv7Xhox1sLURtpUR-j8kyYf5IO3fVf40X3xApz6Sza0J46_SzVNRGwEHIBceCUi0V0bENQyycvJq5YAvvP9PcYUY4gZkMZcZKF6fYn0HEQUeXRNKISIc4ckx1KfW17_d-G)
15. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGbOVzKpE-tZ-kky5Ttt0MC3ix4pH4BcbwQpqFvvM3GL3MyRTH0AAISCxVX7-i8nkM88LzKy3QSu17cNGYAsw24tyW2T7B9_ihd12w2nDkyGCatA9fGxgpo7ml_u0IZmVd4d-ferfMIMAxtcWL2Y-k6lPUu1l2p-ye3iigsQ7ydgQALIgAByt6ikq9LrswyqpEtfcLeSJNXcXt2ImZfB_aYSp5nMjoQDFtsNSXcd7vKQ_2UTkmlj-ZlPOcqqGaZWnbjnuc9YEZrGw==)
16. [emergentmind.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYu0HXRZqaHyZRHRgxwoyP4g-lNlj7_O5GTyR1e7ic3S6s_dyMhbZ0bPrqoj2-UkhwxHuzkbwWLSI1qtIvSugosN7wrMYwiEIQxGw1bQzkyyWZZBORZYyV6dDCja-yCSxHDegF4z-VjEEQ5inJ)
17. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFz2WfUzfFDhOp_WIo9feT5-5RwD5hcCIeMrijWhfVi1ByPdzkYyMLsZ0ads-Z9PKAQXulChHpJwVNnnXGzjRLLJ8MnS1xWuDd2sh-LTpMbXoizNDhGhg==)
18. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3AP5nWd8Mo8tpf5r5KzkHgjYvNZnPjEAkGXg5UIYwAxHYflMQ_AzZTd5-_m9729_U-Bhck0v2vh0fm_WcY-xdpYGJYmxx-_ceAh7Yk6QCg3pB_JBcXv6rZMgkAyjR8CJ1bEVpc-7sO06QjEMlk083pKZJwlbHafQAYcv-KyINlFsf9KzVZL3Ia-8p3OrWVg==)
19. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGESLcFaVmPTrNP7s180SqvWVza49cmUwWqbFyhODkOoKj1aY9xVCsPI3A98tn3mAcV8y--BsQ9B_iCYzLsrcF7NGuMoYj5o2ppX7g_ZtxW3OOGvaUL_9a5OpP7D_kkLCpEW14=)
20. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1FwhKmCvcoWQ4fxJ0_AmIvJSV6Q02CwioREEjahQxmqyuoAqeX0yTUxm4GGuM4sJJ-N11l6-4blAgZb-IbxikorkXQ3hivnmKLVyeYsmsY2sFLC-uALM=)
21. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPR-wBE9Glzq60KKkjOUZriRP-yHgObRsioQXuyJezzZRp0s1Ia5tV2dSB_QAFzlk3tX2OTS6DogxBqCUlWThXOZRVrJbHEMLSQH6GYnHLork96-X1paHN-C8g-J6bCg==)
22. [ioai.no](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGI14vwh6RZhMCJXjcn-73Zx6zuRmYeOS-L38X2yB_x6eCvKrTEcUDysFB5dqbIDdpP3uzzF7Wi-nX32ZUq9eja54N7twX9lsPzGGmsSDTelkLUs2GybyBkMxJn7wvN8_CzvseW5dv9SO_5wTw5NVN2ZQSJT3AFMLfwuE1VN7ayrzTLkxwt6-y7N3F8pgijmWM3KRby8JU=)
23. [biorxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFg40PeP8uDIkSSYQGspWbp2MMLPnYCfqlXW4xmFS16Cdr8dm19WTvGwWJfD1oUfNFRG3MedF4qT8gNWmSvr_kKhhf7CdP4UlMKlyUs1BxTAMRxsXssYw08d2C4U2OUqsOY5c8r6WRURfqe37Xa3Rkf2fP-4s8=)
24. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3MAWkHH2Z15re3Y58PCeSdxnS9KsiYVmy1feYnFWargUiFzxml6OZr97OZLt0gRLufXP6MwuZizMYjrNTn7TQUCmgb0zsEh5Gbo5MImVO2Tv6oTcKnQ==)
25. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGcL-q_SqGlHEn2bfNpfSx-hLgpPQ92Mils7FsRboyJJDQTquspTtIT4c87D9N5v_a2ubf4Pf8tFVfTR4GObRP-LRJIvKMiXkX7KElf1sfAOJuOMM-6419RMMsyDsUD1A==)
26. [ieee.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEDu-PLcqwR-1WR_kMT079TPWyaGoLLNREKlngUgBL3JlLXXepW8LxMyacEhcbo3RoQtvDOci7o_niFsHeAqbzjeSY5ZsFLTaLRvPX4NXqS-0At_6nzRUppKfFChfCS3v5Rw0Ty)
27. [medrxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHUoXX8N4ZVfa2inMDSEilfC8z7gj8IXShsOeJqUy3GwGyoKaDUYLicxKkZNdkF2NBAOOxA1YFePD49DOcmF4FFtFpNOZ8sP2FDbpW1kr7ixUDlEESPST10yQWMzai4yGLuS1UZb6G3qkDh4pV5ukHInUPQkYNVNvMaPo8WL1E=)
28. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEWvR6pAwoNknZyMt0ZyOwlOTJ0wbrTxq9tNSGMC8HgH-EGhIGAcYcbJGDiIBTB3z6wLpg6JD1yHf1GOkJPBcYqRs-n2VyYGGiUYVfZENKwYehNelLODTSTOw==)
29. [openreview.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGAHe-2z-82W9UPSW7fRSAwOf4XIpxDH85lCVRQY39w3N7P3rc4jvSpReqZsBEugx_Ap0TDGLzJ-6N5ptD844O0N02WInv0LpEUTXPakGeKxponWK0dxQKq8XIkZZLP1B4qtchhvedS8xJXal1RjH_OCj9EqXPRwHueKbA3hg==)
30. [frontiersin.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsEBuJVtK9UzKp97QxSG5p5tCZ0fbLWyFszDymUFUOPJaExNbCuspszDhnL5oBokLZ9fETB4i2qzf1dennNXid1aidYrc8pfih1SAZ4SSP3yU5U4GhTaz2CivehgTnYT-Vo30oqPRpgrdyENAhOqKnZZ5WDZRBor-7pBW1npiixKUaVBByUzr6FiyIUz1sAoFNosatFk_80JC6)
31. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGrlkkGB9TPeVjkd2H5ZmRPv8et-vAPDZvOTt11iRLezYpYryV16veKeD9BvmY_RjlJF9OKWoC8SzWNbjeWYe41HvOsOtP_bHh6i005ySAzZXPZ5oOv9fKkSg==)
32. [aimodels.fyi](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGWp64HBXaNeln5QTSwYZ9flZK4aVO59PDtGh6jr4mYeqMPSBVoQMNOZPw-1dGlJu9xSzaUMzSNiS7LmrN9ApNAVegXrETXcL5dTL73r9kMIc4LxNTPXIGIzCFUu7I-IykwCBWcl3x9DV2wZlG0ozUdlKbFojSNNGzudjQSi0A-wo5YH9QFrdbZ8TGQ)
33. [paperpeel.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFHT8qb1HcAKRfmqN88Ghn2h_-RCK2JoGcVhwVEmU1REBfCv9NZloNG8Y9ZtbneVtFxVmIV_KvgesBW0eKQMYu6rwBxW33NMKDt_0Jae2n0UYL64Wh6HXa7QVVfLECzI6I=)
34. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHI81fXHkIvOEgihRtiAnSbGeJlzE99PmznxBG0TlOFQWiOa-2GZ9tfnxZ52Sagp_2cLrOjrgOxl5GZYiom580EXubKR47C1hNPbRjGDjQw1bUDrZ0U2sJkCh8aZJ8=)
35. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCchRSoFLj0LFUgsEH-cT1yU0AN_h8Rsg_yR5ChbMA-nnWXwWMVnFtlrn4W4eI68pr8MBFB2-6iwnURbnc4TNmEn41XtQ5nL6JDWmJ5Okjwqdhatoyj1ikWw==)
36. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHJf2ajERU-wMyiZQdvG6XxmU2yw2UJliLZON8ZcCMJyJyydNioa4RZbKJZH72xSrbbL_WXLpnv7nVUY_UfSTkdpuwh5UViRkTvhbdxJbcBoJM-nt-T7Am35rCPLwJiz0srJRmGF1gGmWPFqI3qpc7posXsUK8zViRVrjjKTaF-xG5CJnA83U_GTLCJhuIRV3o7WLyUP_t8z6wtnR0WIRTn5Zo_8y8tVEhTHztQjauFd_5QK-N1sZ7jmAzRdEpu09zUe2DAX3oDRE95pTUb4guyJ3dfZIkV)
37. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFqOvy2RZ2mJ04IZgUTa2drW6TCAbF8DFg6HMx_GG-wbfeQtac-rLUCF0N46BrYRvaMV-izqdWqfPxRGyQr1XTujNBxLt82ueYQLQZoYPcCR5LbPlfDbBAHog==)
38. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH-B_kPPkpGftXWikL8fAi4eTMnbIQzJAjc_1y8GsObkDf6c7IGWAWO3qfSnpb0mFNQ1z-SKZTXZ8aUpHeQ-eOfy-WUAu-2DyVmEBxlG_Fsv0Id7t3kk0nO1PfLVno=)
39. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7xWKiJ5Awv3NMwJmGCHLUAju45vdMqInOPxDfSfFJbnUYnpz6P2VF42bh8PlYBwIwbVHHTZTj1pYXvQlJgzMMJXJ-bhyfroCUgKSCK9-HjDYkvWfezg-5Y9w=)
40. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEdESKN3Nigh4t0HYf5soFhwFlq3PeVcxatZFJq-2PEzfjFB9QmUuf0n3vxMyRyD3POETzPFZa0QkFaXhTGZ1RKmiSCeyGlsUnqvMdF6ocipn-6tzlrLD3tbfe29PY=)
41. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGLrI5mPRIZmhtH3CDZnXJve9mQWrZERHjd9xfN9YvIlhN_SYRLp59Hu08FZtk0I_aZg10Lg_yYnDtdrkl2Ryhwl22pHAFVjaCQAKb-WieJsON1hQTyeVfBXA==)
42. [lifearchitect.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG--GfYXq3C8VSYc4cCI18Q2R-jEBfDDXGCj5ZUNIAtUgqujLZOce8OaALePCfZIO8UdLvesG3ELttnaV432vVwL0ok294-RmIGFiICMxiTP9k4Ir3X8oI2SH0=)
43. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEpAdm0cNeTdclj8KpIPbQdZjyVSShbkhzUm6SuQB4gkZtWgbH6j3f21SuAa0ARr0b8WkG_S_IodGWRMqVMLdrsA_L2Rk5AtWb3ORIC0eaqMIbqFAauDxDpy_s4MUD8MUt6ktKjePAVdQ==)
44. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpEPzUu7xjPw9lZQnA1CbCqblTdM2kr4SNlcaO6sb_4_o7w7n3qY3rQSPJmMuMwS0LjjVO7IBXh3_RrSzKXkeunPMRiRGcY1vnbYyYVyEm1RfXeuJRc7ws1Q==)
45. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEM81JyKNy_yMMpuOjUYaR3gLvggnmpd62s9PxQg03Y7TacFTBm9NXGKCl51So4PU5nL_lk52DaBNgDmGf6Q37V7MAI1duHQPvzFRaeUYcZFEPKKYm1jjfXq_lTM4CHRtvp_AeAv1a9fnyfxvKVXuxJJfrQ3hwOaZ1aiA3UR1v5S2CYafAY_dwLAu1Wmwl8MA==)
46. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDZQKVD9B7nZRJHBEai9F_Lx8cR_6nplLPw2Ge2vR_5CO6TMvFQhhJ74oZbtxTIfjGHIjv8obTJKAYcoFk7a0c7qukk2oY1UVlwgmBJjKn5bLjU_RgHKat6g==)
47. [nsf.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHQbQtM5U9QQlRV5Lykcwae4w2sGiXRyh3oo9oMvK5XLawPH2Wo-NhHwHOLP9r5q0pM-_m20z_Tq7KCjSeBSFKhb3_2qFe-bulY0kfiBDFnhlMcWNVswF7HA9xLD4ESoyE=)
48. [alphaxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGRf4hs48P3wVsAsXoo5XlvVDRcS5T0Traqxb0Bnw4c2zSJgsqk6TVkNb9_SfAi-SXkNvJGk67lkpMPPc8M52Y5-qEix3BI_NfZu5Z3l_eH1R3DCTNXpvxbFLO1VZg=)
49. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHIdyzqeIhw6pGTGvtslaCNwUuVS0QgiX73x_vL6EaODB_XGBCktElQ9K24xKNOnjeq9JlvYPCdrYXtBQVoIIgg0-nsF2_sWn6FpDJSUElFUeT7r9ArC5UKUFQ714vPq4IkYG8Gt93mzIGNGez5rIKOUkZv7Q1zrinQeKFUYnrrp6jTawhTuOyAKqbVvEuDQNpJABXEEStR9ySoPRyqxapZHDCgYaXk0kZNTU_l1rhmYyw4czMQBlMwsPWleG6kETLsCEia)
50. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWHr88-JrQnNa7uML6OFPFG0h3jS-otOaRvoVjTBGiYjBynj6fqIGo2OO4paErUPEnPO1JGO0lg50yRcvekZXHpEWdnkdV53M_pB_PVVkHfHMKJzEQaqgJiskar2RiKuZYpnPEiSNRgTg=)
51. [medrxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHNZaUyH-HQwQqV4J_RA4-cxM9Mb_R1dwKkWaGpkcNjI50D0wVe3DCxcRBDZ1wdBuITYwod-x4QKphfsIaQWG9KS750iHuxNgToHaPbn_aUT-tkQWN-qNHMt1-L3g4CAWo_tHpG5KijyYXuc_KSPRHRDr-RiA6cK3ed60xayYw=)

---

## Sources (Extracted Metadata & Links)

- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFxLjPkcF-b_8KcFzRvVBi7ZzHj8qFZbXok0nfOzmcTv60UlH4AXbYLfIC2GbbQuljYRp-vma5iCJDY7Zo6NZBVJvKHOuCDQIX8kVCel37e4_hVEosdSEIeZruUAy9f2Yn5Ki4e-jte3Q==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFV2ihmAGGWu9BOthQatYGy9j6y4kKEMyfQA1f91_wkHvtfmbtzdvg7YzjYw7YMvbxDWc3Bm19V9dq_WpWrSsM3FZE4zDVQynGqDBEtohEoGtXUDJ8IatTm27_bdg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEBQ97tzD3ghXCUyGoL-tWGreiAu2v1aTfOwawAAKdLFlPWJSWIsdU0lilQXKxYWTbptENgkV-3vsGzXxfqMpOJavP3qdnDlJyol1seSFWEz-mZwuLQ9D_mQA==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEW3hovD34peNUGABXFUuX39PUMF8xpm5CRWOy14L9_MSabfD_BP1dhItitpKkaB7XAdLvzlGLO4LjyVDg2_Sh9VJ6phBmaJWBBVroqxstqbdG6_Fr1MAr91wdS2ZlhJu0mmdpJT5O-5g==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH-ZU25cvd6K-gvpwE4omsOizaZliNEmGRmUSBk_m4sb92uJHcCtJ1obPeWVBMApVOKRrzpm7O-shBnQ0nkHa0XuDUgMINENejyukct6oxQc3WDvpDoIIcTzw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvnH5ZePxjkcqyHrrXjn6S18XfUQwvFZI3n9zmArOk4zLrfkAYSjee-yOA5uR-bfkms85HG09qKiuMNrPZYlDdhZLyi9mZcGkIAjOqW2J2XXGGnPP42B19OgB6ne6Dr9LwLHp31pFjJA==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHSZCCfcWbDEFCPT9NQnIHVrHSTVYx95qoyA_wwxm2utU0Uns7pgz1d7yFqVyanJeCt-FWYACYm_ZGBiESzNjdtWIacATduiiSNJb07zIGrtLK9EegW3QHrv-TChuAb5UE=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHOcL-1E9-Ht5l-g_wg2YKj3EguhGtL63spr68Y1tSjQ7C2erq4i2vxT04jIzEGYBuHfd6MNe6xMj6OQ01TxjTZkcEQ5gyaBeuyCuhwnmVmPDsaK6mM9FkG8THlBXFlTzERwgCWQrmsXGX5qZlUbRFRvmEkWmlpE7GL84vMLQ==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHbygE88HpTnxTyA1Pfncn7935due4oEGlik7qYtLvU2CFcmvqoSm8kiRzu1--HQU8JEeUKDgOV-_HpCn7wNxkurhWX-CGiI965Rhwhkd6TND6dKO2gU1F-s0kYxjirOTCHD0JMo2SNyu8EJm7AdMjcFsmmYw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpN2PUgYvJ5QkZiYRfGjkQH48nDvcFVbFepF6ARtJhappWUWz1X-0AqS_EX8SUd5GWAW45G2E6gmZOnu4nrKy6S-k3IIrws1J6OzvleNC-Sd1jCPX9ZQx-gw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEELpLkwRVC7GjMIYVA8SrnosUeoyhcgUX8WfRe0VFMFKX8hQK4BEMseW3pkzofxuSKL61Ewe4kE4hRmaEvcIg3D5ki35JWiz93jNW05O7R7qISkG18XdvBqFv0sqa0tCtp3V-u4oYsRsB3TvAbVYIawVjjl4ydZEdBYVjw_oAUrkdTCQpitTPJ5lWXK6E6xYKB20Jap0aR3MFb3C-Ly6jUFe1tpN6FxNwrIUW5hyPNqgR4ukaKGR2W6xhwgX5eh9-2uF_B>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGuVAjqrtrdN9XQQasDbZxGqXiwJkhnR08jx8NETJIEilx0pZs0Sbxjs-T1LTK99JzJgrTOYfQBBsKORXn9xI89Nozn9FZiyZ4cl_7wkfKQSvkU1PJkThr9a4lFrs5ENGPUYSNIKkxxd-SzOOWoMZ_RF6pXTbSSXgUaj3jG5xa-wUdrPQK7J8RbA4tBe61ljI3xAud2795L13hIIvDXIZhnNMsvH7Y=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYu0HXRZqaHyZRHRgxwoyP4g-lNlj7_O5GTyR1e7ic3S6s_dyMhbZ0bPrqoj2-UkhwxHuzkbwWLSI1qtIvSugosN7wrMYwiEIQxGw1bQzkyyWZZBORZYyV6dDCja-yCSxHDegF4z-VjEEQ5inJ>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFyMykhaTwf1N7xmAbens3rWWsYfoplJpadOYgBs835-MOaCpnz7VXwwsZk4HWBJ-TVPgpwpcFGoQO4Ehq5EK0UYHv0oz9MHcBWj9XCcSjibxGovuVNq-upzctlOpScfp-9VKMCgOzCxg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGPc4Fv7Xhox1sLURtpUR-j8kyYf5IO3fVf40X3xApz6Sza0J46_SzVNRGwEHIBceCUi0V0bENQyycvJq5YAvvP9PcYUY4gZkMZcZKF6fYn0HEQUeXRNKISIc4ckx1KfW17_d-G>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGbOVzKpE-tZ-kky5Ttt0MC3ix4pH4BcbwQpqFvvM3GL3MyRTH0AAISCxVX7-i8nkM88LzKy3QSu17cNGYAsw24tyW2T7B9_ihd12w2nDkyGCatA9fGxgpo7ml_u0IZmVd4d-ferfMIMAxtcWL2Y-k6lPUu1l2p-ye3iigsQ7ydgQALIgAByt6ikq9LrswyqpEtfcLeSJNXcXt2ImZfB_aYSp5nMjoQDFtsNSXcd7vKQ_2UTkmlj-ZlPOcqqGaZWnbjnuc9YEZrGw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFz2WfUzfFDhOp_WIo9feT5-5RwD5hcCIeMrijWhfVi1ByPdzkYyMLsZ0ads-Z9PKAQXulChHpJwVNnnXGzjRLLJ8MnS1xWuDd2sh-LTpMbXoizNDhGhg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3AP5nWd8Mo8tpf5r5KzkHgjYvNZnPjEAkGXg5UIYwAxHYflMQ_AzZTd5-_m9729_U-Bhck0v2vh0fm_WcY-xdpYGJYmxx-_ceAh7Yk6QCg3pB_JBcXv6rZMgkAyjR8CJ1bEVpc-7sO06QjEMlk083pKZJwlbHafQAYcv-KyINlFsf9KzVZL3Ia-8p3OrWVg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1FwhKmCvcoWQ4fxJ0_AmIvJSV6Q02CwioREEjahQxmqyuoAqeX0yTUxm4GGuM4sJJ-N11l6-4blAgZb-IbxikorkXQ3hivnmKLVyeYsmsY2sFLC-uALM=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGESLcFaVmPTrNP7s180SqvWVza49cmUwWqbFyhODkOoKj1aY9xVCsPI3A98tn3mAcV8y--BsQ9B_iCYzLsrcF7NGuMoYj5o2ppX7g_ZtxW3OOGvaUL_9a5OpP7D_kkLCpEW14=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFg40PeP8uDIkSSYQGspWbp2MMLPnYCfqlXW4xmFS16Cdr8dm19WTvGwWJfD1oUfNFRG3MedF4qT8gNWmSvr_kKhhf7CdP4UlMKlyUs1BxTAMRxsXssYw08d2C4U2OUqsOY5c8r6WRURfqe37Xa3Rkf2fP-4s8=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGI14vwh6RZhMCJXjcn-73Zx6zuRmYeOS-L38X2yB_x6eCvKrTEcUDysFB5dqbIDdpP3uzzF7Wi-nX32ZUq9eja54N7twX9lsPzGGmsSDTelkLUs2GybyBkMxJn7wvN8_CzvseW5dv9SO_5wTw5NVN2ZQSJT3AFMLfwuE1VN7ayrzTLkxwt6-y7N3F8pgijmWM3KRby8JU=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPR-wBE9Glzq60KKkjOUZriRP-yHgObRsioQXuyJezzZRp0s1Ia5tV2dSB_QAFzlk3tX2OTS6DogxBqCUlWThXOZRVrJbHEMLSQH6GYnHLork96-X1paHN-C8g-J6bCg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEDu-PLcqwR-1WR_kMT079TPWyaGoLLNREKlngUgBL3JlLXXepW8LxMyacEhcbo3RoQtvDOci7o_niFsHeAqbzjeSY5ZsFLTaLRvPX4NXqS-0At_6nzRUppKfFChfCS3v5Rw0Ty>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3MAWkHH2Z15re3Y58PCeSdxnS9KsiYVmy1feYnFWargUiFzxml6OZr97OZLt0gRLufXP6MwuZizMYjrNTn7TQUCmgb0zsEh5Gbo5MImVO2Tv6oTcKnQ==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGcL-q_SqGlHEn2bfNpfSx-hLgpPQ92Mils7FsRboyJJDQTquspTtIT4c87D9N5v_a2ubf4Pf8tFVfTR4GObRP-LRJIvKMiXkX7KElf1sfAOJuOMM-6419RMMsyDsUD1A==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHUoXX8N4ZVfa2inMDSEilfC8z7gj8IXShsOeJqUy3GwGyoKaDUYLicxKkZNdkF2NBAOOxA1YFePD49DOcmF4FFtFpNOZ8sP2FDbpW1kr7ixUDlEESPST10yQWMzai4yGLuS1UZb6G3qkDh4pV5ukHInUPQkYNVNvMaPo8WL1E=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEWvR6pAwoNknZyMt0ZyOwlOTJ0wbrTxq9tNSGMC8HgH-EGhIGAcYcbJGDiIBTB3z6wLpg6JD1yHf1GOkJPBcYqRs-n2VyYGGiUYVfZENKwYehNelLODTSTOw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGAHe-2z-82W9UPSW7fRSAwOf4XIpxDH85lCVRQY39w3N7P3rc4jvSpReqZsBEugx_Ap0TDGLzJ-6N5ptD844O0N02WInv0LpEUTXPakGeKxponWK0dxQKq8XIkZZLP1B4qtchhvedS8xJXal1RjH_OCj9EqXPRwHueKbA3hg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsEBuJVtK9UzKp97QxSG5p5tCZ0fbLWyFszDymUFUOPJaExNbCuspszDhnL5oBokLZ9fETB4i2qzf1dennNXid1aidYrc8pfih1SAZ4SSP3yU5U4GhTaz2CivehgTnYT-Vo30oqPRpgrdyENAhOqKnZZ5WDZRBor-7pBW1npiixKUaVBByUzr6FiyIUz1sAoFNosatFk_80JC6>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFHT8qb1HcAKRfmqN88Ghn2h_-RCK2JoGcVhwVEmU1REBfCv9NZloNG8Y9ZtbneVtFxVmIV_KvgesBW0eKQMYu6rwBxW33NMKDt_0Jae2n0UYL64Wh6HXa7QVVfLECzI6I=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGWp64HBXaNeln5QTSwYZ9flZK4aVO59PDtGh6jr4mYeqMPSBVoQMNOZPw-1dGlJu9xSzaUMzSNiS7LmrN9ApNAVegXrETXcL5dTL73r9kMIc4LxNTPXIGIzCFUu7I-IykwCBWcl3x9DV2wZlG0ozUdlKbFojSNNGzudjQSi0A-wo5YH9QFrdbZ8TGQ>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGrlkkGB9TPeVjkd2H5ZmRPv8et-vAPDZvOTt11iRLezYpYryV16veKeD9BvmY_RjlJF9OKWoC8SzWNbjeWYe41HvOsOtP_bHh6i005ySAzZXPZ5oOv9fKkSg==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCchRSoFLj0LFUgsEH-cT1yU0AN_h8Rsg_yR5ChbMA-nnWXwWMVnFtlrn4W4eI68pr8MBFB2-6iwnURbnc4TNmEn41XtQ5nL6JDWmJ5Okjwqdhatoyj1ikWw==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHI81fXHkIvOEgihRtiAnSbGeJlzE99PmznxBG0TlOFQWiOa-2GZ9tfnxZ52Sagp_2cLrOjrgOxl5GZYiom580EXubKR47C1hNPbRjGDjQw1bUDrZ0U2sJkCh8aZJ8=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHJf2ajERU-wMyiZQdvG6XxmU2yw2UJliLZON8ZcCMJyJyydNioa4RZbKJZH72xSrbbL_WXLpnv7nVUY_UfSTkdpuwh5UViRkTvhbdxJbcBoJM-nt-T7Am35rCPLwJiz0srJRmGF1gGmWPFqI3qpc7posXsUK8zViRVrjjKTaF-xG5CJnA83U_GTLCJhuIRV3o7WLyUP_t8z6wtnR0WIRTn5Zo_8y8tVEhTHztQjauFd_5QK-N1sZ7jmAzRdEpu09zUe2DAX3oDRE95pTUb4guyJ3dfZIkV>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFqOvy2RZ2mJ04IZgUTa2drW6TCAbF8DFg6HMx_GG-wbfeQtac-rLUCF0N46BrYRvaMV-izqdWqfPxRGyQr1XTujNBxLt82ueYQLQZoYPcCR5LbPlfDbBAHog==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH-B_kPPkpGftXWikL8fAi4eTMnbIQzJAjc_1y8GsObkDf6c7IGWAWO3qfSnpb0mFNQ1z-SKZTXZ8aUpHeQ-eOfy-WUAu-2DyVmEBxlG_Fsv0Id7t3kk0nO1PfLVno=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEdESKN3Nigh4t0HYf5soFhwFlq3PeVcxatZFJq-2PEzfjFB9QmUuf0n3vxMyRyD3POETzPFZa0QkFaXhTGZ1RKmiSCeyGlsUnqvMdF6ocipn-6tzlrLD3tbfe29PY=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7xWKiJ5Awv3NMwJmGCHLUAju45vdMqInOPxDfSfFJbnUYnpz6P2VF42bh8PlYBwIwbVHHTZTj1pYXvQlJgzMMJXJ-bhyfroCUgKSCK9-HjDYkvWfezg-5Y9w=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGLrI5mPRIZmhtH3CDZnXJve9mQWrZERHjd9xfN9YvIlhN_SYRLp59Hu08FZtk0I_aZg10Lg_yYnDtdrkl2Ryhwl22pHAFVjaCQAKb-WieJsON1hQTyeVfBXA==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG--GfYXq3C8VSYc4cCI18Q2R-jEBfDDXGCj5ZUNIAtUgqujLZOce8OaALePCfZIO8UdLvesG3ELttnaV432vVwL0ok294-RmIGFiICMxiTP9k4Ir3X8oI2SH0=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEpAdm0cNeTdclj8KpIPbQdZjyVSShbkhzUm6SuQB4gkZtWgbH6j3f21SuAa0ARr0b8WkG_S_IodGWRMqVMLdrsA_L2Rk5AtWb3ORIC0eaqMIbqFAauDxDpy_s4MUD8MUt6ktKjePAVdQ==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpEPzUu7xjPw9lZQnA1CbCqblTdM2kr4SNlcaO6sb_4_o7w7n3qY3rQSPJmMuMwS0LjjVO7IBXh3_RrSzKXkeunPMRiRGcY1vnbYyYVyEm1RfXeuJRc7ws1Q==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEM81JyKNy_yMMpuOjUYaR3gLvggnmpd62s9PxQg03Y7TacFTBm9NXGKCl51So4PU5nL_lk52DaBNgDmGf6Q37V7MAI1duHQPvzFRaeUYcZFEPKKYm1jjfXq_lTM4CHRtvp_AeAv1a9fnyfxvKVXuxJJfrQ3hwOaZ1aiA3UR1v5S2CYafAY_dwLAu1Wmwl8MA==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDZQKVD9B7nZRJHBEai9F_Lx8cR_6nplLPw2Ge2vR_5CO6TMvFQhhJ74oZbtxTIfjGHIjv8obTJKAYcoFk7a0c7qukk2oY1UVlwgmBJjKn5bLjU_RgHKat6g==>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHQbQtM5U9QQlRV5Lykcwae4w2sGiXRyh3oo9oMvK5XLawPH2Wo-NhHwHOLP9r5q0pM-_m20z_Tq7KCjSeBSFKhb3_2qFe-bulY0kfiBDFnhlMcWNVswF7HA9xLD4ESoyE=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGRf4hs48P3wVsAsXoo5XlvVDRcS5T0Traqxb0Bnw4c2zSJgsqk6TVkNb9_SfAi-SXkNvJGk67lkpMPPc8M52Y5-qEix3BI_NfZu5Z3l_eH1R3DCTNXpvxbFLO1VZg=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHIdyzqeIhw6pGTGvtslaCNwUuVS0QgiX73x_vL6EaODB_XGBCktElQ9K24xKNOnjeq9JlvYPCdrYXtBQVoIIgg0-nsF2_sWn6FpDJSUElFUeT7r9ArC5UKUFQ714vPq4IkYG8Gt93mzIGNGez5rIKOUkZv7Q1zrinQeKFUYnrrp6jTawhTuOyAKqbVvEuDQNpJABXEEStR9ySoPRyqxapZHDCgYaXk0kZNTU_l1rhmYyw4czMQBlMwsPWleG6kETLsCEia>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWHr88-JrQnNa7uML6OFPFG0h3jS-otOaRvoVjTBGiYjBynj6fqIGo2OO4paErUPEnPO1JGO0lg50yRcvekZXHpEWdnkdV53M_pB_PVVkHfHMKJzEQaqgJiskar2RiKuZYpnPEiSNRgTg=>
- <https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHNZaUyH-HQwQqV4J_RA4-cxM9Mb_R1dwKkWaGpkcNjI50D0wVe3DCxcRBDZ1wdBuITYwod-x4QKphfsIaQWG9KS750iHuxNgToHaPbn_aUT-tkQWN-qNHMt1-L3g4CAWo_tHpG5KijyYXuc_KSPRHRDr-RiA6cK3ed60xayYw=>
