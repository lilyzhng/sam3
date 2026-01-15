Question: Design an Automated Validation & Triage system that reduces validation time from 1 week to 1 day. The system should automatically: (1) Accept high-confidence correct labels, (2) Reject low-quality labels, (3) Route ambiguous cases to humans.

Scale: millions of frames/day

Inputs: pre-annotation outputs (boxes/classes/scores), raw camera + LiDAR, scene metadata (weather, time-of-day, sensor pose), VLM for semantic validation

Output per frame (or per clip): {ACCEPT, REVIEW, REJECT} plus auditable "reasons" and localized "regions of concern"

KPIs: cut human review volume ≥70% while keeping missed-error rate low (high precision on ACCEPT)

## Design

**Q1. How would you design the automated validation & triage system?**

> I will use a VLM as an intelligent component to review the top-K uncertain candidates. I'll use confidence scores from the pre-labels to select uncertain examples and review low-confidence samples one by one. Since VLMs don't work well on object-level information, whenever the off-board model produces a pseudo label, it will indicate which label is uncertain. I'll use that to select the dedicated region. For example, if the label is uncertain whether it's a motorcyclist or cyclist, we can pass the grounding bounding box, 3D bounding box from the pseudo label, and the class to ask the VLM to review it. This imitates how humans review—they also look at the dedicated region to decide if it's a false positive detection or misclassification. The VLM will provide accept/reject decisions with auditable reasons and localized regions of concern. If the VLM is also uncertain, we pass it to human labelers for final check.

**Q2. What would you include in the VLM prompt? How would you instruct it to evaluate quality beyond just the class?**

> The input is the pre-annotation output containing boxes, classes, and scores, plus metadata. For low-confidence inputs, I'll review them against the original output. The prompt will be: "You are an expert scene analyzer. Given these inputs and classes, cross-validate if the class and location of the object is accurate. Look at this region originally annotated by another model. Verify if: (1) the orientation is correct, (2) the bounding box is tight, (3) the object class is correct, or (4) if there's no object at all (return empty box, don't hallucinate). Provide structured output in JSON format including accept/review/reject and the reason for your decision."

**Q3. How are you thinking about inference cost and latency when processing millions of frames daily with VLM queries?**

> I'm thinking we can evaluate one example per sequence since samples from one sequence are often very similar to each other. We need to detect if a certain frame needs to be re-annotated by humans. If we find one frame from the slice has an issue, we'll tag that slice to be re-annotated because all frames from a slice must be correct. By selecting one representative example from the video sequence, we can drastically reduce the compute cost. We need something to tell us whether we need to re-evaluate certain slices or not.

**Q4. If one frame in a sequence has an issue and you tag the entire slice for re-annotation, how do you avoid over-rejecting good frames? What if the issue is localized?**

> This is a very good question. Ideally we should provide fine-grained details on which frames need to be re-annotated. If the assumption is that we only need to re-annotate certain frames (not entire sequences), we can use a bottom-to-top approach. First, check the first frame. If it has issues, then check the last frame. If the last frame has the same issue, very likely across this entire time window we have issues. If the last frame doesn't have the issue, check the middle frame. Basically, we update the timestep start and end to consolidate a time window that might have issues by doing a divide-and-conquer approach—a left and right pointer approach starting from both ends. We reduce the window size using these left and right pointers.

**Q5. How does this "divide and conquer" strategy impact VLM inference cost compared to your original single-frame sampling idea?**

> Yes, this divide-and-conquer approach will increase the inference cost compared to the original one-sample approach. But this is inevitable if we want better, more precise localization on which frames are problematic—we have to use this approach.

> *WEAK DEFENSE. Claims the cost increase is 'inevitable'. Misses opportunities for cheaper filtering steps.*

**Q6. What if issues are very sparse or not easily localized through binary search? How confident are you this won't miss errors, impacting your 'low missed-error rate' KPI?**

> We can create a validation dataset which has been corrected by human labelers using the original pseudo labels. This will serve as ground truth, meaning we know the input data has false examples that have been corrected by human labelers, giving us clean output. We'll use that to evaluate VLM performance. If the VLM can catch all the mistakes confidently, we can measure how much turnaround time it saved in human cost and labor compared to the manual approach, so we get the impact and numbers. We also get how well or poorly the model handles it. If the model is doing a horrible job, we'll first adjust the prompting system or introduce an iterative approach, because sometimes if we ask the model to do too many things at once, it won't do well. We can divide it into multiple steps, or if there's a large domain gap—for example, it works well in city scenes but doesn't work well on highway/high-speed environments—we might consider fine-tuning the model with highway scenes to bridge the domain gap.

> *Deflects to explaining offline validation methodology rather than fixing the runtime sampling logic.*

**Q7. How do you quantify 'uncertainty' from the VLM's output to trigger human handover?**

> Asking an LLM/VLM for a raw confidence score is naive and known to be poorly calibrated (hallucinated confidence). We can't trust the VLM's self-reported confidence directly. We should use 'Self-Consistency' (querying multiple times with high temperature and checking agreement) or look at the token log-probabilities to derive a robust uncertainty metric.

# Industry
Automated QA in Labeling Platforms (Scale AI, 2024–2025): Production data-annotation platforms have adopted confidence filtering and consensus checks to ensure quality at scale. For example, Scale AI’s pipelines automatically accept annotations that meet high-confidence thresholds or consensus among multiple labelers, and only tasks falling below a confidence threshold get an extra review layer
scale.com
. For standard computer-vision tasks like bounding boxes or polygons, Scale reports using consensus scoring and model confidence filters to catch errors: if automated models (or multiple human votes) agree with high confidence, the label is approved, whereas low-confidence or conflicting cases are flagged for human QA
. This hybrid approach allows the majority of routine labels to pass through quickly while ensuring that ambiguous or low-quality annotations are intercepted for manual verification, maintaining high overall accuracy.

# Action Items
- what are common labeling error that can be addressed by automatic triaging ? get nuscenes mini dataset, load the 3d ground truth, create some false labels, e.g. wrong classes, off bounding bboxes, temporal inconsistency

# Reading
1. FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos https://yulugan.com/projects/FoundationMotion.html
- 什么是 Qwen2.5-VL语义引导的Grounded-DINO