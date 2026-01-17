Question: You are the tech lead to build an **ML segmentation system** for a robotic workcell in a factory.

Brainstorming: https://chatgpt.com/c/69680d9a-2840-8333-8b91-4ecbbdfbafa8
Datature benchmark: https://datature.io/blog/sam-3-a-technical-deep-dive-into-metas-next-generation-segmentation-model

The system is used in two modes:
1. **Autonomous robotic manipulation**
    - Automatically segment target objects required for pick-and-place
    - Segment regions to avoid (fixtures, tools, hands)
    - Output masks are consumed by downstream planners
    - Pseudo-labels are filtered and used to continuously train/refresh the online model.

2. **Human-in-the-loop operation**
    - Operators can:
        - Provide **high-level intent or constraints**, for example: what category to pick or reject (e.g. "defective items only")
        - Perform **safety gating**: approve / pause / resume actions when risk is detected
    - The system must respond interactively and be correct enough for action

> Mental model: 1. problem & scope -> 2. high level design & buy-in -> 3. design deep dive -> 4. Wrap-up

---

# Step 1. Problem & Scope

## 1. Problem Framing: What are we actually solving?

- 在面试官问出问题之后，我们不仅要看到技术要求technical requirements, 更重要的是看到问题的本质是什么，这里关键并不是说怎么去做 segmentation，本质应该是怎么样做到 reliable perception for robot action under industry settings，with human fallback to maintain throughput and safety. 
- **DONT**：不要在没有充分理解clarity之前，就开始大谈特谈技术问题。 我们要 **force clarity**，了解到问题的本质，可以去问几个问题。

**A. [Operating context & Actor] Where does the system live and who interacts with it?**

	- Operating context:
		- 这是一个实时在线系统，还是离线数据/训练系统？因为这两个设定会根本性地改变系统设计目标.
		- online system路线: 低延迟, 高可靠性, redundancy
		- offline system路线: 数据质量, 标注一致性, long tail, cost reduction
		- 如果interviewer说两者都有，那是一个很好的展示思辨trade-off 的机会
			- 那我会把它拆成两个 mode
				- 一个是 runtime perception path，优先safety/low latency.
				- 一个是 offline data curation path, 优先accuracy, long tail.
		- operator 他即可以是real-time system的supervisor可以进行干预。也可以是offline数据的Quality gate.
	- Who is the user?
		- primary user: factory operator（直接和 segmentation 结果交互）
		- secondary user: robotic system 或 training pipeline（consume the output of segmentation）

**B. [Problem Signal] What is their pain point? what is breaking today?**

	- 承上启下的问问题：基于上一个问题我们谈到了这个user主要是operator，robot system，那他们面临的问题是什么？
	- Question: Given that operators and robots are the primary consumer of the segmentation system, where does the system fail today, what causes stops, errors, or manual intervention?
	- 情况 1：Interviewer 给你 failure modes
		- segmentation unstable under occlusion
		- false positives near fixtures
		- mis-pick on reflective parts
	- 情况 2：Interviewer 没给你具体 failure modes
		- 面试官不给确定的信息, 这也是非常常见的, 这里的处理方法是。明确假设
    - If we don't have concrete failure modes yet, I'll make these reasonable assumptions
			- In industrial robotic perception, failures usually come from:
				- long-tail visual conditions like occlusion or lighting
				- ambiguity between similar parts
				- domain generalization to factory environment
	- **Alignment is the key**: 无论是情况1还是情况2，这里的目的并不是拿到所有的 failure mode，而是和面试官对齐一个 dominant failure signal。这个要么来自于 interviewer， 要么来自于你做的explicit assumption。
	
**C. [North Star] From first principles, what does success look like?**

	- 在前两个问题进展得顺利的时候，你这里已经有足够的context。那么一个成熟的做法是自然而然问一句"What is the successful system look like? Is success mainly about safety, throughput, or reducing human intervention?" 
	- 情况 1：Interviewer 给你明确的success definition
		- 比如他说：real-time latency is the top priority or throughput matters more than perfect masks.
		- 你可以总结：Got it. So success here is not perfect segmentation, but enabling safe robot action most of the time, and gracefully falling back to human input when uncertainty is high
	- 情况 2：Interviewer 回答比较模糊
		- 比如他把问题抛回给你： how would you define success?
		- 最好的情况是你对这个公司的背景有所了解，把 interviewer心中的 "success criteria" 用你的话来说出来：The system succeeds if it enables reliable robot action in most cases, and avoids costly or unsafe failures by asking for human input when confidence is low.  

---

## 2. Dependency & Scope: What does this system own?

**A. What this system owns**

	- A segmentation service that supports two interaction modes
		- robot mode: automatic target segmentation for pick decisions
		- operator mode: interactive selection via click / lightweight text intent
	- Global context, Temporal consistency
    - Same object shouldn't flicker frame-to-frame
		- ID consistency or mask smoothing
	- Domain adaptation
		- factory domain shift: lighting / occlusion / new SKU
		- Own the data flywheel: collect hard cases + minimal human corrections → retrain / fine-tune
		
**B. Dependencies**

	- 简单概括，Upstream跟Downstream在日常工作当中其实是非常重要的。但是呢，因为我们在面试时间不是特别多，所以就不展开。 
	- Upstream：camera stream + calibration / timestamps
	- Downstream：grasp planner consumes mask + confidence
	- Human input: requirements via _prompt_ 

**C. What this system does NOT own**

- out of scope在工作里很重要，但在面试里必须点到为止，否则容易偏题。
- **用「声明式」而不是「询问式」**: I'll assume this system focuses on perception and does not own motion planning or hard safety enforcement, but integrates with them via clear contracts.

---

## 3. Input/Output: Ground design with examples

如果不讲 input 跟 output，那 design 就是空中楼阁。一定要准确地讲例子。

- 情况1: 问面试官，So what exactly is the input?
	- At high level，the system takes multiview camera video data + text prompts, and outputs accurate segmentation segmentation mask
	- 如果他多给了信号（e.g. depth, LiDAR, events），你就 **顺势接住**
- 情况2: 当 interviewer 不给 input / output 时，自己补上
	- 提出一个合理假设 + 明确声明这是你的假设 + 请求纠正
        - Step 1: If inputs and outputs are not specified, I'll make a reasonable assumption and let me know this align with your thoughts
        - Step 2: I'll assume the system takes camera video from a robotic workcell as input, with optional lightweight operator intent, and outputs instance segmentation masks plus a confidence signal for robot action.
		- **Step 3: Does that align with what you had in mind?**
- 用一个真实场景落地Concrete example
    - Input: Assuming we are at a robotic workcell where a robot arm picks metal parts from a bin. There are parts, fixtures, and occasionally a human operator's hand entering the scene. 一句话已经把 **robot + object + human** 都放进来了
	- 在这个工厂环境下，物体严重遮挡、叠在一起，反光导致边界错， 新 SKU / 新形态。因此存在异常处理 / long-tail recovery，operator设定一个指标，what to pick / what to reject， 系统自动执行. 
	- Output: segmentation masks for target objects and forbidden regions, plus an actionability/confidence signal.
	- Operational assumption:
		- Safety gating: 当检测到异常情况时 (人手进入工位)，operator **override / resume / confirm** system
		- 自动化负责 95% 的正常流程；operator 只负责 5% 的不确定和异常，让系统不停线、且更安全
		
---

# Step 2. High Level Design & Get Buy-in

第二部分先展示广度，把每个component的相互关系展示出来, get the buy-in 
	- At high level, the design is divided into two ...
	- Go through concrete use case

**1. Prompt / Ontology Layer (ask → propose)**

	- Alignment: How often do your target/forbidden definitions change, and what are the must-have pick targets and must-avoid regions in this workcell?
	- Proposal: Based on that, I would define a lightweight ontology that separates **pick targets** vs **forbidden regions** (hands, fixtures, tools). Use versioned prompts / labels that map consistently to this ontology, so we can handle ontology drift (new SKUs, new defect types) without relabeling everything immediately.
    
**2. Segmentation Model (ask → propose)**

- Alignment: What's the hardest part of the vision problem here—reflective surfaces and clutter, small parts, or occlusions—and do we need video stability or is per-frame enough?
- Proposal: Given that, I'd start from a promptable foundation segmentation model SAM3 and fine-tune it on factory data, so we get strong base generalization plus fast domain adaptation. If video stability matters, we'll use the video/tracking capability rather than treating frames independently.
	- Alignment: Are we targeting on-device real-time on Thor/Orin, or is off-board inference acceptable for the first iteration?
- On compute: if this needs to run on an edge GPU, for example NVIDIA-Thor, we'll size the runtime by controlling resolution, max tracked objects, and precision (FP16/INT8). Technically SAM3 (**~500 TOPS**) can fit in Thor of 1035 TFLOPS
		- Ops ≈ O(num_layers × tokens² × hidden_dim)
    
**3. Temporal Consistency / Tracking (ask → propose)**

	- Alignment: In your workcell, do you see more failures coming from temporal issues—mask drift/flicker through occlusion, picking the wrong identical-looking part over time—versus purely single-frame boundary accuracy?
- Proposal: If temporal failures are a major driver, I'd use a video-capable promptable segmenter like SAM3 so we get both strong per-frame masks and built-in tracking/temporal memory. That gives us stable masks across frames and reduces drift without having to bolt on a separate tracker

**4. Training / Fine-tuning Pipeline (ask → propose)**
    
	- Alignment: What training signal do we actually have or can we realistically get—pixel masks on images only, or short video clips with consistent object IDs (even sparsely)?
- Proposal: SAM3 is a detector + tracker sharing a Perception Encoder (PE). After training the detector, we freeze PE and train the tracker. I'd use a staged fine-tuning pipeline: first fine-tune SAM3 on factory images/keyframes to close the appearance gap (spatial masks), then—if we can get clips with minimal ID supervision—freeze the spatial encoder and fine-tune the tracker/temporal components to improve stability and reduce drift. We gate each stage with held-out factory eval before updating the model.

**5. Evaluation / Release Gate (ask → propose)**
    
- Alignment: When deciding a model is 'good enough' to ship, what matters most—pick success/safety incidents, intervention rate, or pure mask accuracy—and what's the minimum bar you require?
- I'd use a gated release checklist:
		- (1) spatial mask quality on held-out factory scenes (IoU + boundary F-score on targets/forbidden), 
		- (2) temporal stability on short clips (drift/flicker/ID switches), 
    - (3) a small regression suite to ensure we didn't break general promptability
			- only ship if all pass and the key business metric improves.

---

# Step 3. Design Deep Dive

## 1. Interviewer Feedbacks / Push Back

1. **Data & supervision（最大风险）**
	- 你到底有什么数据？image masks 有多少？video 有没有 ID / track supervision？
	- 如果没有 video IDs，你怎么训 temporal？靠 propagation + correction？怎么控噪声？
	- 你准备用什么 prompt 分布来训练？text vs visual prompts 怎么混？
	
2. **Fine-tuning recipe（你说要 spatial→freeze→temporal，我会逼你讲清楚）**
	- 具体怎么 freeze？freeze 哪部分？为什么？
    - 你怎么避免 catastrophic forgetting，保证 promptability 还在？
	- 全量 FT vs 分 stage FT，你怎么选，怎么验证？
	
3. **Temporal stability / tracking failure modes（工厂最常见痛点）**
	- mask drift/flicker 在你的场景会导致什么 downstream failure？
	- 相似零件、遮挡、反光时 tracking 怎么处理？
	- 你用什么 metric 证明 temporal 变好了？

4. **Evaluation & release gate（你说 gate，我会问 gate 的细节）**
	- 你的 offline eval set 怎么 split 才不 leakage？
    - 什么叫 'good enough'？哪个指标是 hard gate？
	- prompt regression suite 用什么构成?

5. **（如果你提了 Thor/Orin）Compute feasibility（我会快速 sanity check）**
	- 你假设的 FPS / 分辨率 / objects-per-frame 是多少？
	- 如果达不到实时，你的 fallback 是什么（downsample / fewer objects / batch / off-board）？

---

## 2. 开场自己带节奏

Alignment: For deep dive, I can go into (A) data & supervision, (B) staged fine-tuning for spatial+temporal, or (C) eval & release gating. Which area matters most to you?

> Remember to talk about tradeoffs, edge cases

---

### Block A. Data supervision (data 可获得性)

> 可执行的最小闭环（collect → label-assist → QA → train）

1. **Data availability**: Do we already have recorded workcell video/images available for training, or do we need to add logging? 我们现在有没有可用的录像/图片数据？能不能加日志采集？
    - Proposal: Lightweight logging + event-triggered sampling. 只在关键事件（mis-pick / low confidence / operator override）前后采集短 clip，平时低频采样保证覆盖面。

2. **Supervision signal（决定能不能训 temporal)**: can we get short clips around events, and do we have any weak signals like pick success/failure timestamps to trigger sampling?
    - Proposal: 用现有 segmentation model 做 pseudo masks；人只修 keyframes/失败片段。Video 的话：keyframe 标注 + propagation + correction。

3. **Quality control**: "Filtering + small human audit."
    - Proposal: confidence/consistency filter + 每批少量人工抽检，防止 pseudo-label poisoning。

Block A（Data & supervision）在 Step 3 里你可以这样处理：**不问"去哪收数据/你们内部怎么做"**，只做两件事：**设定合理假设 + 确认监督信号类型**，然后立刻切到 training recipe

1. 只问一个高杠杆对齐问题: For this design, should I assume we have (a) image masks only, or (b) short video clips with sparse IDs / temporal correspondence? That choice determines whether we can train temporal tracking model.
2. 立刻给一个通用的数据闭环: I'll assume we can log multi-view frames/clips in the workcell. We'll start with a small set of high-quality masks, then scale via model-assisted labeling + sparse correction, with a lightweight QA filter.
3. 立刻过渡到你想讲的重点（training recipe）: Given that supervision assumption, the key is the staged recipe: spatial adaptation first, then freeze encoder and train temporal/tracking.

**Image Data Requirements**

| Requirement | Recommended           | Notes                                        |
| ----------- | --------------------- | -------------------------------------------- |
| Images      | 300-500               | Cover lighting variations, angles, clutter   |
| Annotations | COCO JSON format      | Bounding boxes + RLE masks                   |
| Classes     | Your factory concepts | e.g., "gripper", "part A", "defect", "screw" |
| Negatives   | 10-20% images         | Images where target is NOT present           |

**Video/Temporal Training Settings**

| Setting              | Value              | Rationale                               |
| -------------------- | ------------------ | --------------------------------------- |
| Video clips          | 50-100             | Short clips with sparse annotation      |
| Epochs               | 20                 | Fewer epochs (more compute per epoch)   |
| Frames per clip      | 4                  | num_stages_sample: 4                    |
| LR scale             | 0.05               | Half of Stage 1 (fine-tuning)           |
| Tracking loss weight | 2.0x               | trk_loss_scale_pos: 2.0                 |
| Resume from          | Stage 1 checkpoint | resume_from: ${paths.stage1_checkpoint} |

---

### Block B. Training recipe（spatial → freeze → temporal)

**1) 具体怎么 freeze？freeze 哪部分？为什么？**

I'd freeze the parts that encode general visual understanding, and train the parts that handle domain specifics and temporal association.

具体：

		- **Stage 1（spatial）**：先训练 detector/segmentation 能力来适配工厂外观差异。
		    - 训练：mask decoder / transformer heads（更靠近输出）
		    - 视觉 backbone 可以 **low-LR** 或 partial tune（视数据量和过拟合情况）
		        
		- **Stage 2（temporal）**：训练 tracking/temporal 模块时，我会 **freeze Perception Encoder / vision backbone**（paper style）。
    - 训练：memory encoder / temporal association / tracking queries（与时间一致性相关的模块）

为什么：
- temporal stage 的目标是 "association and stability"，不需要重学 "what is an object"。
- Freeze encoder 能减少 drift：避免 tracker training 把通用表征拉偏，从而破坏 promptability/generalization。
		    
**2) 怎么避免 catastrophic forgetting，保证 promptability 还在？**
	
	A) **Constrain updates**
		- Differential learning rate / layer-wise decay：越 pretrained 的部分更新越小（尤其 text tower）。
		- Early stopping + conservative tuning。

	B) **Keep prompt distribution**
		- 训练时用 **prompt mix**：text + visual prompts（point/box/mask）。
- 还要加 "noisy prompts / ambiguous prompts"，否则线上会崩。

	C) **Regression gate**
- Build a small "promptability regression suite"：一小组通用场景/通用 prompts，确保 fine-tune 后仍能对新对象/新描述 work。
		- 如果 regression fail → 回滚或减小更新范围。

> Punchline: I prevent forgetting via constrained updates, training on the same prompt distribution we expect at inference, and a regression gate that explicitly checks promptability

**3) 全量 FT vs 分 stage FT，你怎么选？怎么验证？**

	选择逻辑：
	- 默认我会选 **staged FT（spatial → freeze → temporal）**，因为 spatial 与 temporal 的目标不同，分阶段更稳。
- 只有在"数据充足 + 不担心通用能力 + 明确只服务一个固定场景"的情况下才考虑全量 end-to-end 联合训练。
	    
	怎么验证（不讲太细但可落地）：
	- 用三类指标做 gate：
	    1. Spatial mask quality（IoU + boundary）在 held-out factory scenes
	    2. Temporal stability（drift / flicker / ID switches）在 held-out clips
	    3. Promptability regression（通用 prompts 不明显退化）
	
	并且用 ablation：
	- baseline：no FT
- stage1 only
- stage1 + stage2

	        看 stage2 是否真正提升 temporal metrics，而不伤 promptability。

**4) Prompt mix + augmentation + early stopping（你要怎么说才像 TL）**

Training-wise I'll use mixed prompting (text + visual), strong factory-style augmentations (lighting/specular/blur/occlusion), and early stopping on a held-out workcell split to avoid overfitting.

如果 interviewer 追问 prompt mix 细节（你准备 2 句即可）：
		- text prompts：controlled vocabulary + synonyms（避免 prompt brittleness）
		- visual prompts：中心点、边界点、tight/loose box、negative point（提高鲁棒性）

---

### Block C. Evaluation & release gate（把"gate"落地）

**3) Offline eval set 怎么 split 才不 leakage？**

先说原则（interview-ready）：

> "I'll split by _correlated units_ rather than random frames. Random split leaks near-duplicate frames and overstates performance."

具体做法（按优先级）：
		- **Time-based split**：按日期/班次切（train 用 earlier days，test 用 later days），模拟真实 drift（lighting, wear, new SKUs）。
		- **Scene / run split**：同一段视频/同一次运行不能跨 train/test（avoid near-duplicates）。
- **SKU / part variant split（如果有）**：留出一部分新 SKU 或新包装做 "novelty test"。
		- **Workcell / camera split（如果多工位）**：留出一个工位/相机做跨环境泛化。

**4) 什么叫 "good enough"？what is release gate?**

先把指标分成三层：**safety-critical hard gates**、**model-quality hard gates**、**soft metrics**。

> Punchline: "Good enough" means: safety doesn't regress (forbidden recall), temporal stability doesn't regress, and promptability doesn't regress; then we require target-mask boundary quality to meet the grasp tolerance.

**Hard gates（必须过，不然不发版）：**

a. **Forbidden regions high-recall gate（安全）**: 对手/工具/fixture 这类 "must-avoid" 区域：目标是 **高 recall**（宁可误报也不能漏）

b. **Temporal stability gate（如果用于视频动作）**: If temporal drift increases beyond a threshold, we block.

c. **Promptability regression gate（能力不被 fine-tune 毁掉）**: If general prompt behavior regresses, we block.

**Model-quality gates（通常也是 hard gate，但更可讨论）：**
	- **Target masks boundary quality**（抓取相关）：boundary F-score / contour error
	- **Overall mask quality**：IoU / AP(mask)

**5) Prompt regression suite 用什么构成？**

	 > Punchline: Prompt regression suite is a small, fixed set of scenes × prompts that covers in-domain intents, general prompts, and adversarial phrasing—used as a hard gate against catastrophic forgetting.

1. 目标：验证 fine-tune 后仍然具备 "promptable foundation model" 的核心能力。

2. 构成我建议三块（小而强，几十到几百样本就够）：
    - **Factory-in-domain prompt suite（你关心的）**: prompt robustness + ontology mapping没坏。
    - **General capability mini-suite（防 catastrophic forgetting）**: 少量通用场景/物体（可以来自公开小集或内部非敏感集）。目的：保证模型没变成 "factory-only segmenter"。
    - **Adversarial prompt suite（专门测脆弱点）**: ambiguous prompts / distractors. 验证模型不会因为 prompt 变化就崩。

---

# Step 4. Wrap up

> **完整度 + thought leadership**

## Recap 模板（5 点，90 秒以内）

Let me recap the design end-to-end.

**1. North star（1 句）**

We're optimizing for reliable perception for robot action in a factory workcell, with safety as a hard constraint and minimal human intervention as the fallback.
    
**2. System decomposition（2 句）**

High level we decomposed into: ontology/prompt layer → SAM3 segmentation + video tracking → staged fine-tuning pipeline → evaluation/release gate.

The ontology layer stabilizes what we mean by targets vs forbidden regions; SAM3 gives us strong spatial masks plus temporal tracking.
    
**3. Key architectural bets / trade-offs（2–3 句）**

	- Key bet: use a promptable foundation segmenter (SAM3) to handle long-tail and ontology drift faster than training bespoke models.
- Training bet: staged adaptation—spatial first, then freeze perception encoder and train temporal modules—to improve stability without destroying promptability.
- Release bet: hard gates on forbidden-region recall and temporal stability, plus prompt regression to prevent catastrophic forgetting.

**4. First MVP**

Next step is to run a baseline (zero-shot), then Stage-1 spatial fine-tune on a small high-quality set, measure gains, and only then invest in Stage-2 temporal fine-tune if drift/flicker is a real bottleneck.

**5. Main risks**

Main risks are limited video supervision and confirmation bias/forgetting; we mitigate with minimal clip supervision, conservative updates, and regression gating.

---

## Next iteration（你可以用来结束并互动）

**Quick check before we wrap**: does this match your expectations, and which risk would you de-risk first? so I can prioritize what we iterate on the first MVP?

- Next steps / iteration plan
- Risk register + mitigations（2 句）
- Main risks are data scarcity for video tracking, and confirmation bias/forgetting during fine-tuning.
- We mitigate with minimal video supervision (short clips + sparse labels), conservative updates, and regression gating.
