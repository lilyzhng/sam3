Here’s a simple “execute immediately” plan using Unsloth on RunPod (A100/H100). This stays v0-minimal: train a Qwen3-VL verifier that answers ACCEPT/REJECT for (image + text + mask-group overlay). You can later swap the training backend to pure HF once the data format is stable.

1. Pick the base model (don’t overthink)

* Use Qwen3-VL-8B-Instruct (not Thinking)[https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct]  for v0 verifier training. Thinking models often add extra reasoning tokens that make classification-style SFT noisier.
* Unsloth explicitly supports Qwen3-VL fine-tuning. ([Unsloth][1])

2. Stand up RunPod

* Choose an image with CUDA + PyTorch (any standard RunPod PyTorch template is fine).
* Attach a persistent volume for datasets + outputs.

3. Install Unsloth + deps (one-time)

* Install: `unsloth`, `transformers` (Unsloth docs usually pin versions), `datasets`, `accelerate`, `peft`, `bitsandbytes`, plus `opencv-python` or `Pillow` for overlay rendering.
* Follow Unsloth’s Qwen3-VL “how to run & fine-tune” page as the reference baseline. ([Unsloth][1])

4. Get MASKGROUPS-HQ

* Download from the project page / paper resources (MaskGroups-HQ[https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ] + 2M are part of the Ref2Any release). ([arXiv][2])
* Sanity check you can load:

  * image
  * text query
  * mask groups (multiple masks per group)

5. Build your verifier dataset (no synthetic corruption, just “wrong group” negatives)
   For each (image, query, correct mask group):

* Positive example: correct group.
* Negatives (3–6 per positive), sampled from the dataset:
  a) wrong mask group from the same image (hard negative)
  b) wrong mask group from a different image (easy negative)
  This is “real negatives” in the sense that they are valid mask groups, just mismatched to the query.

6. Convert each example to “overlay image + short judge prompt”

* Render: original image with the candidate mask group overlaid (semi-transparent) + boundary outline. (This is important; don’t try to pass raw masks first.)
* Prompt template (keep constant):
  System: “You are a segmentation QA verifier.”
  User: “Query: <text>. Does the highlighted mask group match the query? Answer exactly: ACCEPT or REJECT.”
* Label: “ACCEPT” or “REJECT”.

This format is ideal for Unsloth-style VLM SFT: image + short text in, short text out. ([Unsloth][3])

7. Fine-tune with Unsloth (LoRA/QLoRA)
   Start with:

* QLoRA 4-bit (fast + robust, and Unsloth provides pre-quantized variants) ([Hugging Face][4])
* LoRA target: attention + MLP (Unsloth defaults are usually fine)
* 1 epoch v0 (MaskGroups-HQ is large; you’ll see signal quickly)

Recommended stable hyperparams (v0)

* Image size: 672 or 896 (pick one and stick to it)
* Effective batch size: 64 (use grad accumulation if needed)
* LR: 1e-4 (LoRA), warmup 3%
* Max text length: 256
* Save adapter every N steps + final

8. Minimal eval (so you can deploy as a quality gate immediately)
   Split 90/10.
   Track:

* Accuracy (ACCEPT vs REJECT)
* Precision at high-confidence ACCEPT (this matters most for QA gates)
* A small hand-audit: 100 samples where the model is confident ACCEPT/REJECT—check obvious failure types.

9. First deployment loop (the “quality gate” usage)
   Given your own candidate masks from SAM3 (or any segmenter):

* Render overlay
* Ask verifier: ACCEPT/REJECT
* Keep ACCEPT as auto-pass; route REJECT to human / second-pass prompting

10. Next iteration hooks (don’t do them now, but plan them)

* Add “BORDERLINE” third class once binary is stable.
* Add EV-style “exhaustive?” task later (set-based judgment), but keep v0 to MV only.
* Move from overlay-image input to “mask tokens” input only if you need speed/control (that’s when HF low-level is worth it).

If you want, I can give you a very small JSON schema for the training rows (path_to_overlay_image, prompt_text, label_text) and a matching Unsloth dataloader pattern so you’re not guessing about formatting.

[1]: https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune?utm_source=chatgpt.com "Qwen3-VL: How to Run Guide"
[2]: https://arxiv.org/abs/2506.05342?utm_source=chatgpt.com "Refer to Any Segmentation Mask Group With Vision ..."
[3]: https://unsloth.ai/docs/basics/vision-fine-tuning?utm_source=chatgpt.com "Vision Fine-tuning | Unsloth Documentation"
[4]: https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit?utm_source=chatgpt.com "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
