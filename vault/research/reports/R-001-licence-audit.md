# Licence audit (all checked 13 Oct 2026)

Most of the strongest candidates pass: Qwen3-VL (all sizes), Qwen3.5 (all sizes), LightOnOCR-1/2, PP-OCRv5/v6, Kraken and timm ConvNeXt v1 are Apache 2.0. The traps are Qwen2.5-VL-3B and everything built on it, Chandra, LFM2-VL, ConvNeXt V2 and DINOv3.

**How to read "cited":** I read each repo's own card front matter through the Hugging Face API, and for the rows that quote licence wording I also opened the raw README or LICENSE file. Every URL is `https://huggingface.co/<repo id>` unless given. I did not read most cards in full.

| Model (repo id) | Params | Licence as written | Base and its licence | Passes? | Status |
|---|---|---|---|---|---|
| Qwen/Qwen2-VL-2B-Instruct, -7B-Instruct | 2.2B, 8.3B | apache-2.0 | Qwen2-VL-2B / -7B, apache-2.0 | yes | cited |
| Qwen/Qwen2-VL-72B-Instruct | 73B | "Qwen LICENSE AGREEMENT" (tag `tongyi-qianwen`) | same | unclear | cited (LICENSE) |
| Qwen/Qwen2.5-VL-3B-Instruct | 3.75B | "Qwen RESEARCH LICENSE AGREEMENT", "FOR NON-COMMERCIAL PURPOSES ONLY" | none | **no** | cited (LICENSE) |
| Qwen/Qwen2.5-VL-7B-Instruct, -32B-Instruct | 8.3B, 33B | apache-2.0 | none | yes | cited |
| Qwen/Qwen2.5-VL-72B-Instruct | 73B | "Qwen LICENSE AGREEMENT"; licence needed above 100M monthly users | none | unclear | cited (LICENSE) |
| Qwen/Qwen3-VL-2B/4B/8B/32B-Instruct and -Thinking; -30B-A3B, -235B-A22B | 2.1B, 4.4B, 8.8B, 33B | apache-2.0 | none | yes | cited |
| Qwen/Qwen3.5-0.8B, -2B, -4B, -9B, -27B, -35B-A3B (all take images) | 0.87B, 2.3B, 4.7B, 9.7B, 28B, 36B | apache-2.0 (LICENSE file is Apache 2.0) | Qwen3.5-*-Base, apache-2.0 | yes | cited |
| lightonai/LightOnOCR-1B-1025, LightOnOCR-2-1B, -2-1B-base | 1.16B, 1.0B | "Apache License 2.0" | Pixtral-based encoder, Qwen3-based decoder (v1 card) | yes | cited |
| allenai/olmOCR-7B-0225-preview | 8.3B | apache-2.0 | Qwen2-VL-7B-Instruct, apache-2.0 | yes | cited |
| allenai/olmOCR-7B-0725, -0825, olmOCR-2-7B-1025 | 8.3B | apache-2.0; card adds "intended for research and educational use" | Qwen2.5-VL-7B-Instruct, apache-2.0 | yes | cited |
| rednote-hilab/dots.ocr (now redirects to dots-studio/dots.ocr); dots-studio/dots.mocr | 3.0B | mit, plus a "dots.ocr LICENSE AGREEMENT" that allows commercial use and lists prohibited uses | "1.7B LLM", unnamed | yes, with caveat | cited |
| stepfun-ai/GOT-OCR2_0; stepfun-ai/GOT-OCR-2.0-hf | 0.72B; 0.56B | apache-2.0 | none | yes | cited |
| deepseek-ai/DeepSeek-OCR; DeepSeek-OCR-2 | 3.3B; 3.4B | mit; apache-2.0 | none stated | yes | cited |
| PaddlePaddle/PaddleOCR-VL, -VL-1.5, -VL-1.6 | 0.96B | apache-2.0 | baidu/ERNIE-4.5-0.3B-Paddle (licence not checked) | yes | cited |
| nanonets/Nanonets-OCR-s, Nanonets-OCR2-3B | 3.75B | none declared | Qwen2.5-VL-3B-Instruct, Qwen Research | **no** | cited |
| nanonets/Nanonets-OCR2-1.5B-exp | 1.65B | apache-2.0 | Qwen2-VL-2B-Instruct, apache-2.0 | yes | cited |
| ibm-granite/granite-docling-258M | 258M | Apache 2.0 | none | yes | cited |
| docling-project/SmolDocling-256M-preview (ds4sd/ redirects here) | 256M | cdla-permissive-2.0 | SmolVLM-256M-Instruct, apache-2.0 | yes (inference: permissive) | cited |
| opendatalab/MinerU2.5-2509-1.2B | 1.16B | agpl-3.0 | none | unclear | cited |
| opendatalab/MinerU2.5-Pro-2604-1.2B, -2605-1.2B | 1.16B | apache-2.0 | none | yes | cited |
| opendatalab/MinerU2.0-2505-0.9B | 1.03B | none declared; README missing | n/a | unclear | unconfirmed |
| datalab-to/chandra; datalab-to/chandra-ocr-2 | 8.77B; 5.3B | "AI PUBS OPEN RAIL-M LICENSE (MODIFIED)" | not stated | **no** | cited (LICENSE) |
| zai-org/GLM-OCR | 1.3B | MIT | none | yes | cited |
| baidu/Qianfan-OCR; FireRedTeam/FireRed-OCR | 4.7B; 2.1B | apache-2.0 | FireRed: Qwen3-VL-2B-Instruct, apache-2.0 | yes | cited |
| tencent/HunyuanOCR | 1.1B | "TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT"; excludes EU, UK, South Korea | none | unclear | cited (LICENSE) |
| stanford-oval/churro-3B | 3.75B | `other` / `qwen-research`; its LICENSE is the Qwen Research text | Qwen2.5-VL-3B-Instruct | **no** | cited |
| microsoft/Florence-2-base, -large, -base-ft, -large-ft | 0.23B, 0.77B | MIT | none | yes | cited |
| google/paligemma-3b-pt-448; paligemma2-3b-pt-448/-896; gemma-3-4b-it, -12b-it; gemma-3n-E4B-it | 2.9B to 12B | `gemma` (Gemma Terms of Use), manually gated | gemma-3-*-pt, gemma | unclear | cited |
| google/gemma-4-E2B-it, -E4B-it, -26B-A4B-it, -31B-it | 5.1B, 8.0B, 26B, 31B | apache-2.0 | gemma-4-*, apache-2.0 | yes | cited |
| HuggingFaceTB/SmolVLM-Instruct, -256M, -500M; SmolVLM2-2.2B, -500M-Video | 0.26B to 2.2B | apache-2.0 | SmolLM2 plus SigLIP (SigLIP so400m is apache-2.0) | yes | cited |
| HuggingFaceM4/Idefics3-8B-Llama3 | 8.5B | Apache 2.0 | Meta-Llama-3.1-8B-Instruct, `llama3.1` community licence | unclear | cited |
| OpenGVLab/InternVL3-1B, -2B, -8B | 0.94B, 2.1B, 7.9B | front matter apache-2.0 (with a stray `license_name: qwen`); body says "MIT License", Qwen2.5 component "Apache-2.0" | Qwen2.5 language models | yes (stray tag is an inconsistency) | cited |
| OpenGVLab/InternVL3_5-1B, -2B, -4B, -8B, -38B | 1.1B to 38B | apache-2.0 | Qwen3, apache-2.0 | yes | cited |
| openbmb/MiniCPM-V-2_6 | 8.1B | no tag; card: "MiniCPM Model License", commercial use only after a registration questionnaire | n/a | unclear | cited |
| openbmb/MiniCPM-V-4, -V-4_5, MiniCPM-o-2_6 | 4.1B, 8.7B | Apache-2.0 | Qwen3-8B plus SigLIP2 (4.5) | yes | cited |
| microsoft/Phi-3.5-vision-instruct; Phi-4-multimodal-instruct | 4.1B; 5.6B | MIT | none | yes | cited |
| mistralai/Pixtral-12B-2409 | 12B | Apache 2.0 | Pixtral-12B-Base-2409 | yes | cited |
| vikhyatk/moondream2 | 1.9B | apache-2.0 | none | yes | cited |
| moondream/moondream3-preview; moondream3.1-9B-A2B | 9.3B | "Business Source License 1.1 with an Additional Use Grant"; `moondream-model-license-1.0` (not read) | none | unclear | cited; 3.1 unconfirmed |
| LiquidAI/LFM2-VL-450M, -1.6B, -3B; LFM2.5-VL-1.6B | 0.45B to 3.0B | "LFM Open License v1.0"; commercial use only below $10M annual revenue | none | **no** (unless the host is under the threshold) | cited (LICENSE) |
| microsoft/trocr-base-handwritten | 333M | mit | BEiT plus RoBERTa initialisation | yes | cited |
| microsoft/trocr-large-handwritten, -small-handwritten, -base/-large-printed, -base/-large/-small-stage1 | 384M to 608M | no licence on the card; the microsoft/unilm repository LICENSE is MIT | same | unclear, probably yes | card cited as empty |
| PaddlePaddle/PP-OCRv5_server_rec, _mobile_rec, en_/latin_ mobile; PP-OCRv6_medium_rec, _small_rec, _tiny_rec | not listed | apache-2.0 | none | yes | cited |
| Kraken (software, github.com/mittagessen/kraken) | n/a | Apache License 2.0 | n/a | yes | cited |
| CATMuS Kraken models on Zenodo (McCATMuS 13788177, CATMuS-Print 10592716, CATMuS Medieval 21488839) | n/a | cc-by-4.0 | n/a | yes, with attribution | cited from Zenodo search metadata only |
| timm/convnext_large.fb_in1k, .fb_in22k, .fb_in22k_ft_in1k; convnext_base.fb_in22k_ft_in1k; facebook/convnext-large-224, -224-22k, -base-224-22k | 89M to 230M | apache-2.0 | none | yes | cited |
| timm/convnextv2_large.fcmae_ft_in22k_in1k | 198M | cc-by-nc-4.0 | none | **no** | cited |
| facebook/convnextv2-large-22k-224 | 198M | apache-2.0 (conflicts with the timm card for the same family) | none | unclear; avoid | cited |
| facebook/dinov3-convnext-large/-base-pretrain-lvd1689m; dinov3-vitb16; timm/convnext_large.dinov3_lvd1689m | 86M to 196M | `dinov3-license` (DINOv3 License, 14 Aug 2025), gated | distilled from dinov3-vit7b16 | unclear | cited |
| facebook/dinov2-base, -large; timm/vit_base_patch14_dinov2.lvd142m | 87M, 304M | apache-2.0 | none | yes | cited |
| Kansallisarkisto/multicentury-htr-model | not listed | Apache 2.0 | microsoft/trocr-large-handwritten (no licence on its card) | yes on licence | cited |

Names that do not exist as queried: `PaddlePaddle/PP-OCRv6_server_rec` and `_mobile_rec` (v6 sizes are tiny, small and medium), `rednote-hilab/dots.ocr-1.5`, `datalab-to/chandra-2`.

## Safe to use
- Qwen2-VL 2B/7B, Qwen2.5-VL 7B/32B, all Qwen3-VL, all Qwen3.5.
- LightOnOCR-1/2, olmOCR (all), GOT-OCR2, DeepSeek-OCR/-2, PaddleOCR-VL, GLM-OCR, Granite-Docling, MinerU2.5-Pro, Nanonets-OCR2-1.5B-exp.
- Florence-2, Gemma 4, SmolVLM/SmolVLM2, InternVL3.5, MiniCPM-V 4/4.5, Phi vision, Pixtral, moondream2.
- trocr-base-handwritten, PP-OCRv5/v6, Kraken, timm and facebook ConvNeXt v1, DINOv2.

## Do not use
- **Qwen2.5-VL-3B and its fine-tunes** (churro-3B, Nanonets-OCR-s, Nanonets-OCR2-3B): non-commercial only. This confirms the churro denial: its own LICENSE file is the Qwen Research text.
- **Chandra 1 and 2:** barred for anyone with more than $2M revenue or funding, or who competes with Datalab; also share-alike.
- **timm ConvNeXt V2 (fcmae):** CC-BY-NC-4.0.
- **LFM2-VL:** commercial use is capped at $10M revenue.

## Needs a staff ruling
- **Gemma Terms** (Gemma 3, 3n, PaliGemma 1/2): no non-commercial clause, but section 3.1 requires passing the use restrictions down and Google reserves remote restriction. Gemma 4 avoids this.
- **Qwen LICENSE AGREEMENT** (72B models): commercial use allowed below 100M monthly users; irrelevant at our budget.
- **DINOv3 License:** the fetched summary shows no commercial ban, but it bars military and nuclear uses, requires redistribution under the same agreement and "Built with DINOv3" attribution, and is gated.
- **Idefics3:** declared Apache, but built on Llama 3.1, whose community licence passes down an acceptable-use policy.
- **dots.ocr:** MIT prevails, but the supplementary agreement adds prohibited uses.
- **MinerU2.5-2509:** AGPL-3.0 allows commercial use but is copyleft.
- **moondream3:** BSL 1.1 bars competing paid services.
- **MiniCPM-V 2.6:** commercial use needs registration.
- **HunyuanOCR:** territory excludes the EU, UK and South Korea.
- **TrOCR other than base-handwritten:** no licence on the card; only the GitHub repository says MIT.
- **CATMuS (CC-BY 4.0):** licence passes, and the organisers' starter pack used one. The same goes for the Kansallisarkisto model. Whether a reader pretrained on outside handwriting is acceptable is a separate question from licence.

## Opened and not opened
- **Opened (front matter via API):** every repo in the table.
- **Opened (raw LICENSE or README lines):** Qwen2.5-VL-3B/72B, Qwen2-VL-72B, Qwen3.5-9B, churro-3B, chandra-ocr-2, LFM2-VL-1.6B, dots.ocr, HunyuanOCR, MiniCPM-V 2.6/4.5, InternVL3-2B, InternVL3_5-2B/8B, moondream3-preview, olmOCR-2, LightOnOCR-1/2, Idefics3, Kansallisarkisto, TrOCR cards, GLM-OCR, Florence-2 and Phi-4 LICENSE, Kraken LICENSE, unilm LICENSE.
- **Opened as a fetched summary:** ai.google.dev/gemma/terms (last modified 1 Apr 2026), ai.google.dev/gemma/docs/gemma_4_license, ai.meta.com/resources/models-and-libraries/dinov3-license.
- **Could not open:** MinerU2.0 README, individual Zenodo CATMuS record pages (second query failed), Moondream model licence 1.0, the Llama 3.1 licence text, the Gemma prohibited-use policy.
- **2026 leaderboard leaders** (PaddleOCR-VL-1.6, MinerU2.5-Pro, GLM-OCR on OmniDocBench) come from a search snippet only; their licences are cited from their cards.

## Sources
- https://huggingface.co/api/models/<repo id> and https://huggingface.co/<repo id>/raw/main/README.md or /LICENSE for every repo above
- https://huggingface.co/dots-studio/dots.ocr/raw/main/dots.ocr%20LICENSE%20AGREEMENT
- https://ai.google.dev/gemma/terms
- https://ai.google.dev/gemma/docs/gemma_4_license
- https://ai.meta.com/resources/models-and-libraries/dinov3-license
- https://raw.githubusercontent.com/mittagessen/kraken/main/LICENSE
- https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE
- https://zenodo.org/api/records?q=CATMuS%20kraken (records 13788177, 10592716, 21488839)
- https://unstract.com/blog/best-opensource-ocr-tools/ and https://www.spheron.network/blog/best-open-source-ocr-vlm-self-host-gpu-cloud-2026/ (snippets)
