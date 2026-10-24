# Third-party components

| Included files | Origin | License | Changes |
|---|---|---|---|
| `slgsea/vision/attention.py`, `pixel_decoder.py`, `position_encoding.py`, `criterion.py`, `matcher.py`, `misc.py` | [Mask2Former](https://github.com/facebookresearch/Mask2Former), including credited DETR-derived routines | MIT, Meta and original notices | Local imports, direct module construction; decoder is extended in `slgsea/decoder.py` |
| `slgsea/vision/deformable.py`, `deformable_core.py` | Deformable-DETR / SenseTime, as included in Mask2Former | Apache-2.0 | Differentiable PyTorch attention implementation |
| `slgsea/language/transformer.py`, `tokenizer.py`, BPE vocabulary | [OpenAI CLIP](https://github.com/openai/CLIP) | MIT, OpenAI | Text-transformer components and tokenizer |
| `slgsea/panoptic/evaluation.py`, `utils.py` | [COCO panoptic API](https://github.com/cocodataset/panopticapi) | BSD-2-Clause, Alexander Kirillov | Relative import, single-process evaluation wrapper, and zero-active-category guard for subset checks |

License texts and source information are provided in `licenses/`.
