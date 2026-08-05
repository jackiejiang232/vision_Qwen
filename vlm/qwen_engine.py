import torch

from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from .prompt_builder import (
    INSTRUCTION_TO_DINO_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_instruction_to_dino_prompt,
    build_user_prompt,
)
from .vlm_config import DINO_MAX_NEW_TOKENS, SCENE_MAX_NEW_TOKENS


def resize_for_qwen(image_pil, max_side=448):
    width, height = image_pil.size
    longest = max(width, height)

    if longest <= max_side:
        return image_pil

    scale = max_side / float(longest)
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )

    return image_pil.resize(new_size)


class QwenVLEngine:
    def __init__(self, model_path):
        self.processor = (
            AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
        )

    def infer_instruction_to_dino_query(
            self,
            instruction_text,
    ):
        prompt = build_instruction_to_dino_prompt(
            instruction_text
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_TO_DINO_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=DINO_MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(
                inputs.input_ids,
                generated,
            )
        ]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def infer(
        self,
        image_pil,
        instruction_payload,
        detection_payload,
        query_payload=None,
    ):
        image_pil = resize_for_qwen(image_pil)

        prompt = build_user_prompt(
            instruction_payload,
            detection_payload,
            query_payload=query_payload,
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_pil,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        text = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            self.model.device
        )

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=SCENE_MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = [
            output[len(input_ids):]
            for input_ids, output
            in zip(
                inputs.input_ids,
                generated,
            )
        ]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]