import argparse
import json
import os
import sys
from pathlib import Path
from loguru import logger

# 引入 MingAudio 相关依赖（假设您的环境与原始代码一致）
import torch
import warnings
from peft import PeftModel
from transformers import AutoProcessor
import random
import numpy as np
import re
import yaml

# os.environ["PYTHONPATH"] = "Ming-UniAudio:" + os.environ.get("PYTHONPATH", "")
sys.path.append("./Ming-UniAudio")  # 确保 Ming-UniAudio 模块在 PYTHONPATH 中

# 如果 MingAudio 的模型定义在其它文件中，请确保能正确 import，或者直接将前面的 MingAudio 类定义放在本文件中
from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from processing_bailingmm import BailingMMProcessor
from configuration_bailingmm import BailingMMConfig
from sentence_manager.sentence_manager import SentenceNormalizer

warnings.filterwarnings("ignore")

def seed_everything(seed=1895):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

class MingAudio:
    """MingAudio 类 (如已在外部定义可直接 import，此处为了脚本完整性保留)"""
    def __init__(self, model_path, lora_path=None, device="cuda:0", use_grouped_gemm=False):
        self.device = device
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)

        if use_grouped_gemm and not self.model.config.llm_config.use_grouped_gemm:
            self.model.model.fuse_experts()

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model = self.model.eval().to(torch.bfloat16).to(self.device)
        self.processor = AutoProcessor.from_pretrained('./Ming-UniAudio', trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.sample_rate = self.processor.audio_processor.sample_rate
        self.patch_size = self.processor.audio_processor.patch_size
        self.normalizer = self.init_tn_normalizer(tokenizer=self.tokenizer)

    def init_tn_normalizer(self, config_file_path=None, tokenizer=None):
        if config_file_path is None:
            # 请确保该路径在您的执行环境中存在
            config_file_path = "sentence_manager/default_config.yaml"
        
        if not os.path.exists(config_file_path):
            # 如果没有文件，提供一个空的 normalizer
            return SentenceNormalizer({})

        with open(config_file_path, 'r') as f:
            self.sentence_manager_config = yaml.safe_load(f)
        
        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []
        
        assert isinstance(self.sentence_manager_config["split_token"], list)
        if tokenizer is not None:
            self.sentence_manager_config["split_token"].append(re.escape(tokenizer.eos_token))

        normalizer = SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))
        return normalizer

    def speech_edit(self, messages, output_wav_path='out.wav', use_cot=True):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.device)

        if use_cot:
            ans = torch.tensor([self.tokenizer.encode('<answer>')]).to(inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], ans], dim=1)
            attention_mask = inputs['attention_mask']
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=attention_mask.dtype)
            
        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        
        logger.info(f"input: {self.tokenizer.decode(inputs['input_ids'].cpu().numpy().tolist()[0])}")

        edited_speech, edited_text = self.model.generate_edit(
            **inputs,
            tokenizer=self.tokenizer,
            output_wav_path=output_wav_path
        )
        return edited_speech, edited_text

def get_args():
    parser = argparse.ArgumentParser(description="MingAudio Edit Batch Inference from JSONL")
    parser.add_argument("--model-path", type=str, default="inclusionAI/Ming-UniAudio-16B-A3B-Edit", help="MingAudio Edit Model path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the model on.")

    # JSONL I/O parameters
    parser.add_argument(
        "--jsonl-files", 
        type=str, 
        nargs='+', 
        required=True, 
        help="One or more input jsonl files (e.g. metadata.jsonl transcription_del.jsonl)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output_dir", 
        help="Root directory to save wav files and scp files."
    )
    
    args = parser.parse_args()
    return args

def build_edit_prompt(data):
    """
    根据输入的 JSON 数据，抽取 operation 作为 MingAudio 的提示词。
    """
    edit_kwargs = data.get("edit_kwargs", {})
    edit_type = data.get("edit_type", "")
    
    # 优先获取 "operation" (如 transcription 任务)
    operation = edit_kwargs.get("edit_prompt", "")

    # 如果都没有，给出基于 edit_type 的兜底
    if not operation:
        operation = f"Apply {edit_type} effect to the speech."

    # 按照 MingAudio 的 Prompt 模板包装
    prompt = f"<prompt>Please recognize the language of this speech and transcribe it. And {operation}\n</prompt>"
    return prompt

def process_jsonl():
    args = get_args()
    
    logger.info(f"Loading model from {args.model_path} on {args.device}...")
    model = MingAudio(model_path=args.model_path, device=args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            logger.warning(f"File not found: {jsonl_file}")
            continue

        # 获取文件前缀名 XXX (如 transcription_del)
        file_name_stem = Path(jsonl_file).stem
        
        # 构建保存目录: out_dir/XXX/
        save_dir = os.path.join(args.output_dir, file_name_stem)
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建 SCP 文件路径: out_dir/XXX.scp
        scp_path = os.path.join(args.output_dir, f"{file_name_stem}.scp")
        
        logger.info(f"--- Processing {jsonl_file} ---")
        
        with open(jsonl_file, 'r', encoding='utf-8') as fin, \
             open(scp_path, 'w', encoding='utf-8') as fscp:
             
            for line_idx, line in enumerate(fin):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"JSONDecodeError at line {line_idx}. Skipped.")
                    continue
                
                utt_id = data["id"]
                prompt_audio = data.get("audio_path", "")
                
                if not os.path.exists(prompt_audio):
                    logger.warning(f"Audio not found: {prompt_audio}. Skipped [{utt_id}].")
                    continue

                # 提取 operation 并构建 MingAudio 所需格式的 prompt
                text_prompt = data['edit_prompt']
                
                # 目标保存绝对路径 out_dir/XXX/ID.wav
                save_audio_path = os.path.abspath(os.path.join(save_dir, f"{utt_id}.wav"))

                logger.info(f"[{utt_id}] Audio: {save_audio_path}")
                
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [
                            {"type": "audio", "audio": prompt_audio, "target_sample_rate": 16000},
                            {
                                "type": "text",
                                "text": text_prompt,
                            },
                        ],
                    },
                ]

                try:
                    if os.path.exists(save_audio_path):
                        logger.info(f"[{utt_id}] Output audio already exists. Skipping inference.")
                    else:
                        response_speech, response_text = model.speech_edit(
                            messages=messages, 
                            output_wav_path=save_audio_path
                        )
                        logger.info(f"[{utt_id}] Successfully edited. Model Output Text: {response_text}")
                    
                    # 写入 SCP（格式：id 绝对路径）
                    fscp.write(f"{utt_id}\t{save_audio_path}\n")
                    fscp.flush()
                    
                    
                except Exception as e:
                    logger.error(f"Failed to process {utt_id}: {e}")

        logger.info(f"--- Finished {file_name_stem}. Results saved to {save_dir} and {scp_path} ---")

if __name__ == "__main__":
    process_jsonl()