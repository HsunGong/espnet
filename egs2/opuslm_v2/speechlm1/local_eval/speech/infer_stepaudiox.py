import argparse
import json
import os
import torchaudio
from pathlib import Path
import sys

from tqdm import tqdm

os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
# os.environ["PYTHONPATH"] = "Step-Audio-EditX:" + os.environ.get("PYTHONPATH", "")
sys.path.append("Step-Audio-EditX")  # 确保 Step-Audio-EditX 模块在 PYTHONPATH 中

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS


def get_args():
    parser = argparse.ArgumentParser(description="Step-Audio EditX Batch Inference from JSONL")
    parser.add_argument("--model-path", type=str, default="Step-Audio-EditX/Step-Audio-EditX", help="Model path.")

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

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="Step-Audio-EditX/Step-Audio-Tokenizer",
        help="Path to Step-Audio-Tokenizer directory. If not specified, auto-detects sibling directory"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "fp8"],
        help="Enable quantization for vLLM: awq, gptq, or fp8"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization ratio 0.0-1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=3072,
        help="Maximum model sequence length, affects KV cache size (default: 8192)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for model (default: bfloat16)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graphs to save ~0.5GB GPU memory (slower inference)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        help="KV cache data type: fp8_e5m2 reduces KV cache memory by ~50%% (default: auto, uses model dtype)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences (default: 256, lower = less memory)"
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration (default: max_model_len, lower = less activation memory)"
    )
    
    # CosyVoice vocoder parameters
    parser.add_argument(
        "--cosyvoice-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="CosyVoice vocoder dtype: bfloat16 reduces memory by ~50%% (default: float32)"
    )
    parser.add_argument(
        "--no-cosyvoice-cuda-graph",
        dest="cosyvoice_cuda_graph",
        action="store_false",
        help="Disable CUDA Graph for CosyVoice vocoder (saves memory but slower)"
    )
    parser.set_defaults(cosyvoice_cuda_graph=True)

    args = parser.parse_args()
    return args


def load_model(args) -> StepAudioTTS:
    step_audio_editx_model_path = args.model_path
    step_audio_tokenizer_path = args.tokenizer_path

    step_audio_tokenizer = StepAudioTokenizer(
        step_audio_tokenizer_path,
        model_source=args.model_source
    )
    step_audio_editx = StepAudioTTS(
        step_audio_editx_model_path,
        step_audio_tokenizer,
        model_source=args.model_source,
        tts_model_id=args.tts_model_id,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        cosyvoice_dtype=args.cosyvoice_dtype,
        cosyvoice_cuda_graph=args.cosyvoice_cuda_graph
    )
    return step_audio_editx


def map_jsonl_to_step_audio_params(data):
    """
    根据输入的 JSONL 字典数据，映射出 Step-Audio 需要的参数
    返回: (step_edit_type, step_edit_info, target_text)
    """
    jsonl_edit_type = data.get("edit_type", "")
    kwargs = data.get("edit_kwargs", {})
    target_text = data["target_text"]

    step_edit_type = None
    step_edit_info = ""

    # 1. 文本增删改 (对应 Step-Audio 的 Zero-Shot Cloning)
    if jsonl_edit_type in ["transcription_ins", "transcription_del", "transcription_sub", "transcription_replace_sentence"]:
        step_edit_type = "clone"
    
    # 2. 插入副语言标签
    elif jsonl_edit_type == "transcription_add_paralinguistic":
        step_edit_type = "paralinguistic"
    
    # 3. 说话风格 (whisper 等)
    elif jsonl_edit_type == "style_whisper":
        step_edit_type = "style"
        step_edit_info = kwargs["style"]
    
    # 4. 情感
    elif jsonl_edit_type == "style_emotion":
        step_edit_type = "emotion"
        step_edit_info = kwargs["style"]
    
    # 5. 语速控制
    elif jsonl_edit_type == "audio_effect_speed":
        step_edit_type = "speed"
        rate = kwargs.get("speed_rate", 1.0)
        # 将倍率粗略映射到 Step-Audio 支持的档位
        if rate >= 1.5:
            step_edit_info = "more faster"
        elif rate > 1.0:
            step_edit_info = "faster"
        elif rate <= 0.8:
            step_edit_info = "more slower"
        else:
            step_edit_info = "slower"
    elif jsonl_edit_type == "audio_effect_dereverb":
        step_edit_type = "denoise"

    # Step-Audio 不原生支持的音效（如 reverb, volume, pitch），可根据实际情况选择是否补充额外逻辑
    # 若不支持则返回 None 忽略处理
    # print(">>>>", step_edit_type, step_edit_info, target_text)
    return step_edit_type, step_edit_info, target_text


def process_jsonl():
    args = get_args()
    model = load_model(args)
    
    os.makedirs(args.output_dir, exist_ok=True)

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"[Warning] File not found: {jsonl_file}")
            continue

        # 获取文件前缀名 XXX (如 transcription_del)
        file_name_stem = Path(jsonl_file).stem
        
        # 1. 构建目录: out_dir/XXX/
        save_dir = os.path.join(args.output_dir, file_name_stem)
        os.makedirs(save_dir, exist_ok=True)
        
        # 2. 构建 SCP 文件路径: out_dir/XXX.scp
        scp_path = os.path.join(args.output_dir, f"{file_name_stem}.scp")
        
        print(f"--- Processing {jsonl_file} ---")
        
        with open(jsonl_file, 'r', encoding='utf-8') as fin, \
             open(scp_path, 'w', encoding='utf-8') as fscp:
            lines = fin.readlines()
            for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"JSONDecodeError at line {line_idx}. Skipped.")
                    continue
                
                utt_id = data["id"]
                prompt_audio = data.get("audio_path", "")
                prompt_text = data.get("text", "")
                
                if not os.path.exists(prompt_audio):
                    print(f"[Warning] Audio not found: {prompt_audio}. Skipped.")
                    continue

                step_type, step_info, target_text = map_jsonl_to_step_audio_params(data)

                if step_type is None:
                    print(f"[{utt_id}] Ignored unsupported edit_type '{data.get('edit_type')}'.")
                    continue
                
                # 目标保存绝对路径 out_dir/XXX/ID.wav
                save_audio_path = os.path.abspath(os.path.join(save_dir, f"{utt_id}.wav"))

                try:
                    print(f"[{utt_id}] Running {step_type} | Info: {step_info} | Expect to save Audio: {save_audio_path}")

                    if os.path.exists(save_audio_path):
                        print(f"[{utt_id}] Output already exists at {save_audio_path}. Skipped.")
                    else:
                        if step_type == "clone":
                            output_audio, output_sr = model.clone(
                                prompt_wav_path=prompt_audio,
                                prompt_text=prompt_text,
                                target_text=target_text
                            )
                        else:
                            output_audio, output_sr = model.edit(
                                prompt_wav_path=prompt_audio,
                                prompt_text=prompt_text,
                                target_text=target_text if step_type == "paralinguistic" else "",
                                edit_type=step_type,
                                edit_info=step_info
                            )

                        torchaudio.save(save_audio_path, output_audio.cpu(), output_sr)

                    fscp.write(f"{utt_id}\t{save_audio_path}\n")
                    fscp.flush()

                except Exception as e:
                    print(f"[Error] Failed to process {utt_id}: {e}")

        print(f"--- Finished {file_name_stem}. Results saved to {save_dir} and {scp_path} ---")


if __name__ == "__main__":
    process_jsonl()
