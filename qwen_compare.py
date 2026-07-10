import os
import time
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_QWEN_DIR = "merge_qwen"
BASE_QWEN_MODEL = LOCAL_QWEN_DIR
PROMPT = "Explain machine learning in simple words."
MAX_NEW_TOKENS = 128


def ensure_tokenizer_config(model_path: str):
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {token: token for token in extra}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def load_model(model_path: str, use_auth_token=None, dtype=torch.float16, device_map="auto", trust_remote_code=False):
    ensure_tokenizer_config(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_auth_token=use_auth_token,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def run_inference(tokenizer, model, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, latency


def print_section(title: str):
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference comparison on device: {device_name}")
    print(f"Prompt: {PROMPT}")

    auth_token = os.environ.get("HUGGINGFACE_TOKEN")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # Load local merged Qwen model
    print_section("Local merged Qwen model")
    local_dir = os.path.join(os.getcwd(), LOCAL_QWEN_DIR)
    if not os.path.isdir(local_dir):
        print(f"Local model directory not found: {local_dir}")
        return

    local_tokenizer, local_model = load_model(
        local_dir,
        use_auth_token=None,
        dtype=dtype,
        device_map=device_map,
    )
    local_output, local_latency = run_inference(local_tokenizer, local_model, PROMPT, MAX_NEW_TOKENS)
    print(f"Model path: {local_dir}")
    print(f"Latency: {local_latency:.4f}s")
    print("Output:")
    print(local_output)

    # Load base Qwen 0.5 instruct model
    print_section("Base Qwen 0.5 instruct model")
    base_model_source = os.environ.get("BASE_QWEN_MODEL", BASE_QWEN_MODEL)
    if os.path.isdir(base_model_source):
        print(f"Using local base Qwen model: {base_model_source}")
        trust_remote_code = False
    else:
        print(f"Using remote base Qwen model: {base_model_source}")
        trust_remote_code = True

    try:
        base_tokenizer, base_model = load_model(
            base_model_source,
            use_auth_token=auth_token,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        base_output, base_latency = run_inference(base_tokenizer, base_model, PROMPT, MAX_NEW_TOKENS)
        print(f"Model id: {base_model_source}")
        print(f"Latency: {base_latency:.4f}s")
        print("Output:")
        print(base_output)
    except Exception as exc:
        print(f"Failed to load or run base Qwen model: {exc}")
        if base_model_source != LOCAL_QWEN_DIR and os.path.isdir(LOCAL_QWEN_DIR):
            print(f"Falling back to local base Qwen model: {LOCAL_QWEN_DIR}")
            base_tokenizer, base_model = load_model(
                LOCAL_QWEN_DIR,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=False,
            )
            base_output, base_latency = run_inference(base_tokenizer, base_model, PROMPT, MAX_NEW_TOKENS)
            print(f"Model id: {LOCAL_QWEN_DIR}")
            print(f"Latency: {base_latency:.4f}s")
            print("Output:")
            print(base_output)
        else:
            return

    print_section("Comparison")
    print(f"Local merged Qwen latency: {local_latency:.4f}s")
    print(f"Base Qwen 0.5 instruct latency: {base_latency:.4f}s")
    print("\nLocal merged Qwen output:\n")
    print(local_output)
    print("\nBase Qwen 0.5 instruct output:\n")
    print(base_output)


if __name__ == "__main__":
    main()
