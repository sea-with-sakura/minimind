import argparse
import warnings
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    torch_dtype = None
    if args.torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == 'float16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'float32':
        torch_dtype = torch.float32

    if 'torch_model' in args.load_from:
        base = Path(args.load_from)
        config_path = (base if base.is_dir() else base.parent) / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with config_path.open('r') as f:
            config_data = json.load(f)

        model = MiniMindForCausalLM(MiniMindConfig(**config_data))
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        moe_suffix = '_moe' if config_data.get('use_moe') else ''
        hidden_size = config_data.get('hidden_size')
        pattern_first = f"{args.weight}_{hidden_size}{moe_suffix}*.pth"
        pattern_any = "*.pth"
        candidates = sorted(base.glob(pattern_first)) or sorted(base.glob(pattern_any))
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoint found under {base}")
        ckp = candidates[0]
        print(f'Loading model from {ckp} based on MiniMind...')
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{hidden_size}.pth')
    elif 'hf_model' in args.load_from:
        print(f'Loading model from {args.load_from} based on Transformers...')
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True, torch_dtype=torch_dtype
        )
    else:
        raise ValueError("Unsupported model load_from path.")

    get_model_params(model, model.config)
    model = model.eval()
    print(f'Model loaded on {args.device} with dtype {model.dtype}')

    return model.to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='./torch_model/llama-3-8B-Instruct', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--torch_dtype', default='bfloat16', choices=['auto', 'float32', 'float16', 'bfloat16'], help="模型权重精度（auto/float32/float16/bfloat16）")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: 
            print(f'👶: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': 
            templates["enable_thinking"] = True # 仅Reason模型使用
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')

if __name__ == "__main__":
    main()