import torch
from PIL import Image
from pathlib import Path
import argparse
from dataloader.data import *
from models import condition
from models.ad_llava import ADLlavaModel

from transformers import AutoProcessor, LlavaForConditionalGeneration

from peft import get_peft_model
from train_utils.llava_trainer import LLaVATrainer
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LEGNTH = 1024


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    group_by_modality_length: bool = field(default=False)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('generate verbose images')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    parser.add_argument('--root_path', type=str, default="", help='default path')
    parser.add_argument('--data_path', type=str, default="", help='data path')
    parser.add_argument('--image_folder', type=str, default="", help='data path')
    parser.add_argument('--model_path', type=str, default="", help='model path')
    parser.add_argument('--deepspeed_config', type=str, default="", help='model path')
    parser.add_argument('--train_stage', type=int, default=1, help='training stage I or II')
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    use_lora = True

    processor = AutoProcessor.from_pretrained(args.model_path)
    data_args = DataArguments()
    data_args.image_processor = processor.image_processor
    data_args.is_multimodal = True
    data_args.data_path = args.data_path
    data_args.image_folder = args.image_folder
    data_args.mm_use_im_start_end = False
    
    print(data_args.image_folder)
    
    
    data_args.train_stage = args.train_stage
    data_module = make_supervised_data_module(tokenizer=processor.tokenizer,
                                            data_args=data_args)
    if args.train_stage == 1:

        training_args = TrainingArguments(
            output_dir=args.root_path + '/results',
            #use_cpu=True,
            per_device_train_batch_size=2,
            #per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Increase as needed based on your GPU memory
            num_train_epochs=3,
            learning_rate=1e-05,
            #fp16=True,
            #fp16_backend="auto",
            bf16=True,
            warmup_steps=20,
            logging_dir=args.root_path + "/log",
            #logging_steps=20,
            deepspeed=args.deepspeed_config,
            group_by_modality_length=False
        )

        # 激活指定参数，使其可以训练
        model = ADLlavaModel.from_pretrained(args.model_path,attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to('cuda')
        for param in model.parameters():
            param.requires_grad = False
        model.ad_query_tokens.requires_grad = True
        model.ad_query_tokens_2.requires_grad = True

        for param in model.mm_projector.parameters():
            param.requires_grad = True
        for param in model.ad_part.parameters():
            param.requires_grad = True

        trainer = LLaVATrainer(model=model,
                            tokenizer=processor.tokenizer,
                            args=training_args,
                            **data_module)
        trainer.train()

        print("Training finished.")
        trainer.save_model(args.root_path + "/Final")
        print("Model saved to: {}".format(args.root_path + "/Final")) 

    elif args.train_stage == 2:
        training_args = TrainingArguments(
            output_dir=args.root_path + '/results',
            #use_cpu=True,
            per_device_train_batch_size=2,
            #per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Increase as needed based on your GPU memory
            num_train_epochs=3,
            learning_rate=1e-05,
            #fp16=True,
            #fp16_backend="auto",
            bf16=True,
            warmup_steps=300,
            logging_dir=args.root_path + "/log",
            #logging_steps=20,
            deepspeed=args.deepspeed_config,
            group_by_modality_length=False
        )

        # model load ....
        model = ADLlavaModel.from_pretrained("./models/weights",attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to('cuda')
    
        for param in model.parameters():
            param.requires_grad = False

        # peft lora
        if use_lora:
            def find_all_linear_names(model):
                cls = torch.nn.Linear
                lora_module_names = set()
                multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler',\
                                    'ad_part','multi_modal_projector']
                for name, module in model.named_modules():
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

                if 'lm_head' in lora_module_names: # needed for 16-bit
                    lora_module_names.remove('lm_head')
                print(lora_module_names)
                return list(lora_module_names)
            
            lora_config = LoraConfig(
                r=256,
                #target_modules=[".*(language_model).*(_proj)$"],
                target_modules=find_all_linear_names(model),
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            lora_trainer = LLaVATrainer(model=model,
                                tokenizer=processor.tokenizer,
                                args=training_args,
                                **data_module)
            lora_trainer.train()

            print("Training finished.")
            lora_trainer.save_model(args.root_path + "/Final")
            print("Model saved to: {}".format(args.root_path + "/Final")) 
        else:

            for param in model.language_model.parameters():
                param.requires_grad = True

            trainer = LLaVATrainer(model=model,
                                tokenizer=processor.tokenizer,
                                args=training_args,
                                **data_module)
            trainer.train()

            print("Training finished.")
            trainer.save_model(args.root_path + "/Final")
            print("Model saved to: {}".format(args.root_path + "/Final")) 

if __name__ == '__main__':
    args = parse_args()
    main(args)

