import os
import re
import csv, json
import argparse
import subprocess
import time
from typing import List
import sys

import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, LlavaConfig

from reg_dataset import REGDataset, ResultDataset, ChallengeDataset
from report_evaluator import ScoreEvaluator
MODEL_BASE_DIR = "/home/hanjh/longitudinal_llm/"

sys.path.append(os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA"))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from llava.model import LlavaLlamaForCausalLM

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

class TextGenerationModel:
    def __init__(self, generation_args: dict, encoder_type: str, llm_type:str, device: str = "cuda"):
        """
        모델과 토크나이저 설정, 디바이스 설정
        
        Args:
            generation_args (dict): 답변 생성 시 사용할 args 설정
            encoder_type (str): 인코더 모델 타입 ("base", "finetuned", "aligned")
            llm_type (str): llm 모델 타입 ("base", "finetuned", "aligned")
            device (str, optional): 사용할 디바이스
        """
        self.generation_args = generation_args
        self.device = device

        self.model = None
        self.processor = None
        self.tokenizer = None

        if llm_type == "base":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b")
        elif llm_type == "base_prev":
            self.model_path = os.path.join(MODEL_BASE_DIR, "llava-hf/llava-1.5-7b-hf")
        elif llm_type == "zeroshot":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/llava-hf/llava-1.5-7b-hf-lora-zeroshot")
        elif llm_type == "t_report":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b-lora-t_report")
        elif llm_type == "t_report-8":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b-lora-t_report-8")
        elif llm_type == "t_report-8_nonaligned":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b-lora-t_report-8_nonaligned")
        elif llm_type == "all":
            self.model_path = os.path.join(MODEL_BASE_DIR, "finetuning/LLaVA/checkpoints/liuhaotian/llava-v1.5-7b-lora-all")
        else:
            raise ValueError("Invalid type. Choose from ['base', 'zeroshot', 'fewshot', 't_1_report', 't_report'].")
        
        self._set_model()

        if encoder_type == "base":
            print("No pretrained image encoders")
            self.encoder_path = "None"
        elif "lora" in self.model_path:
            print("Using finetuned image encoder")
            self.encoder_path = "aligned"
        else:
            print(f"Replacing image encoders with {encoder_type} encoders...")
            if encoder_type == "finetuned":
                self.encoder_path = os.path.join(MODEL_BASE_DIR, "model/model_epoch_5_val_loss_0.2764_llava_v2.pth")
            elif encoder_type == "aligned" or encoder_type == "aligned_test_2":
                self.encoder_path = os.path.join(MODEL_BASE_DIR, "model/llava_epoch_5_based_alignment_nf_cxr.pt")
            elif encoder_type == "aligned_test_1":
                self.encoder_path = os.path.join(MODEL_BASE_DIR, "model/1_epoch_loss_3.6022_align_model_5_2ver_llava_v2.pt")
            # self._set_encoder(self.encoder_path, encoder_type)
            self._set_encoder(self.encoder_path, encoder_type)

    def _set_model(self):
        if "liuhaotian" in self.model_path:
            self.model_name = get_model_name_from_path(self.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                self.model_path, "liuhaotian/llava-v1.5-7b", self.model_name
            )
        elif self.model_path == "llava-hf/llava-1.5-7b-hf":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _set_encoder(self, encoder_path: str, encoder_type: str):
        print(f"encoder_path: {encoder_path}")
        # clip_model = torch.load('/home/hanjh/longitudinal_llm/model/llava_med_vision/llava_med_vision_model.pt')
        # image_weight = torch.load(encoder_path, map_location=self.device)
        
        # new_sd = {}
        # if "aligned" in encoder_type:
        #     for ckpt_key, ckpt_value in image_weight.items():
        #         if ckpt_key.startswith("image_encoder.vision_model."):
        #             new_key = ckpt_key.replace("image_encoder.", "")
        #             new_sd[new_key] = ckpt_value
        #         if ckpt_key.startswith("graph_encoder."):
        #             break
        # elif "finetuned" in encoder_type:
        #     for ckpt_key, ckpt_value in image_weight.items():
        #         if ckpt_key.startswith("visual_encoder."):
        #             new_key = ckpt_key.replace("visual_encoder.", "")
        #             new_sd[new_key] = ckpt_value
        # else:
        #     raise ValueError("Invalid type. Choose from ['base', 'finetuned', 'aligned'].")

        # clip_model.load_state_dict(new_sd, strict = True)
        # clip_model = clip_model.to(self.device)
        # print(self.model.vision_tower())
        # self.model.get_vision_tower().vision_tower.vision_model = clip_model.vision_model
        # torch.cuda.empty_cache()
        # self.model = self.model.bfloat16().to(self.device)
    
        vision_tower = self.model.get_vision_tower()
        print(vision_tower.vision_tower.vision_model.encoder.layers[23].mlp.fc2.weight[10][:20])

        clip_model = torch.load(os.path.join(MODEL_BASE_DIR, 'model/llava_med_vision/llava_med_vision_model.pt'))
        print(clip_model.vision_model.encoder.layers[23].mlp.fc2.weight[10][:20])
        image_weight = torch.load(encoder_path, map_location=self.device)

        new_sd = {}
        for ckpt_key, ckpt_value in image_weight.items():
            if ckpt_key.startswith("image_encoder.vision_model."):
                new_key = ckpt_key.replace("image_encoder.", "")
                new_sd[new_key] = ckpt_value
            if ckpt_key.startswith("graph_encoder."):
                break

        clip_model.load_state_dict(new_sd, strict = False)
        vision_tower.vision_tower.vision_model = clip_model.vision_model
        print(vision_tower.vision_tower.vision_model.encoder.layers[23].mlp.fc2.weight[10][:20])
        vision_tower.to(dtype=torch.bfloat16, device=self.device)

    def generate_text(self, image: Image, prompt_type: str):
        if "liuhaotian" in self.model_path:
            content = self.create_prompt(prompt_type=prompt_type)

            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in content:
                if self.model.config.mm_use_im_start_end:
                    content = re.sub(IMAGE_PLACEHOLDER, image_token_se, content)
                else:
                    content = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, content)
            else:
                if self.model.config.mm_use_im_start_end:
                    content = image_token_se + "\n" + content
                else:
                    content = DEFAULT_IMAGE_TOKEN + "\n" + content

            if "llama-2" in self.model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in self.model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in self.model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in self.model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images = [image]
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    **self.generation_args,
                    use_cache=True,
                )

            generated_report = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            return content, generated_report
        else:
            content = self.create_prompt(prompt_type=prompt_type)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {"type": "image"},
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)

            output = self.model.generate(
                **inputs,  
                **self.generation_args
            )
            decoded_output = self.processor.decode(output[0][2:], skip_special_tokens=True)
            generated_report = decoded_output.split("ASSISTANT: ")[-1].strip()

            return content, generated_report

    def create_prompt(self, prompt_type: str) -> str:
        base_prompt = f"You are a skilled pathologist. You are given a histopathological image from a biopsy. Analyze the image and generate a pathology report using the format below. Use standardized medical terms and avoid unnecessary explanation."
        
        format_prompt = "\n\n# Answer must be in the following format: <Anatomical Site>, <Procedure Type>; <Pathological Diagnosis>"
        
        value_example = """

# Possible values for <Anatomical Site>:
- Breast
- Nipple
- Urinary bladder
- Uterine cervix
- Colon
- Rectum
- Anus
- Lung
- Prostate
- Stomach
- Bladder
- Cervix
- ... (and other anatomical sites)

# Possible values for <Procedure Type>:
- core-needle biopsy
- sono-guided core biopsy
- sono-guided mammotome biopsy
- encor biopsy
- mammotome biopsy
- vacuum-assisted biopsy
- sono-guided biopsy
- punch biopsy
- stereotactic biopsy
- biopsy
- transurethral resection
- colposcopic biopsy
- polypectomy biopsy
- frozen biopsy
- colonoscopic biopsy
- colonoscopic polypectomy
- colonoscopic mucosal resection
- colonoscopic submucosal dissection
- incisional biopsy
- endoscopic biopsy
- endoscopic submucosal dissection
- endoscopic mucosal resection
- Loop Electrosurgical Excision Procedure
- ... (and other procedure types)

# Possible values for <Pathological Diagnosis>:
- 1. Acinar adenocarcinoma, Gleason's score 6 (3+3), grade group 1, tumor volume: 20%2. Chronic granulomatous inflammation without necrosis 
- 1. Columnar cell lesion2. Apocrine metaplasia3. Usual ductal hyperplasia
- 1. Ductal carcinoma in situ - Type: Cribriform - Nuclear grade: Intermediate - Necrosis: Absent2. Microcalcification
- 1. Lobular carcinoma in situ2. Intraductal papilloma with usual ductal hyperplasia
- 1. Micro-invasive carcinoma2. Ductal carcinoma in situ
- 1. Non-invasive papillary urothelial carcinoma, low grade2. Urothelial carcinoma in situ Note) The specimen includes muscle proper.
- 1. Papillary neoplasm2. Atypical ductal hyperplasia
- 1. Usual ductal hyperplasia2. Apocrine metaplasia3. Sclerosing adenosis
- Acinar adenocarcinoma, Gleason's score 10 (5+5), grade group 5, tumor volume: 10%
- Apocrine metaplasia
- Atypical ductal hyperplasia
- Carcinoid/neuroendocrine tumor, NOS
- Chronic active cervicitis
- Ductal carcinoma in situ- Type: Cribriform and micropapillary- Nuclear grade: Intermediate- Necrosis: Absent
- Ductal carcinoma in situ- Type: Flat- Nuclear grade: High- Necrosis: Absent
- Ductal carcinoma in situ- Type: Solid and papillary- Nuclear grade: Intermediate- Necrosis: Absent
- Endocervical adenocarcinoma in situ (AIS), HPV-associated
- Fibroepithelial tumor, favor fibroadenoma
- Inflammatory polyp
- Intraductal papilloma with usual ductal hyperplasia
- Invasive carcinoma of no special type, grade I (Tubule formation: 1, Nuclear grade: 1, Mitoses: 1)
- Invasive carcinoma of no special type, grade III (Tubule formation: 3, Nuclear grade: 2, Mitoses: 3)
- Invasive carcinoma with features of mucinous carcinoma
- No tumor presentNote) The specimen includes muscle proper.
- Non-invasive papillary urothelial carcinoma, high grade Note) The specimen does not include muscle proper.
- Non-invasive papillary urothelial carcinoma, low grade Note) The specimen includes muscle proper.
- Non-small cell carcinoma, favor adenocarcinoma
- Papillary neoplasm with atypical ductal hyperplasia
- Pseudoangiomatous stromal hyperplasia
- Sclerosing adenosis
- Sessile serrated lesion with low grade dysplasia
- Squamous cell carcinoma
- Traditional serrated adenoma
- Tubular adenoma with high grade dysplasia
- Tubulovillous adenoma with high grade dysplasi
- Urothelial carcinoma in situ Note) The specimen does not include muscle proper.
- Usual ductal hyperplasia
- ... (and other pathological diagnoses)
"""

        fewshot_examples = """

# Medical Report Examples:
## example 1:
Breast, sono-guided core biopsy;   Invasive carcinoma of no special type, grade II (Tubule formation: 3, Nuclear grade: 3, Mitoses: 1)

## example 2:
Colon, colonoscopic biopsy;   Tubular adenoma with low grade dysplasia

## example 3:
Bladder, transurethral resection;   Invasive urothelial carcinoma, with involvement of subepithelial connective tissue  Note) The specimen does not include muscle proper.
"""
        take_a_deep_breath = "\n\nTake a deep breath and work through this carefully. This is a complex strategic decision that requires examining our assumptions and considering multiple layers of implications."

        answering_prompt = f"\n\n# Predicted Medical Report (Your Answer):"

        if prompt_type == "zeroshot" or prompt_type == "zeroshot_all":
            return f"{base_prompt}{format_prompt}{answering_prompt}{take_a_deep_breath}{answering_prompt}"

        elif prompt_type == "fewshot" or prompt_type == 'fewshot_all':
            return f"{base_prompt}{format_prompt}{value_example}{fewshot_examples}{take_a_deep_breath}{answering_prompt}"
        
        else:
            raise ValueError("Invalid type. Choose from ['zeroshot', 'fewshot'].")


class ReportGenerator:
    def __init__(self, dataset: REGDataset, model: TextGenerationModel, prompt_type: str):
        """
        의료 보고서 생성기 초기화
        
        Args:
            dataset: 데이터셋
            model (TextGenerationModel): 텍스트 생성 모델
        """
        self.dataset = dataset
        self.model = model
        self.prompt_type = prompt_type

    def generate_reports(self, save_im_file=False) -> List[dict]:
        """
        의료 보고서 생성

        Returns:
            results (List[dict]): 생성된 의료 보고서 결과
        """
        results = []
        # file_path = f"temp/temp_report_im_{self.prompt_type}_{os.path.splitext(os.path.basename(self.model.encoder_path))[0]}_{os.path.basename(self.model.model_path)}.csv"
        # temp_data = pd.read_csv(file_path, header=None)

        if self.prompt_type in ['zeroshot', 'fewshot']:
            for idx in tqdm(range(len(self.dataset)), desc="Generating Medical Report", unit="idx"):

                data = self.dataset[idx]

                # if not temp_data.empty:
                #     prompt = self.model.create_prompt(
                #         prompt_type=self.prompt_type)
                #     generated_text = temp_data.iloc[0, 0]
                #     temp_data = temp_data.iloc[1:].reset_index(drop=True)

                prompt, generated_text = self.model.generate_text(
                    image=data['image'],
                    prompt_type=self.prompt_type
                )

                results.append(self._create_result(data, prompt, generated_text))

                if save_im_file:
                    self._save_temp_im_results(generated_text)

        else:
            raise ValueError("Invalid type. Choose from ['zeroshot', 'fewshot'].")

        return results

    def _extract_text(self, text):
        lines = text.split("\n")
        findings = ""
        impression = ""
        start_findings = None
        start_impression = None

        for i, line in enumerate(lines):
            if "FINDINGS:" in line:
                start_findings = i
            elif "IMPRESSION:" in line:
                start_impression = i
                break

        if start_findings is not None:
            for line in lines[start_findings:]:
                if start_impression is not None and start_impression <= start_findings:
                    break
                if "IMPRESSION:" in line:
                    break
                findings += line.strip() + " "

        if start_impression is not None:
            for line in lines[start_impression:]:
                impression += line.strip() + " "

        # 여러 개의 연속된 공백을 단일 공백으로 대체
        findings = re.sub(r'\s+', ' ', findings)
        impression = re.sub(r'\s+', ' ', impression)

        return findings.strip(), impression.strip()

    def _create_result(self, data, prompt, generated_text) -> dict:
        """
        결과 딕셔너리 생성
        
        Args:
            data: 현재 데이터
            prompt: 생성된 프롬프트
            generated_text: 생성된 텍스트
            
        Returns:
            dict: 결과 딕셔너리
        """
        return {
            "id": data["id"],
            "prompt": prompt,
            "gt_report": data["gt_report"],
            "report": generated_text,
        }

    def _save_temp_im_results(self, generated_text):
        """
        중간 결과를 CSV 파일에 저장하는 함수
        """
        temp_path =  f"temp/temp_report_im.csv"
        with open(temp_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([generated_text])

    def save_results(self, results: List[dict], result_path:str) -> str:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)

        df.to_csv(result_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"[Completed] Result Saved at \"{result_path}\"")

        filtered_results = [{"id": r["id"], "report": r["report"]} for r in results]
        json_path = "/output/text-report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)

        print(f"[Completed] Result Saved at \"{json_path}\"")

"""
원래 실행 파일
"""

# def main(args):
#     print("##########################################################################")
#     print("ARGS SETTING")
#     print(f"csv_file_path: {args.csv_file_path}")
#     print(f"prompt_type: {args.prompt_type}")
#     print(f"encoder_type: {args.encoder_type}")
#     print(f"llm_type: {args.llm_type}")
#     print(f"image_type: {args.image_type}")
#     print("##########################################################################")

#     generation_args = {
#         "max_new_tokens": 1000, 
#         "num_beams": 3,
#         "temperature": 0.5, 
#         "do_sample": True,
#         "top_p": 0.9,
#     }

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     dataset = REGDataset(csv_file_path=args.csv_file_path, img_type=args.image_type, transform=None)
#     model = TextGenerationModel(generation_args=generation_args, 
#                                 encoder_type=args.encoder_type,
#                                 llm_type=args.llm_type,
#                                 device=args.device)
    
#     report_generator = ReportGenerator(dataset, model, prompt_type=args.prompt_type)
    
#     results = report_generator.generate_reports(save_im_file=True)
#     result_path = f"results/result_{args.prompt_type}_{args.encoder_type}_{args.llm_type}_{args.image_type}.csv"
#     report_generator.save_results(results, result_path)

#     dataset = ResultDataset(csv_file_path=result_path)
    
#     evaluator = ScoreEvaluator(dataset, result_path)
#     evaluator.get_scores()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate medical reports using LLM")
#     parser.add_argument("-f", "--csv_file_path", type=str, default="data/REG_data.csv", help="Path to the CSV file")
#     parser.add_argument("-pt", "--prompt_type", type=str, default="fewshot_all", help="Type of prompt. Choose from ['zeroshot', 'fewshot'].")
#     parser.add_argument("-et", "--encoder_type", type=str, default="base", help="Type of encoder. Choose from ['base', 'finetuned', 'aligned'].")
#     parser.add_argument("-lt", "--llm_type", type=str, default="base", help="Type of LLM. Choose from ['base', 'zeroshot', 'fewshot', 't_1_report', 't_report'].")
#     parser.add_argument("-it", "--image_type", type=str, default="tiff", help="Type of image. Choose from ['tiff', '224', '224_padding'].")
#     parser.add_argument("-d", "--device", type=str, default="cuda:0")

#     args = parser.parse_args()
#     main(args)

"""
REG2025용 실행 코드
"""

def main():
    print("##########################################################################")
    print("ARGS SETTING")
    print(f"prompt_type: {args.prompt_type}")
    print(f"encoder_type: {args.encoder_type}")
    print(f"llm_type: {args.llm_type}")
    print(f"image_type: {args.image_type}")
    print("##########################################################################")

    generation_args = {
        "max_new_tokens": 1000, 
        "num_beams": 3,
        "temperature": 0.5, 
        "do_sample": True,
        "top_p": 0.9,
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ChallengeDataset(transform=None)

    model = TextGenerationModel(generation_args=generation_args, 
                                encoder_type=args.encoder_type,
                                llm_type=args.llm_type)
    
    report_generator = ReportGenerator(dataset, model, prompt_type=args.prompt_type)
    
    results = report_generator.generate_reports()

    filtered_results = [{"id": r["id"], "report": r["report"]} for r in results]
    json_path = "/output/text-report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)

    print(f"[Completed] Result Saved at \"{json_path}\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate medical reports using LLM")
    parser.add_argument("-pt", "--prompt_type", type=str, default="fewshot", help="Type of prompt. Choose from ['zeroshot', 'fewshot'].")
    parser.add_argument("-et", "--encoder_type", type=str, default="base", help="Type of encoder. Choose from ['base', 'finetuned', 'aligned'].")
    parser.add_argument("-lt", "--llm_type", type=str, default="base", help="Type of LLM. Choose from ['base', 'zeroshot', 'fewshot', 't_1_report', 't_report'].")
    parser.add_argument("-it", "--image_type", type=str, default="tiff", help="Type of image. Choose from ['tiff', '224', '224_padding'].")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)