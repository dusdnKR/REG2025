import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import glob
Image.MAX_IMAGE_PIXELS = None

class ChallengeDataset:
    def __init__(self, input_dir="/input/images/he-staining/", transform=None):
        self.image_paths = glob.glob(f"{input_dir}/*.tif")
        self.transform = transform
        print(f"[Loaded] {len(self.image_paths)} test images from {input_dir}")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.split("/")[-1].split(".")[0]
        gt_report = ""

        image = Image.open(image_path).convert("RGB")
        image.thumbnail((12000, 12000), Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)

        return {
            "id": image_id,
            "gt_report": gt_report,
            "image": image,
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.image_paths)

class ImageREGDataset:
    def __init__(self, csv_file_path: str, transform: transforms.Compose = None, img_type: str = '224'):
        """
        REG Image 데이터 불러오기

        Args:
            csv_file_path (str): 읽을 CSV 파일의 경로
            transform (transforms.Composem, optional): PubMedCLIP 코드에 있던 transform
        """
        self.csv_file_path = csv_file_path
        self.transform = transform
        self.img_type = img_type
        self.test_data = pd.read_csv(csv_file_path, usecols=[f"image_path_{self.img_type}"])
        print("[Completed] Image Dataset Loaded")
        print(f"Total Test Samples: {len(self.test_data)}")

    def _get_image(self, idx: int) -> torch.Tensor:
        image_path = self.test_data.iloc[idx][f"image_path_{self.img_type}"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def __getitem__(self, idx: int) -> dict:
        image = self._get_image(idx)
        
        return {
            "image": image
        }

    def __len__(self):
        return len(self.test_data)
    
class REGDataset:
    def __init__(self, csv_file_path: str, img_type: str = '224', transform: transforms.Compose = None):
        """
        REG 데이터 불러오기

        Args:
            csv_file_path (str): 읽을 CSV 파일의 경로
            embedding_path (str): 이미지 임베딩 모델의 경로(이름)
        """
        self.csv_file_path = csv_file_path
        self.transform = transform
        self.img_type = img_type
        self.test_data = pd.read_csv(csv_file_path, usecols=["id", f"image_path_{self.img_type}", "report"])
        print("[Completed] Dataset Loaded")
        print(f"Total Test Samples: {len(self.test_data)}")

    def _get_id(self, idx: int) -> str:
        id = self.test_data.iloc[idx]['id']
        return id

    def _get_gt_report(self, idx: int) -> str:
        gt_report = self.test_data.iloc[idx]['report']
        return gt_report

    def _get_image(self, idx: int) -> torch.Tensor:
        image_path = self.test_data.iloc[idx][f"image_path_{self.img_type}"]
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((12000, 12000), Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def __getitem__(self, idx: int) -> dict:
        id = self._get_id(idx)
        gt_report = self._get_gt_report(idx)
        image, image_path = self._get_image(idx)
        
        return {
            "id": id,
            "gt_report": gt_report,
            "image": image,
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.test_data)
    
class ResultDataset:
    def __init__(self, csv_file_path: str):
        """
        결과 데이터 불러오기

        Args:
            csv_file_path (str): 읽을 CSV 파일의 경로
        """
        self.csv_file_path = csv_file_path
        self.result_data = pd.read_csv(csv_file_path, usecols=["id", "prompt", "gt_report", "report"])
        print("[Completed] Result Data Loaded")
        print(f"Total Test Samples: {len(self.result_data)}")

    def _get_id(self, idx: int) -> str:
        id = self.result_data.iloc[idx]['id']
        return id
    
    def _get_gt_report(self, idx: int) -> str:
        gt_report = self.result_data.iloc[idx]['gt_report']
        return gt_report
    
    def _get_pred_report(self, idx: int) -> str:
        pred_report = self.result_data.iloc[idx]['report']
        return pred_report

    def __getitem__(self, idx: int) -> dict:
        id = self._get_id(idx)
        gt_report = self._get_gt_report(idx)
        pred_report = self._get_pred_report(idx)
        
        return {
            "id": id,
            "gt_report": gt_report,
            "pred_report": pred_report,
        }

    def __len__(self):
        return len(self.result_data)

# Example usage:
if __name__ == "__main__":
    csv_file_path = "data/REG_data.csv"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = REGDataset(csv_file_path, transform)
    sample = dataset[0]  # Fetch the first sample

    print(sample['image'].size())
    # print(sample['dicom_id'])
    # print(sample['subject_id'])
    # print(sample['study_id'])
    # print(sample['label'])
    # print(sample['date_diff'])
    # print(sample['sequence_index'])
    # print(sample['gt_report'])
    # print(sample['image_embedding'])

    # results = ResultDataset("results/result_pubmed-clip-vit-base-patch32_Phi-3.5-vision-instruct.csv")
    # result = results[20]

    # print(result['gt_label'])
    # print(result['pred_label'])
    # print(type(result['pred_label']))