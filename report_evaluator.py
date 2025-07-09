# BLEU, ROUGE, Chexbert-F1/Precision/Recall
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from reg_dataset import ResultDataset
import argparse
import csv
import os

class MetricsCalculator:
    def __init__(self):
        self.smoothie = SmoothingFunction().method1
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True
        )

    def calc_bleu(self, reference: str, prediction: str):
        bleu1 = sentence_bleu([reference], prediction, weights=(1, 0, 0, 0), smoothing_function=self.smoothie)
        bleu2 = sentence_bleu([reference], prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothie)
        bleu3 = sentence_bleu([reference], prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothie)
        bleu4 = sentence_bleu([reference], prediction, smoothing_function=self.smoothie)
        return bleu1, bleu2, bleu3, bleu4

    def calc_rouge(self, reference: str, prediction: str):
        scores = self.rouge_scorer.score(reference, prediction)
        return (
            scores['rouge1'].fmeasure, 
            scores['rouge2'].fmeasure, 
            scores['rougeL'].fmeasure, 
            scores['rougeLsum'].fmeasure
        )

    def calc_f1(self, reference: list, prediction: list):
        return f1_score(reference, prediction, average="macro")

    def calc_precision(self, reference: list, prediction: list):
        return precision_score(reference, prediction, average="macro")

    def calc_recall(self, reference: list, prediction: list):
        return recall_score(reference, prediction, average="macro")

    def calc_accuracy(self, reference: list, prediction: list):
        return accuracy_score(reference, prediction)

    def calc_chexbert_metrics(self, reference: list, prediction: list):
        return (
            self.calc_f1(reference, prediction),
            self.calc_precision(reference, prediction),
            self.calc_recall(reference, prediction),
            self.calc_accuracy(reference, prediction)
        )

class ScoreEvaluator:
    def __init__(self, dataset, csv_file_path):
        self.dataset = dataset
        self.csv_file_path = csv_file_path
        self.calculator = MetricsCalculator()

    def get_scores(self):
        results = []
        for data in self.dataset:
            id, gt_report, pred_report = data.values()

            bleu1, bleu2, bleu3, bleu4 = self.calculator.calc_bleu(gt_report, pred_report)
            rouge1, rouge2, rougeL, rougeLsum = self.calculator.calc_rouge(gt_report, pred_report)

            results.append([bleu1, bleu2, bleu3, bleu4, rouge1, rouge2, rougeL, rougeLsum])

        new_file_name = os.path.splitext(os.path.basename(self.csv_file_path)[7:])[0]
        self.save_results(results, save_path=f"scores/score_{new_file_name}_details.csv")
        print(f"[Completed] Detailed Scores Saved at \"scores/score_{new_file_name}_details.csv\"")

        result_avg = np.mean(results, axis=0)
        print(result_avg)
        self.save_results([result_avg], save_path=f"scores/score_{new_file_name}.csv")
        print(f"[Completed] Average Score Saved at \"scores/score_{new_file_name}.csv\"")

    def _select_report(self, gt_findings: str, gt_impression: str, pred_findings: str, pred_impression: str):
        if gt_findings and gt_impression:
            return f"{gt_findings}\n\n{gt_impression}", f"{pred_findings}\n\n{pred_impression}"
        if gt_findings:
            return gt_findings, pred_findings
        if gt_impression:
            return gt_impression, pred_impression
        return "", ""

    def save_results(self, results, save_path="results.csv"):
        header = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        
        with open(save_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for result in results:
                writer.writerow(result)

def main(args):
    dataset = ResultDataset(csv_file_path=args.csv_file_path)

    evaluator = ScoreEvaluator(dataset, args.csv_file_path)
    evaluator.get_scores()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate medical reports using LLM")
    parser.add_argument("-f", "--csv_file_path", type=str, default="results/result_t_report_None_Phi-3.5-vision-instruct.csv", help="Path to the result CSV file")

    args = parser.parse_args()
    main(args)