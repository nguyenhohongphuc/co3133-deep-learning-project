"""Check diagnostic denominators, CSV alignment, confidence intervals and sampling."""
import csv
from pathlib import Path
import tempfile
import unittest

from scripts.a2_error_analysis import (class_adjusted_extreme, diverse, group_stat,
                                      load_predictions, top_confusions, wilson)
import numpy as np
import torch
from scripts.a2_gradcam import gradcam


class ErrorAnalysisTests(unittest.TestCase):
    def test_confusion_rates_use_true_class_support(self):
        matrix=np.array([[90,10],[1,1]])
        rows=top_confusions('model',matrix,['A','B'])
        self.assertEqual(rows[0]['count'],10)
        self.assertAlmostEqual(rows[0]['rate_within_true_class'],.1)
        self.assertAlmostEqual(rows[1]['rate_within_true_class'],.5)
        self.assertAlmostEqual(rows[0]['share_of_all_errors'],10/11)

    def test_empty_and_small_groups_have_valid_intervals(self):
        self.assertEqual(wilson(0,0),(None,None))
        empty=group_stat('model','empty',[])
        self.assertIsNone(empty['error_rate'])
        self.assertEqual(empty['support'],0)
        lo,hi=wilson(0,5)
        self.assertAlmostEqual(lo,0)
        self.assertGreater(hi,.4)
        row=group_stat('m','g',[dict(correct=False,confidence=.95),dict(correct=True,confidence=.7)])
        self.assertEqual(row['errors'],1);self.assertEqual(row['error_rate'],.5)

    def test_selection_is_deterministic_and_prefers_class_diversity(self):
        rows=[dict(path=str(i),true_class=label) for i,label in enumerate(['A','A','B','C'])]
        result=diverse(rows,lambda r:r['path'],3)
        self.assertEqual([r['true_class'] for r in result],['A','B','C'])
        self.assertEqual(result,diverse(list(reversed(rows)),lambda r:r['path'],3))

    def test_class_adjustment_respects_extreme_class_mix(self):
        rows=[dict(true_label=0,aspect_ratio=1,correct=True) for _ in range(9)]
        rows += [dict(true_label=0,aspect_ratio=1,correct=False)]
        rows += [dict(true_label=1,aspect_ratio=1,correct=False) for _ in range(10)]
        rows += [dict(true_label=0,aspect_ratio=3,correct=False) for _ in range(2)]
        result=class_adjusted_extreme(rows)
        self.assertEqual(result['observed_errors'],2)
        self.assertAlmostEqual(result['expected_errors_using_same_class_normal_rates'],.2)

    def test_prediction_rows_must_match_manifest_identity(self):
        expected=[dict(path='Test/A/1.jpg',label=0,md5='hash')]
        row=dict(index=0,path=expected[0]['path'],md5='wrong',true_label=0,pred_label=0,
                 true_class='A',pred_class='A',correct=1,width=12,height=10,confidence=.9,
                 true_class_probability=.9,confidence_margin=.8,aspect_ratio=1.2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.csv'
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            with self.assertRaisesRegex(ValueError,'alignment mismatch'):
                load_predictions(path,expected,['A','B'])

    def test_gradcam_targets_and_zero_maps_do_not_change_weights(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.net=torch.nn.Module()
                self.net.layer4=torch.nn.Sequential(torch.nn.Conv2d(3,1,1,bias=False),torch.nn.ReLU())
                self.head=torch.nn.Linear(1,2,bias=False)
                with torch.no_grad():
                    self.net.layer4[0].weight.fill_(1);self.head.weight.copy_(torch.tensor([[1.],[-1.]]))
            def forward(self,x):return self.head(self.net.layer4(x).mean(dim=(2,3)))
        model=Toy().eval();before={n:p.clone() for n,p in model.named_parameters()}
        image=torch.zeros(1,3,4,4);image[:,:,0,0]=1
        maps=gradcam(model,image,[0,1])
        self.assertEqual(maps[0]['map'].shape,(4,4))
        self.assertAlmostEqual(float(maps[0]['map'][0,0]),1)
        self.assertEqual(float(maps[1]['map'].max()),0)
        self.assertTrue(all(torch.equal(before[n],p) for n,p in model.named_parameters()))
        self.assertEqual(len(model.net.layer4[-1]._forward_hooks),0)


if __name__=='__main__':unittest.main()
