import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rq2'));sys.path.insert(0,str(ROOT/'rq4'))
from protocol import load_team_data,project_rows
from common.data import tokenize,text_for
from run_rq4 import transform

class ContributionTests(unittest.TestCase):
    def test_actual_dataset_and_visible_response_budget(self):
        original,info=load_team_data(ROOT/'data/processed/raid_balanced.csv')
        self.assertEqual(info['rows_by_split'],{'train':1280,'validation':160,'test':160})
        for before,after in zip(original,project_rows(original)):
            self.assertEqual(tokenize(after['generation']),tokenize(before['generation'])[:256])
            self.assertLessEqual(len(tokenize(text_for(after,'both'))),385)
            self.assertEqual(before['id'],after['id'])
            self.assertEqual(before['split'],after['split'])

    def test_shuffling_preserves_tokens_and_is_deterministic(self):
        text='Alpha beta!\n* One detail and another detail.'
        a=transform(text,'token_order_shuffled');b=transform(text,'token_order_shuffled')
        self.assertEqual(a,b)
        self.assertCountEqual(tokenize(text),tokenize(a))

    def test_formatting_preserves_visible_words_without_extra_content(self):
        text='Alpha!\n* Beta gamma 42.'
        self.assertEqual(transform(text,'formatting_removed'),'Alpha Beta gamma 42')

if __name__=='__main__':unittest.main()
