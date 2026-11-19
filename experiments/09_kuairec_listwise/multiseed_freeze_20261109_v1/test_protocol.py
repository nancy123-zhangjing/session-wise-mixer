import unittest
from protocol import counts, skipped

def plan():
    rows=[dict(kind='R',epoch=e,updates=693) for e in range(1,6)]
    for cycle in range(10):
        epoch=5+3*cycle
        rows.append(dict(kind='T',epoch=epoch,updates=40))
        rows.extend(dict(kind='R',epoch=e,updates=693) for e in range(epoch+1,epoch+4))
    return rows

class ScheduleTests(unittest.TestCase):
    def test_same_through_r20(self):
        p=plan()
        for phase in range(26):
            cursor=dict(phase=phase,update=0)
            self.assertEqual(counts(p,cursor,'H0'),counts(p,cursor,'H1'))
        self.assertEqual(counts(p,dict(phase=25,update=0),'H1'),dict(R=13860,T=200))

    def test_freeze_skips_exactly_last_five_t_blocks(self):
        p=plan()
        self.assertEqual([r['epoch'] for r in p if skipped(r,'H1')],[20,23,26,29,32])
        self.assertEqual(counts(p,dict(phase=45,update=0),'H1'),dict(R=24255,T=200))
        self.assertEqual(counts(p,dict(phase=45,update=0),'H0'),dict(R=24255,T=400))

    def test_resume_partial_r_and_forbid_frozen_t_update(self):
        p=plan()
        self.assertEqual(counts(p,dict(phase=26,update=100),'H1'),dict(R=13960,T=200))
        with self.assertRaises(ValueError): counts(p,dict(phase=25,update=1),'H1')
        self.assertEqual(counts(p,dict(phase=25,update=1),'H0'),dict(R=13860,T=201))

if __name__=='__main__': unittest.main()
