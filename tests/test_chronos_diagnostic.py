import unittest
import numpy as np
import pandas as pd
from experiments.chronos_october_diagnostic import build_cases,COVS


class DiagnosticTests(unittest.TestCase):
    def test_controls_change_only_the_intended_input_and_mask_future(self):
        d=pd.DataFrame({'date':pd.date_range('2020-01-31',periods=83,freq='ME'),
                        'y':np.ones(83)*10,**{c:np.ones(83) for c in COVS}})
        d.loc[d.date==pd.Timestamp('2026-09-30'),'y']=129.81
        before=d.copy(deep=True);cases,normal=build_cases(d);by={r['name']:r for r in cases}
        pd.testing.assert_frame_equal(d,before)
        self.assertEqual(normal,10.)
        for r in cases:
            x=r['context'];self.assertEqual(x.date.max(),pd.Timestamp('2026-10-31'))
            self.assertTrue(x.iloc[-1][['y',*COVS]].isna().all())
        self.assertEqual(by['full']['context'].y.iloc[-2],129.81)
        self.assertEqual(by['normal_september_inflow']['context'].y.iloc[-2],10.)
        pd.testing.assert_frame_equal(by['normal_september_inflow']['context'][list(COVS)],by['full']['context'][list(COVS)])
        self.assertNotIn('rain_mm',by['without_rain_mm']['covariates'])
        self.assertEqual(by['inflow_only']['covariates'],())


if __name__=='__main__':unittest.main()
