import unittest
from tensorfold.families.glm_moe_dsa.indexer_plan import visible_token_bound,score_width


class VisibleBound(unittest.TestCase):
    def test_mixed_requests_bound_logical_positions_and_crosses_topk_boundary(self):
        self.assertEqual(visible_token_bound([0,1023,17,2047],804000),2048)
        self.assertEqual(score_width(804000,2048),2048)
        self.assertEqual(score_width(804000,2049),4096)
        self.assertEqual(score_width(804000,360000),524288)
        self.assertEqual(score_width(360000,360000),360000)
        self.assertEqual(score_width(804000),804000)
        self.assertEqual(score_width(17,17),2048)

    def test_graph_bucket_never_truncates_a_valid_position(self):
        for cap in (17,2048,2049,3072,360000,804000,1048576):
            for p in {0,cap//2,cap-1}:
                bound=visible_token_bound([p,0],cap)
                self.assertGreaterEqual(score_width(cap,bound),p+1)
                self.assertLessEqual(score_width(cap,bound),max(2048,cap))
        for positions in ([],[True],[-1],[17],[0.0],None):
            with self.assertRaises(ValueError):visible_token_bound(positions,17)
        for bound in (0,-1,True,18,1.0):
            with self.assertRaises(ValueError):score_width(17,bound)


if __name__=='__main__':unittest.main()
