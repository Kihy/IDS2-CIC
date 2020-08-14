from after_image.FeatureExtractor import *
from tqdm import tqdm
import unittest
import numpy as np



class TestStringMethods(unittest.TestCase):

    @unittest.skip("too time consuming, dont run often")
    def test_get_stats(self):
        pcap_path="../../TrafficManipulator/example/test.pcap"
        feature_extractor=FE(pcap_path)

        for i in tqdm(range(feature_extractor.num_pkt)):
            features = feature_extractor.get_next_vector()

            true_stats=feature_extractor.nstat.updateGetStats(*features)

            test_stats=feature_extractor.nstat.get_stats(*features)

            np.testing.assert_almost_equal(true_stats, test_stats)

    def test_roll_back(self):
        pcap_path="../../TrafficManipulator/example/test.pcap"
        feature_extractor=FE(pcap_path)

        #update both nstat with features
        features = feature_extractor.get_next_vector()
        feature_extractor.nstat.updateGetStats(*features)
        feature_extractor.dummy_nstat.updateGetStats(*features)
        print(features)

        # simulate adding adversarial crafted packets
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.46, 70)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.47, 60)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.48, 80)
        # nstat and dummy_nstat should not be the same
        true_stats=feature_extractor.nstat.get_stats(*features)
        assert not np.array_equal(after_craft , true_stats)

        #roll back dummy_nstat
        feature_extractor.roll_back()
        # should be the same
        test_stats=feature_extractor.dummy_nstat.get_stats(*features)
        true_stats=feature_extractor.nstat.get_stats(*features)
        np.testing.assert_almost_equal(test_stats, true_stats)

        #get next vector
        features = feature_extractor.get_next_vector()
        true_stat=feature_extractor.nstat.updateGetStats(*features)
        dummy_stats=feature_extractor.dummy_nstat.get_stats(*features)
        assert not np.array_equal(dummy_stats , true_stats)
        feature_extractor.dummy_nstat.updateGetStats(*features)

        # simulate adding adversarial crafted packets
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.46, 70)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.47, 60)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.48, 80)
        # nstat and dummy_nstat should not be the same
        true_stats=feature_extractor.nstat.get_stats(*features)
        assert not np.array_equal(after_craft , true_stats)




if __name__ == '__main__':
    unittest.main()
