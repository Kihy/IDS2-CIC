from after_image.feature_extractor import *
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

        features, packet = feature_extractor.get_next_vector()
        while features == []:
            features, packet = feature_extractor.get_next_vector()
        feature_extractor.nstat.updateGetStats(*features)
        feature_extractor.dummy_nstat.updateGetStats(*features)


        # simulate adding adversarial crafted packets
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.46, 70)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.47, 60)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.48, 80)
        # nstat and dummy_nstat should not be the same
        assert feature_extractor.dummy_nstat.num_updated != feature_extractor.nstat.num_updated

        #roll back dummy_nstat
        feature_extractor.roll_back()
        # should be the same
        assert feature_extractor.dummy_nstat.num_updated == feature_extractor.nstat.num_updated

        #get next vector
        features, packet = feature_extractor.get_next_vector()
        while features == []:
            features, packet = feature_extractor.get_next_vector()

        true_stat=feature_extractor.nstat.updateGetStats(*features)
        assert feature_extractor.dummy_nstat.num_updated != feature_extractor.nstat.num_updated
        feature_extractor.dummy_nstat.updateGetStats(*features)

        # simulate adding adversarial crafted packets
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.46, 70)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.47, 60)
        after_craft=feature_extractor.dummy_nstat.updateGetStats(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', 'arp', '192.168.2.152', 'arp', 1540450873.48, 80)
        # nstat and dummy_nstat should not be the same
        assert feature_extractor.dummy_nstat.num_updated != feature_extractor.nstat.num_updated




if __name__ == '__main__':
    unittest.main()
