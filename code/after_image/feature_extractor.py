#Check if cython code has been compiled
import os
import subprocess
import copy
#Import dependencies
import after_image.net_stat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess
import pickle


#Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
class FE:
    def __init__(self,file_path,limit=np.inf, nstat=None, dummy_nstat=None):
        self.path = file_path
        self.limit = limit
        self.parse_type = None #unknown
        self.curPacketIndx = 0
        self.tsvin = None #used for parsing TSV file
        self.scapyin = None #used for parsing pcap with scapy

        ### Prep pcap ##
        print("Reading PCAP file via Scapy...")

        # self.scapyin = rdpcap(self.path, count=self.max_pkt)
        self.scapyin = PcapReader(self.path)

        ### Prep Feature extractor (AfterImage) ###
        self.maxHost = 100000000000
        self.maxSess = 100000000000

        if nstat is not None:
            self.nstat=nstat
        else:
            self.nstat = ns.netStat(np.nan, self.maxHost, self.maxSess)

        if dummy_nstat is not None:
            self.dummy_nstat=dummy_nstat
        else:
            self.dummy_nstat=ns.netStat(np.nan, self.maxHost, self.maxSess)

    def get_nstat(self):
        return self.nstat, self.dummy_nstat

    def get_next_vector(self, craft=False):
        pkt_tuple = self.scapyin.read_packet()
        packet, pkt_metadata=pkt_tuple[0],pkt_tuple[1]

        #only process IP packets,
        if not (packet.haslayer(IP) or packet.haslayer(IPv6)):
            return [], packet

        timestamp = packet.time
        framelen = packet.len
        if packet.haslayer(IP):  # IPv4
            srcIP = packet[IP].src
            dstIP = packet[IP].dst
            IPtype = 0
        elif packet.haslayer(IPv6):  # ipv6
            srcIP = packet[IPv6].src
            dstIP = packet[IPv6].dst
            IPtype = 1
        else:
            srcIP = ''
            dstIP = ''

        if packet.haslayer(TCP):
            srcproto = str(packet[TCP].sport)
            dstproto = str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            srcproto = str(packet[UDP].sport)
            dstproto = str(packet[UDP].dport)
        else:
            srcproto = ''
            dstproto = ''

        srcMAC = packet.src
        dstMAC = packet.dst
        if srcproto == '':  # it's a L2/L1 level protocol
            if packet.haslayer(ARP):  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = packet[ARP].psrc  # src IP (ARP)
                dstIP = packet[ARP].pdst  # dst IP (ARP)
                IPtype = 0
            elif packet.haslayer(ICMP):  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        return [IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, float(timestamp), int(framelen)], packet

    def save_nstat_state(self):
        f=open('tmp_nstat.txt', 'wb')
        pickle.dump( obj=self.nstat,file=f)

    def roll_back(self):
        """Roll back dummy to nstat"""

        self.dummy_nstat = pickle.load(open('tmp_nstat.txt', 'rb'))


    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
