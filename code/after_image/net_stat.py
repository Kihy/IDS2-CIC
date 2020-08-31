import numpy as np
## Prep AfterImage cython package
import os
import subprocess
import pyximport
pyximport.install()
import after_image.after_image as af
from pprint import pformat
#import AfterImage_NDSS as af

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas = np.nan, HostLimit=255,HostSimplexLimit=1000):
        #Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas

        # number of pkts updated
        self.num_updated=0

        #cutoffweight for cleaning
        self.cutoffWeight=1e-6

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*10

        #HTs
        self.HT_jit = af.IncStatDB("HT_jit",limit=self.HostLimit*self.HostLimit)#H-H Jitter Stats
        self.HT_MI = af.IncStatDB("HT_MI",limit=self.MAC_HostLimit)#MAC-IP relationships
        self.HT_H = af.IncStatDB("HT_H",limit=self.HostLimit) #Source Host BW Stats
        self.HT_Hp = af.IncStatDB("HT_Hp",limit=self.SessionLimit)#Source Host BW Stats

    def getHT(self):
        return {"HT_jit":self.HT_jit,
        "HT_MI" :self.HT_MI,
        "HT_H":self.HT_H,
        "HT_Hp":self.HT_Hp,
        "num_updated":self.num_updated}

    def setHT(self, HT_dict):
        self.HT_jit=HT_dict["HT_jit"]
        self.HT_MI=HT_dict["HT_MI"]
        self.HT_H=HT_dict["HT_H"]
        self.HT_Hp=HT_dict["HT_Hp"]
        self.num_updated=HT_dict["num_updated"]

    def __repr__(self):
        return "HT_jit"+pformat(self.HT_jit, indent=2)+"\nHT_MI"+pformat(self.HT_MI, indent=2)+"\nHT_H"+pformat(self.HT_H, indent=2)+"\nHT_Hp"+pformat(self.HT_Hp, indent=2)

    def findDirection(self,IPtype,srcIP,dstIP,eth_src,eth_dst): #cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype==0: #is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype==1: #is IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else: #no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        #MAC.IP: Stats on src MAC-IP relationships
        MIstat =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats("{}_{}".format(srcMAC,srcIP), timestamp, datagramSize, i)

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat =  np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,i)
        # Host-Host Jitter:
        HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats("{}_{}".format(srcIP,dstIP), timestamp, 0,i,isTypeDiff=True)

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, i)
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats("{}_{}".format(srcIP, srcProtocol), "{}_{}".format(dstIP , dstProtocol), timestamp, datagramSize, i)

        self.num_updated+=1

        #clean our records every 100 updates
        if self.num_updated%100==0:
            self.HT_MI.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_H.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_jit.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_Hp.cleanOutOldRecords(self.cutoffWeight, timestamp)

        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat))  # concatenation of stats into one stat vector

    def get_stats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, t1, frame_len=None, Lambda=1):
        """get stats of a packet, framelen not needed"""

        MIstat =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            ID=srcMAC+srcIP
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.get_1D_Stats(ID,self.Lambdas[i])

        HHstat =  np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.get_1D_Stats(srcIP,self.Lambdas[i])+self.HT_H.get_2D_stats(srcIP, dstIP,self.Lambdas[i],t1,level=2)


        # amount of traffic arriving out of time order
        HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.get_1D_Stats(srcIP+dstIP,self.Lambdas[i])

        HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.get_1D_Stats(srcMAC,self.Lambdas[i])+self.HT_Hp.get_2D_stats(srcMAC, dstMAC,self.Lambdas[i],t1,level=2)
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.get_1D_Stats(srcIP + srcProtocol, self.Lambdas[i])+self.HT_Hp.get_2D_stats(srcIP + srcProtocol, dstIP + dstProtocol,self.Lambdas[i],t1,level=2)

        return np.concatenate((MIstat,HHstat, HHstat_jit,HpHpstat))


    def getNetStatHeaders(self):

        MIstat_headers = self.HT_MI.get_headers()
        HHstat_headers = self.HT_H.get_headers(True)
        HHjitstat_headers = self.HT_jit.get_headers()
        HpHpstat_headers = self.HT_Hp.get_headers(True)


        return MIstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers
