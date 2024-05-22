import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    met_filters,
    HLTSFs,
    eleSFs,
    muSFs,
    puwei,
    btagSFs,
    JME_shifts,
    Roccor_shifts,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
    flatten,
    update,
    uproot_writeable,
    dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer
from BTVNanoCommissioning.utils.selection import (
    jet_id, mu_idiso, 
    ele_cuttightid,
    btag_wp_dict,
    btag_wp,
)
from BTVNanoCommissioning.helpers.ttcc2L2Nu_helper import (
    sel_HLT,
    to_bitwise_trigger,
)

class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=True,
        noHist=False,
        chunksize=75000,
        isTTbar=False
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._campaign)
        self.isTTbar = isTTbar
        

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        print("**************************************")
        print("This is ", self._campaign, "ttcc dilepton channel producer")
        print("**************************************")
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        events = missing_branch(events)
        shifts = []
        if "JME" in self.SF_map.keys():
            syst_JERC = self.isSyst
            if self.isSyst == "JERC_split":
                syst_JERC = "split"
            shifts = JME_shifts(
                shifts, self.SF_map, events, self._campaign, isRealData, syst_JERC
            )
        else:
            if "Run3" not in self._campaign:
                shifts = [
                    ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
                ]
            else:
                shifts = [
                    ({"Jet": events.Jet, "MET": events.PuppiMET, "Muon": events.Muon,}, None,)
                ]
        if "roccor" in self.SF_map.keys():
            shifts = Roccor_shifts(shifts, self.SF_map, events, isRealData, False)
        else:
            shifts[0][0]["Muon"] = events.Muon

        for collections, name in shifts:
            print('n ', name)
            print('n ', name)
            print('n ', name)
            print('col', collections)
            print('col', collections)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        ## Create histograms
        _hist_event_dict = (
            {"": None} if self.noHist else histogrammer(events, "ttcc2L2Nu")
        )
        if _hist_event_dict == None:
            _hist_event_dict[""]
        output = {
            "sumw": processor.defaultdict_accumulator(float),
            **_hist_event_dict,
        }
        if isRealData:
            output["sumw"] = len(events)
        else:
            output["sumw"] = ak.sum(events.genWeight)

        #################
        #    Selections    #
        #################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        if isRealData:
            MET_filters = met_filters[self._campaign]["data"]
        else: MET_filters = met_filters[self._campaign]["mc"]

        filters = [f for f in MET_filters]
        print('filters: ', filters)
        print('filters: ', filters)

        checkFlag = ak.Array([hasattr(events.Flag, _filt) for _filt in filters])
        if ak.all(checkFlag == False):
            raise ValueError("Flag paths:", filters, " are all invalid in", dataset)
        elif ak.any(checkFlag == False):
            print(np.array(filters)[~checkFlag], " not exist in", dataset)

        filt_arrs = [
            events.Flag[_filt] for _filt in filters if hasattr(events.Flag, _filt)
        ]
        for _filt in filters:
            print(_filt, ' ', events.Flag[_filt])

        req_flag = np.ones_like(len(events), dtype="bool")
        for f in filt_arrs:
            req_flag = req_flag & f

        print(req_flag)
        print(req_flag)
        ###########
        #    HLT   #
        ###########

        ttdilep_HLT_chns = sel_HLT(dataset, self._campaign)
        triggers = [t[0] for t in ttdilep_HLT_chns]

        checkHLT = ak.Array([hasattr(events.HLT, _trig) for _trig in triggers])
        if ak.all(checkHLT == False):
            raise ValueError("HLT paths:", triggers, " are all invalid in", dataset)
        elif ak.any(checkHLT == False):
            print(np.array(triggers)[~checkHLT], " not exist in", dataset)
        trig_arrs = [
            events.HLT[_trig] for _trig in triggers if hasattr(events.HLT, _trig)
        ]
        req_trig = np.zeros(len(events), dtype="bool")
        for t in trig_arrs:
            req_trig = req_trig | t

        # pass_trig.shape = (nevents, len(triggers))
        pass_trig = np.array([
            events.HLT[_trig] for _trig in triggers if hasattr(events.HLT, _trig)
        ]).T

        pass_trig_chn = {
            chn: np.zeros(len(events), dtype="bool")
            for chn in ["ee", "mm", "em"]
        }
        for i, (trig, chn) in enumerate(ttdilep_HLT_chns):  # loop over ee, mm, em chanenl
            pass_trig_chn[chn] |= pass_trig[:, i]
        
        output["ttbar_trigWord"] = to_bitwise_trigger(
            pass_trig, ak.ArrayBuilder()
        ).snapshot()[:, 0]
        
        #################
        #    Electronss    #
        #################

        electrons = events.Electron[ele_cuttightid(events, self._campaign)]
        nelectrons = ak.num(electrons)
        
        #############
        #    Muons    #
        #############

        muons = events.Muon[mu_idiso(events, self._campaign)]
        nmuons = ak.num(muons)
        
        ###############
        #    Leptons   #
        ###############

        leptons = ak.concatenate([electrons, muons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
        #nleptons = ak.num(leptons)
        nleptons = nmuons+nelectrons
        pair_lep = ak.combinations(leptons, 2) 
        
        ####################
        #  dilep channels  #
        ####################
        # criteria for each channel (ee, mm, em)
        chn_criteria = {
            'ee': (nelectrons==2) & (nmuons==0),
            'mm': (nelectrons==0) & (nmuons==2),
            'em': (nelectrons==1) & (nmuons==1),
        }
        # criteria + HLT
        for chn in chn_criteria.keys():
            chn_criteria[chn] = chn_criteria[chn] & pass_trig_chn[chn]
        
        zeros = ak.zeros_like(events.run, dtype=int)
        channel = zeros
        # ee: 11*11, mm: 13*13, em: 11*13, oppositely-charged < 0
        channel = ak.where(
            chn_criteria["ee"] & (channel == 0),
            ak.fill_none(11 * 11 * pair_lep["0"].charge * pair_lep["1"].charge, 0),
            channel,
        )
        channel = ak.where(
            chn_criteria["mm"] & (channel == 0),
            ak.fill_none(13 * 13 * pair_lep["0"].charge * pair_lep["1"].charge, 0),
            channel,
        )
        channel = ak.where(
            chn_criteria["em"] & (channel == 0),
            ak.fill_none(11 * 13 * pair_lep["0"].charge * pair_lep["1"].charge, 0),
            channel,
        )

        ch = [] # make flat channel
        for chn in channel:
            if chn==0: ch.append(chn)
            else: ch.append(chn[0])
        channel = np.array(ch)
        
        req_lep = (nleptons == 2) & (channel < 0)

        # invariant mass of two leptons
        pad_leptons = ak.pad_none(leptons, 2, axis=1)
        lep1 = pad_leptons[:,0]
        lep2 = pad_leptons[:,1]
        mll = (lep1+lep2).mass
        mll_mask = mll > 20
        req_mll = ak.fill_none(mll_mask, False)

        mz_mask = abs(mll-91.2) > 15
        req_mz = np.where(
            ~chn_criteria["em"],
            ak.fill_none(mz_mask, False),
            True
        )

        #########
        #  MET  #
        #########

        met_mask = events.MET.pt > 30
        req_met = np.where(
            ~chn_criteria["em"],
            ak.fill_none(met_mask, False),
            True,
        )
        
        ##########
        #  jets  #
        ##########
        
        jets = events.Jet[
            jet_id(events, self._campaign)
            & ak.all(events.Jet.metric_table(muons) > 0.4, axis=2)
            & ak.all(events.Jet.metric_table(electrons) > 0.4, axis=2),
        ]
        njets = ak.num(jets)
        req_jet = njets >= 4

        # b tagged jets (M)
        bjets = jets[
            btag_wp(jets, self._campaign, "DeepFlav", "b", "M")
        ]
        nbjets = ak.num(bjets)
        req_bjet = nbjets >= 2

        ####################
        #  Gen-level info  #
        ####################
        # tt test
        if self.isTTbar:
            print('This is ttbar sample')
        else: print('This is not ttbar sample')

        if self.isTTbar:
            ############
            #  Genlep  #
            ############
            # Genlep: same with nominal BTA workflow
            _fix = lambda x: ak.fill_none(x, 0)
            is_lep = (
                lambda p: (abs(_fix(p.pdgId)) == 11)
                | (abs(_fix(p.pdgId)) == 13)
                | (abs(_fix(p.pdgId)) == 15)
            )
            is_WZ = lambda p: (abs(_fix(p.pdgId)) == 23) | (abs(_fix(p.pdgId)) == 24)
            is_heavy_hadron = lambda p, pid: (abs(_fix(p.pdgId)) // 100 == pid) | (
                abs(_fix(p.pdgId)) // 1000 == pid
            )
            
            genlep_cut = (
                is_lep(events.GenPart)
                & (events.GenPart.hasFlags("isLastCopy"))
                & (events.GenPart.pt > 20)
                & (abs(events.GenPart.eta) < 2.4)
            )
            genlep = events.GenPart[genlep_cut]

            # trace parents up to 4 generations (from BTA code)
            genlep_pa1G = genlep.parent
            genlep_pa2G = genlep.parent.parent
            genlep_pa3G = genlep.parent.parent.parent
            genlep_pa4G = genlep.parent.parent.parent.parent
            istau = abs(genlep_pa1G.pdgId) == 15
            isWZ = is_WZ(genlep_pa1G) | is_WZ(genlep_pa2G)
            isD = is_heavy_hadron(genlep_pa1G, 4) | is_heavy_hadron(genlep_pa2G, 4)
            isB = (
                is_heavy_hadron(genlep_pa1G, 5)
                | is_heavy_hadron(genlep_pa2G, 5)
                | is_heavy_hadron(genlep_pa3G, 5)
                | is_heavy_hadron(genlep_pa4G, 5)
            )
            
            Genlep = ak.zip(
                {
                    "pT": genlep.pt,
                    "eta": genlep.eta,
                    "phi": genlep.phi,
                    "pdgID": genlep.pdgId,
                    "status": genlep.status,
                    "mother": ak.fill_none(ak.where(isB | isD, 5 * isB + 4 * isD, 10 * istau + 100 * isWZ), 0),
                }
            )
            # gen-level jet cleaning aginst prompt leptons from WZ or tau
            genlep_prompt = genlep[(Genlep.mother != 0) & (Genlep.mother % 10 == 0)]
            
            ############
            #  GenJet  #
            ############
            Genjets = events.GenJet
            Genjets = Genjets[
                (Genjets.pt > 20)
                & (abs(Genjets.eta) < 2.4)
                & (ak.all(Genjets.metric_table(genlep_prompt) > 0.4, axis=-1))
            ]
            nGenjets = ak.num(Genjets)
            req_Genjet = nGenjets > 1
            
            addbjet_cut = (
                ((Genjets.nBHadFromT+Genjets.nBHadFromTbar+Genjets.nBHadFromW) == 0)
                & (Genjets.nBHadOther > 0)
            )
            addcjet_cut = (
                ((Genjets.nCHadFromW) == 0)
                & (Genjets.nCHadOther > 0)
            )
            
            Genaddbjets = Genjets[ak.fill_none(addbjet_cut, False)]
            Genaddcjets = Genjets[ak.fill_none(addcjet_cut, False)]
            
            naddbjets = ak.num(Genaddbjets)
            naddcjets = ak.num(Genaddcjets)
            
            ####################
            #  ttbar category  #
            ####################

            ttbar_criteria = {
                'ttbb': naddbjets >= 2,
                'ttbj': naddbjets == 1,
                'ttcc': (naddbjets == 0) & (naddcjets >= 2),
                'ttcj': (naddbjets == 0) & (naddcjets == 1),
                'ttother': (naddbjets+naddcjets == 0) & (nGenjets >=2),
            }
            
            zeros = ak.zeros_like(events.run, dtype=bool)
            isttbb, isttbj = zeros, zeros
            isttcc, isttcj = zeros, zeros
            isttother = zeros
            
            isttbb = ak.to_numpy(
                ak.where(
                    ttbar_criteria['ttbb'],
                    True,
                    isttbb,
                )
            )
            isttbj = ak.to_numpy(
                ak.where(
                    ttbar_criteria['ttbj'],
                    True, 
                    isttbj,
                )
            )
            isttcc = ak.to_numpy(
                ak.where(
                    ttbar_criteria['ttcc'],
                    True,
                    isttcc,
                )
            )
            isttcj = ak.to_numpy(
                ak.where(
                    ttbar_criteria['ttcj'],
                    True,
                    isttcj,
                )
            )
            isttother = ak.to_numpy(
                ak.where(
                    ttbar_criteria['ttother'],
                    True,
                    isttother,
                )
            )

        #####################
        #  Event selections  #
        #####################
        req_event = req_trig & req_lumi & req_lep & req_jet & req_bjet & req_mll & req_mz & req_met & req_flag
        if self.isTTbar: req_event = req_event & req_Genjet
        
        req_event = ak.fill_none(req_event, False)

        if len(events[req_event]) == 0:
            return {dataset: output}
        
        ####################
        # Selected objects #
        ####################
        pad_electrons = ak.pad_none(electrons, 2, axis=1)[req_event]
        pad_muons = ak.pad_none(muons, 2, axis=1)[req_event]
        #pad_jets = ak.pad_none(jets, 6, axis=1)[req_event]
        pad_jets = ak.pad_none(jets, 4, axis=1)[req_event]
        pad_jets = pad_jets[:, :4]
        pad_lep1 = lep1[req_event]
        pad_lep2 = lep2[req_event]

        sel_muons = muons[req_event]
        sel_electrons = electrons[req_event]
        sel_leptons = leptons[req_event]
        #sel_jets = jets[req_event][:,:4]
        sel_jets = jets[req_event]

        ####################
        # Weight & Geninfo #
        ####################
        
        weights = Weights(len(events[req_event]), storeIndividual=True)
        if not isRealData:
            weights.add("genweight", events[req_event].genWeight)

            par_flav = (pad_jets.partonFlavour == 0) & (pad_jets.hadronFlavour == 0)
            genflavor = pad_jets.hadronFlavour + 1 * par_flav
            if len(self.SF_map.keys()) > 0:
                syst_wei = True if self.isSyst != False else False
                if "PU" in self.SF_map.keys():
                    puwei(
                        events[req_event].Pileup.nTrueInt,
                        self.SF_map,
                        weights,
                        syst_wei,
                    )
                if "HLT" in self.SF_map.keys():
                    HLTSFs(pad_lep1, pad_lep2, channel[req_event], self.SF_map, weights)
                if "MUO" in self.SF_map.keys():
                    muSFs(pad_muons, self.SF_map, weights, syst_wei, False)
                if "EGM" in self.SF_map.keys():
                    eleSFs(pad_electrons, self.SF_map, weights, syst_wei, False)
                #if "BTV" in self.SF_map.keys():
                if "btag" in self.SF_map.keys():
                    btagSFs(pad_jets, self.SF_map, weights, "DeepJetC", syst_wei)
                    btagSFs(pad_jets, self.SF_map, weights, "DeepJetB", syst_wei)
        else:
            genflavor = ak.zeros_like(pad_jets.pt, dtype=int)

        # Systematics information
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        exclude_btv = [
            "DeepJetB",
            "DeepJetB",
        ]  # exclude b-tag SFs for btag inputs

        for ind_wei in weights.weightStatistics.keys():
            np.set_printoptions(linewidth=np.inf, threshold=100)

        #######################
        #  Create root files  #
        #######################
        if self.isArray:
            # Keep the structure of events and pruned the object size
            pruned_ev = {'Jet': sel_jets, 'Muon': sel_muons, 'Electron': sel_electrons, 'Lepton': sel_leptons}
            pruned_ev['Channel'] = channel[req_event]
            pruned_ev['trig_bit'] = ak.to_numpy(output["ttbar_trigWord"][req_event])
            pruned_ev['nbJet'] = ak.to_numpy(nbjets[req_event])
            pruned_ev['mll'] = ak.to_numpy(mll[req_event])

            # Create a list of variables want to store. For objects from the PFNano file, specify as {object}_{variable}, wildcard option only accepted at the end of the string
            out_branch = ["events", "run", "luminosityBlock", "Channel", "trig_bit", "nbJet"]
            if not isRealData:
                pruned_ev["weight"] = weights.weight()
                out_branch = np.append(out_branch, "weight")
                for ind_wei in weights.weightStatistics.keys():
                    pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
                        include=[ind_wei]
                    )   
                    out_branch = np.append(out_branch, f"{ind_wei}_weight")

            for kin in ["pt", "eta", "phi", "mass", "dz", "dxy"]:
                for obj in ["Jet", "Electron", "Muon", "Lepton"]:
                #for obj in ["Jet", "Muon"]:
                    if ((obj == "Jet") or (obj == "Lepton")) and "d" in kin:
                        continue
                    out_branch = np.append(out_branch, [f"{obj}_{kin}"])

            out_branch = np.append(
                out_branch,
                [
                    "mll",
                    "Jet_btag*",
                    "Jet_DeepJet*",
                    "Jet_jetId",
                    "Jet_hadronFlavour",
                    "Muon_pfRelIso04_all",
                    "Muon_tightId",
                ],
            )
            if self.isTTbar:
                pruned_ev.update({"addbJet": Genaddbjets[req_event]})
                pruned_ev.update({"addcJet": Genaddcjets[req_event]})
                pruned_ev.update({"Genjets": Genjets[req_event]})

                pruned_ev['isttbb'] = isttbb[req_event]
                pruned_ev['isttbj'] = isttbj[req_event]
                pruned_ev['isttcc'] = isttcc[req_event]
                pruned_ev['isttcj'] = isttcj[req_event]
                pruned_ev['isttother'] = isttother[req_event]

                out_branch = np.append(
                    out_branch,
                    ['isttbb', 'isttbj', 'isttcc', 'isttcj', 'isttother',
                    'addbJet_pt', 'addbJet_eta', 'addbJet_phi', 'addbJet_mass', 
                    'addcJet_pt', 'addcJet_eta', 'addcJet_phi', 'addcJet_mass',]
                )

            # write to root files
            os.system(f"mkdir -p {self.name}/{dataset}")
            with uproot.recreate(
                f"{self.name}/{dataset}/f{events.metadata['filename'].split('_')[-1].replace('.root','')}_{systematics[0]}_{int(events.metadata['entrystop']/self.chunksize)}.root"
            ) as fout:
                fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
