import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
	load_lumi,
	load_SF,
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
from BTVNanoCommissioning.utils.selection import jet_id, mu_idiso, ele_cuttightid

from BTVNanoCommissioning.helpers.ttdilep_helper import (
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
		isArray=False,
		noHist=False,
		chunksize=75000,
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

	@property
	def accumulator(self):
		return self._accumulator

	## Apply corrections on momentum/mass on MET, Jet, Muon
	def process(self, events):
		isRealData = not hasattr(events, "genWeight")
		dataset = events.metadata["dataset"]
		events = missing_branch(events)
		shifts = []
		if "JME" in self.SF_map.keys():
			syst_JERC = True if self.isSyst != None else False
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

		return processor.accumulate(
			self.process_shift(update(events, collections), name)
			for collections, name in shifts
		)

	def process_shift(self, events, shift_name):
		dataset = events.metadata["dataset"]
		isRealData = not hasattr(events, "genWeight")
		## Create histograms
		_hist_event_dict = (
			{"": None} if self.noHist else histogrammer(events, "ttdilep_sf")
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
		#	Selections	#
		#################
		## Lumimask
		req_lumi = np.ones(len(events), dtype="bool")
		if isRealData:
			req_lumi = self.lumiMask(events.run, events.luminosityBlock)
		# only dump for nominal case
		if shift_name is None:
			output = dump_lumi(events[req_lumi], output)

		###########
		#	HLT   #
		###########

		ttdilep_HLT_chns = sel_HLT(self._campaign)
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
		#	Electronss	#
		#################

		electrons = events.Electron
		eleEtaGap  = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
		elePassDXY = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dxy)<0.05) | (abs(events.Electron.eta)>1.479) & (abs(events.Electron.dxy) < 0.1)
		elePassDZ = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)
		electrons = electrons[
			(electrons.pt > 20)
			& (abs(events.Electron.eta) < 2.1)
			& (events.Electron.cutBased >= 4) # pass cut-based tight ID
			& eleEtaGap
			& elePassDXY
			& elePassDZ
		]
		nelectrons = ak.num(electrons)
		
		#############
		#	Muons	#
		#############

		muons = events.Muon
		muons = muons[
			(muons.pt > 20)
			& (abs(muons.eta) < 2.4)
			& muons.tightId  # pass cut-based tight ID
			& (muons.pfRelIso04_all < 0.15)  # muon isolation cut
		]
		nmuons = ak.num(muons)
		
		###############
		#	Leptons   #
		###############

		leptons = ak.concatenate([electrons, muons], axis=1)
		nleptons = ak.num(leptons)
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
		# make flat channel
		ch = []
		for chn in channel:
			if chn==0: ch.append(chn)
			else: ch.append(chn[0])
		channel = np.array(ch)
		
		req_lep = (nleptons == 2) & (channel < 0)

		##########
		#  jets  #
		##########
		
		jets = events.Jet[ 
			(events.Jet.pt > 30) 
			& (abs(events.Jet.eta) < 2.4 )
			& ( events.Jet.jetId > 4 ) 
			& ak.all(events.Jet.metric_table(muons) > 0.4, axis=-1) 
			& ak.all(events.Jet.metric_table(electrons) > 0.4, axis=-1) 
		]
		njets = ak.num(jets)

		req_jet = njets >= 4

		req_event = req_trig & req_lumi & req_lep & req_jet
		req_event = ak.fill_none(req_event, False)

		if len(events[req_event]) == 0:
			return {dataset: output}
		
		####################
		# Selected objects #
		####################
		sel_muons = muons[req_event]
		#smu = smu[:, 0]
		sel_electrons = electrons[req_event]
		#sel = sel[:, 0]
		sel_leptons = leptons[req_event]
		sel_jets = jets[req_event]

		####################
		# Weight & Geninfo #
		####################
		weights = Weights(len(events[req_event]), storeIndividual=True)
		if not isRealData:
			weights.add("genweight", events[req_event].genWeight)
			par_flav = (sel_jets.partonFlavour == 0) & (sel_jets.hadronFlavour == 0)
			genflavor = sel_jets.hadronFlavour + 1 * par_flav
			if len(self.SF_map.keys()) > 0:
				syst_wei = True if self.isSyst != None else False
				if "PU" in self.SF_map.keys():
					puwei(
						events[req_event].Pileup.nTrueInt,
						self.SF_map,
						weights,
						syst_wei,
					)
				if "MUO" in self.SF_map.keys():
					muSFs(smu, self.SF_map, weights, syst_wei, False)
				if "EGM" in self.SF_map.keys():
					eleSFs(sel, self.SF_map, weights, syst_wei, False)
				if "BTV" in self.SF_map.keys():
					btagSFs(sel_jets, self.SF_map, weights, "DeepJetC", syst_wei)
					btagSFs(sel_jets, self.SF_map, weights, "DeepJetB", syst_wei)
					btagSFs(sel_jets, self.SF_map, weights, "DeepCSVB", syst_wei)
					btagSFs(sel_jets, self.SF_map, weights, "DeepCSVC", syst_wei)
		else:
			genflavor = ak.zeros_like(sel_jets.pt, dtype=int)

		# Systematics information
		if shift_name is None:
			systematics = ["nominal"] + list(weights.variations)
		else:
			systematics = [shift_name]
		exclude_btv = [
			"DeepCSVC",
			"DeepCSVB",
			"DeepJetB",
			"DeepJetB",
		]  # exclude b-tag SFs for btag inputs


		#######################
		#  Create root files  #
		#######################
		if self.isArray:
			# Keep the structure of events and pruned the object size
			pruned_ev = events[req_event]
			pruned_ev.Jet = sel_jets
			pruned_ev.Electron = sel_electrons
			pruned_ev.Muon = sel_muons
			pruned_ev.Lepton = sel_leptons
			# Add custom variables
			if not isRealData:
				pruned_ev["weight"] = weights.weight()
				for ind_wei in weights.weightStatistics.keys():
					pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
						include=[ind_wei]
					)

			pruned_ev['nJets'] = ak.num(sel_jets)
			pruned_ev['Lepton_pt'] = sel_leptons.pt
			pruned_ev['Lepton_eta'] = sel_leptons.eta
			pruned_ev['Lepton_phi'] = sel_leptons.phi
			pruned_ev['Lepton_mass'] = sel_leptons.mass
			pruned_ev['nLeptons'] = ak.num(sel_leptons)
			pruned_ev['channel'] = channel[req_event]
			pruned_ev['trig_bit'] = output["ttbar_trigWord"][req_event]

			# Create a list of variables want to store. For objects from the PFNano file, specify as {object}_{variable}, wildcard option only accepted at the end of the string
			out_branch = np.setdiff1d(
				np.array(pruned_ev.fields), np.array(events.fields)
			)
			for kin in ["pt", "eta", "phi", "mass", "dz", "dxy"]:
				for obj in ["Jet", "Electron", "Muon", "Lepton"]:
					if ((obj == "Jet") or (obj == "Lepton")) and "d" in kin:
						print(f"{obj}_{kin}")
						continue
					out_branch = np.append(out_branch, [f"{obj}_{kin}"])
			out_branch = np.append(
				out_branch,
				[
					"Jet_btagDeep*",
					"Jet_DeepJet*",
					"nJets",
					"PFCands_*",
					"Electron_pfRelIso03_all",
					"Muon_pfRelIso03_all",
					"nLeptons",
					"channel",
					"trig_bit",
				],
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
