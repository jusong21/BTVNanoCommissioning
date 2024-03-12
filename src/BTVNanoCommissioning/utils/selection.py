import awkward as ak


## Jet pu ID not exist in Winter22Run3 sample
def jet_id(events, campaign):
	if campaign == "Rereco17_94X":
		jetmask = (
			(events.Jet.pt > 20)
			& (abs(events.Jet.eta) <= 2.5)
			& (events.Jet.jetId >= 5)
			& ((events.Jet.pt > 50) | (events.Jet.puId >= 7))
		)
	elif "Run3" in campaign:
		## Modified based on Run3 changes included by Annika:
		# https://github.com/AnnikaStein/BTVNanoCommissioning/commit/24237031f4deef30f524851646d156d000a8d4cf
		jetmask = (
			(events.Jet.pt > 20)
			& (abs(events.Jet.eta) <= 2.5)
			& (events.Jet.jetId >= 5)
			& (events.Jet.chHEF > 0.01)
		)
	else:
		jetmask = (
			(events.Jet.pt > 30)
			& (abs(events.Jet.eta) < 2.4)
			& (events.Jet.jetId > 4)
		)
	return jetmask


## FIXME: Electron cutbased Id & MVA ID not exist in Winter22Run3 sample
def ele_cuttightid(events, campaign):
	eleEtaGap  = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
	elePassDXY = (abs(events.Electron.eta) <= 1.479) & (abs(events.Electron.dxy)<0.05) | (abs(events.Electron.eta)>1.479) & (abs(events.Electron.dxy) < 0.1)
	elePassDZ = (abs(events.Electron.eta) <= 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)
	elemask = (
		(events.Electron.pt > 20)
		& (abs(events.Electron.eta) < 2.4)
		& eleEtaGap
		& elePassDXY
		& elePassDZ
		& (events.Electron.cutBased >= 4)
	)
	return elemask


def ele_mvatightid(events, campaign):
	elemask = (
		(abs(events.Electron.eta) < 1.4442)
		| ((abs(events.Electron.eta) < 2.5) & (abs(events.Electron.eta) > 1.566))
	) & (events.Electron.mvaIso_WP80 > 0.5)

	return elemask


def softmu_mask(events, campaign):
	softmumask = (
		(events.Muon.pt < 25)
		& (abs(events.Muon.eta) < 2.4)
		& (events.Muon.tightId > 0.5)
		& (events.Muon.pfRelIso04_all > 0.2)
		& (events.Muon.jetIdx != -1)
	)

	return softmumask


def mu_idiso(events, campaign):
	mumask = (
		(events.Muon.pt > 20)
		& (abs(events.Muon.eta) < 2.4)
		& events.Muon.tightId
		& (events.Muon.pfRelIso04_all < 0.15)
	)
	return mumask


def btag_mu_idiso(events, campaign):
	mumask = (
		(abs(events.Muon.eta) < 2.4)
		& (events.Muon.tightId > 0.5)
		& (events.Muon.pfRelIso04_all < 0.12)
	)
	return mumask

btag_wp_dict = {
	"2016preVFP_UL": {
		"DeepFlav": {
			"b": {
				"L": 0.0505,
				"M": 0.2598,
				"T": 0.6502,
			},
			"c": {
				"L": [0.039, 0.327], # CvL, then CvB
				"M": [0.098, 0.370],
				"T": [0.270, 0.256],
			}
		}
	},
	"2016postVFP_UL": {
		"DeepFlav": {
			"b": {
				"L": 0.0480,
				"M": 0.2489,
				"T": 0.6377,
			},
			"c": {
				"L": [0.039, 0.305],
				"M": [0.099, 0.353],
				"T": [0.269, 0.247],
			}
		}
	},
	"2017_UL": {
		"DeepFlav": {
			"b": {
				"L": 0.532,
				"M": 0.3040,
				"T": 0.7476,
			},
			"c": {
				"L": [0.030, 0.400],
				"M": [0.085, 0.340],
				"T": [0.520, 0.050],
			}
		}
	},
	"2018_UL": {
		"DeepFlav": {
			"b": {
				"L": 0.0490,
				"M": 0.2783,
				"T": 0.7100,
			},
			"c": {
				"L": [0.038, 0.246],
				"M": [0.099, 0.325],
				"T": [0.282, 0.267],
			}
		}
	}
}

def btag_wp(jets, campaign, tagger, borc, wp):
	WP = btag_wp_dict[campaign]
	if borc == "b":
		jet_mask = jets[f"btag{tagger}B"] > WP[tagger]["b"][wp]
	else:
		jet_mask = (jets[f"btag{tagger}CvB"] > WP[tagger]["c"][wp][1]) & (
			jets[f"btag{tagger}CvL"] > WP[tagger]["c"][wp][0]
		)
	return jet_mask
