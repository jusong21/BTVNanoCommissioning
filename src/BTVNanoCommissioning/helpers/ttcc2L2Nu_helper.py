import awkward as ak
import numba as nb
import numpy as np
import pandas as pd
import coffea.nanoevents.methods.vector as vector
import os, psutil

###############
#  HLT table  #
###############
def sel_HLT(dataset, campaign):
	if 'Run2016H' in dataset:
		print('Run2016H trigger')
		HLT_chns = [
			('HLT_Ele27_WPTight_Gsf', 'ee'),
			('HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('HLT_DoubleEle33_CaloIdL_MW', 'ee'),
			('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL', 'ee'),
			('HLT_IsoMu24', 'mm'),
			('HLT_IsoTkMu24', 'mm'),
			('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'mm'),
			('HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ', 'mm'),
			('HLT_Ele27_WPTight_Gsf', 'em'),
			('HLT_IsoMu24', 'em'),
			('HLT_IsoTkMu24', 'em'),
			('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
			('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
		]
	elif 'Run2017B' in dataset:
		print('Run2017B trigger')
		HLT_chns = [
			('Ele35_WPTight_Gsf', 'ee'),
			('DoubleEle33_CaloIdL_MW', 'ee'),
			('Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('IsoMu27', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'mm'),
			('Ele35_WPTight_Gsf', 'em'),
			('IsoMu27', 'em'),
			('Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'em'),
			('Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
		]
	elif 'Run2017' in dataset:
		print('Run2017C-F trigger')
		HLT_chns = [
			('Ele35_WPTight_Gsf', 'ee'),
			('DoubleEle33_CaloIdL_MW', 'ee'),
			('Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('IsoMu27', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 'mm'),
			('Ele35_WPTight_Gsf', 'em'),
			('IsoMu27', 'em'),
			('Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'em'),
			('Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
		]

	elif ('2016' in campaign) & ('Run2016H' not in dataset):
		print('Run2016B-G & MC trigger')
		HLT_chns = [
			('HLT_Ele27_WPTight_Gsf', 'ee'),
			('HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('HLT_DoubleEle33_CaloIdL_MW', 'ee'),
			('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL', 'ee'),
			('HLT_IsoMu24', 'mm'),
			('HLT_IsoTkMu24', 'mm'),
			('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL', 'mm'),
			('HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL', 'mm'),
			('HLT_Ele27_WPTight_Gsf', 'em'),
			('HLT_IsoMu24', 'em'),
			('HLT_IsoTkMu24', 'em'),
			('HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'em'),
			('HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL', 'em'),
		]
	elif '2017' in campaign:
		print('2017 MC trigger')
		HLT_chns = [
			('Ele35_WPTight_Gsf', 'ee'),
			('DoubleEle33_CaloIdL_MW', 'ee'),
			('Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('IsoMu27', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8', 'mm'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 'mm'),
			('Ele35_WPTight_Gsf', 'em'),
			('IsoMu27', 'em'),
			('Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'em'),
			('Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
		]
	elif '2018' in campaign:
		print('2018 data & MC trigger')
		HLT_chns = [
			('Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 'ee'),
			('Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'ee'),
			('Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 'ee'),
			('Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'em'),
			('Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
			('Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
			('Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'em'),
		]

	print('cam: ', campaign, ' dataset: ', dataset)
	print('cam: ', campaign, ' dataset: ', dataset)
	print(HLT_chns)
	print(HLT_chns)
	return HLT_chns

################
#  Mass table  #
################
# mass table from https://github.com/scikit-hep/particle/blob/master/src/particle/data/particle2022.csv and https://gitlab.cern.ch/lhcb-conddb/DDDB/-/blob/master/param/ParticleTable.txt
df_main = pd.read_csv(
    "src/BTVNanoCommissioning/helpers/particle2022.csv", delimiter=",", skiprows=1
)
df_back = pd.read_csv(
    "src/BTVNanoCommissioning/helpers/ParticleTable.csv", delimiter=",", skiprows=4
)
df_main, df_back = df_main.astype({"ID": int}), df_back.astype({"PDGID": int})
main = dict(zip(df_main.ID, df_main.Mass / 1000.0))
backup = dict(zip(df_back.PDGID, df_back["MASS(GeV)"]))
hadron_mass_table = {**main, **{k: v for k, v in backup.items() if k not in main}}


###############
#  Functions  #
###############
def is_from_GSP(GenPart):
    QGP = ak.zeros_like(GenPart.genPartIdxMother)
    QGP = (
        (GenPart.genPartIdxMother >= 0)
        & (GenPart.genPartIdxMother2 == -1)
        & (abs(GenPart.parent.pdgId) == 21)
        & (
            (abs(GenPart.parent.pdgId) == abs(GenPart.pdgId))
            | abs(GenPart.parent.pdgId)
            == 21
        )
        & ~(
            (abs(GenPart.pdgId) == abs(GenPart.parent.pdgId))
            & (GenPart.parent.status >= 21)
            & (GenPart.parent.status <= 29)
        )
    )
    rest_QGP = ak.mask(GenPart, ~QGP)
    restGenPart = rest_QGP.parent
    while ak.any(ak.is_none(restGenPart.parent.pdgId, axis=-1) == False):
        mask_forbad = (
            (restGenPart.genPartIdxMother >= 0)
            & (restGenPart.genPartIdxMother2 == -1)
            & (
                (abs(restGenPart.parent.pdgId) == abs(restGenPart.pdgId))
                | abs(restGenPart.parent.pdgId)
                == 21
            )
            & ~(
                (abs(restGenPart.pdgId) == abs(GenPart.parent.pdgId))
                & (restGenPart.parent.status >= 21)
                & (restGenPart.parent.status <= 29)
            )
        )
        mask_forbad = ak.fill_none(mask_forbad, False)
        QGP = (mask_forbad & ak.fill_none((abs(restGenPart.pdgId == 21)), False)) | QGP

        rest_QGP = ak.mask(restGenPart, (~QGP) & (mask_forbad))
        restGenPart = rest_QGP.parent
        if ak.all(restGenPart.parent.pdgId == True):
            break

    return ak.fill_none(QGP, False)


@nb.njit
def to_bitwise_trigger(pass_trig, builder):
	for it in pass_trig:
		# group by every 32 bits
		builder.begin_list()
		for bitidx in range(len(it) // 32 + 1):
			trig = 0
			start = bitidx * 32
			end = min((bitidx + 1) * 32, len(it))
			for i, b in enumerate(it[start:end]):
				trig += b << i
			builder.integer(trig)
		builder.end_list()
	return builder


@nb.vectorize([nb.float64(nb.int64)], forceobj=True)
def get_hadron_mass(hadron_pids):
    return hadron_mass_table[abs(hadron_pids)]


def cumsum(array):
    layout = array.layout
    layout.content
    scan = ak.Array(
        ak.layout.ListOffsetArray64(
            layout.offsets, ak.layout.NumpyArray(np.cumsum(layout.content))
        )
    )
    cumsum_array = ak.fill_none(scan - ak.firsts(scan) + ak.firsts(array), [], axis=0)
    return cumsum_array


def calc_ip_vector(obj, dxy, dz, is_3d=False):
    """Calculate the 2D or 3D impact parameter vector, given the track obj (with 4-mom),
    and its dxy and dz, taking the standard definition from NanoAOD"""

    # 2D impact parameter
    pvec = ak.zip(
        {
            "x": obj.px,
            "y": obj.py,
            "z": obj.pz,
        },
        behavior=vector.behavior,
        with_name="ThreeVector",
    )
    zvec = ak.zip(
        {
            "x": ak.zeros_like(dxy),
            "y": ak.zeros_like(dxy),
            "z": ak.zeros_like(dxy) + 1,
        },
        behavior=vector.behavior,
        with_name="ThreeVector",
    )
    # 2D impact parameter vector: (-py, px) / pt * dxy
    ipvec_2d = zvec.cross(pvec) * dxy / obj.pt

    if is_3d == False:
        return ipvec_2d

    # Then calculate the 3D impact parameter vector
    # first, extend ipvec_2d to 3D space
    ipvec_2d_ext = ak.zip(
        {
            "x": ipvec_2d.x,
            "y": ipvec_2d.y,
            "z": dz,
        },
        behavior=vector.behavior,
        with_name="ThreeVector",
    )
    # then, get the closest distance to the track on 3D geometry
    ipvec_3d = ipvec_2d_ext - ipvec_2d_ext.dot(pvec) / pvec.p2 * pvec
    return ipvec_3d
