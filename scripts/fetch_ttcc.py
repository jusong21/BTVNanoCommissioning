import os
import json

path = "/pnfs/iihe/cms/store/user/jusong/topNanoAOD/v9-1-1"
years = ["2016ULpostVFP", "2016ULpreVFP", "2017UL", "2018UL"]
datasets = ['TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8']


for year in years:
	fdict = {}
	for dataset in datasets:

		p = path+'/'+year+'/'+dataset
		print('Get file lists of ',p)
	
		assert len(os.listdir(p.rstrip())) != 0, 'It has multiple sub directories' 
		sub_dir = p+'/'+os.listdir(p.rstrip())[0]

		assert len(os.listdir(sub_dir.rstrip())) != 0, 'It has multiple sub directories' 
		sub_dir = sub_dir+'/'+os.listdir(sub_dir.rstrip())[0]
		# then, sub_dir = '/pnfs/iihe/cms/store/user/jusong/topNanoAOD/v9-1-1/2016ULpreVFP/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TopNanoAODv9-1-1_2016ULpreVFP/231011_065931'

		dirs = os.listdir(sub_dir.rstrip())
		flist = []
		for dir in dirs:
			sub_sub_dir = sub_dir+'/'+dir
			for f in os.listdir(sub_sub_dir):
				file = sub_sub_dir+'/'+f
				flist.append(file)

		fdict[dataset] = flist
	
	output = '/user/jusong/analysis/BTVNanoCommissioning/metadata/ttcc_'+year+'.json'
	with open(output, 'w') as fp:
		json.dump(fdict, fp, indent=4)
		print('The file is saved at: ', output)






