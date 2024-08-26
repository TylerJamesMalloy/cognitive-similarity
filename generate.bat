python .\generateSims.py -s human  -ofp ./Results/Figure1.pkl
python .\generateSims.py -s human,cosine -ofp ./Results/Figure2.pkl
python .\generateSims.py -s human,weighted -ofp ./Results/Figure3.pkl
python .\generateSims.py -s human,pruned  -ofp ./Results/Figure4.pkl
python .\generateSims.py -s human,semantic -ofp ./Results/Figure5.pkl
python .\generateSims.py -s human,ibis --individual -ofp ./Results/Figure6.pkl
python .\generateSims.py -s human,cosine,weighted,pruned,semantic,ibis --individual -ofp ./Results/Figure7.pkl

python .\generatePlots.py  -fp ./Results/Figure1.pkl
python .\generatePlots.py  -fp ./Results/Figure2.pkl
python .\generatePlots.py  -fp ./Results/Figure3.pkl
python .\generatePlots.py  -fp ./Results/Figure4.pkl
python .\generatePlots.py  -fp ./Results/Figure5.pkl
python .\generatePlots.py  -fp ./Results/Figure6.pkl
python .\generatePlots.py  -fp ./Results/Figure7a.pkl
python .\generatePlots.py  -fp ./Results/Figure7b.pkl
python .\generatePlots.py  -fp ./Results/Figure7c.pkl
python .\generatePlots.py  -fp ./Results/Figure7d.pkl
python .\generatePlots.py  -fp ./Results/Figure7e.pkl