python .\generateSims.py -s human  -ofp ./Results/Figure1.pkl
python .\generateSims.py -s human,cosine -ofp ./Results/Figure2.pkl
python .\generateSims.py -s human,weighted -ofp ./Results/Figure3.pkl
python .\generateSims.py -s human,pruned  -ofp ./Results/Figure4.pkl
python .\generateSims.py -s human,semantic -ofp ./Results/Figure5.pkl
python .\generateSims.py -s human,ibis --individual -ofp ./Results/Figure6.pkl
python .\generateSims.py -s human,cosine,weighted,pruned,semantic,ibis --individual -ofp ./Results/Figure7.pkl

python .\generatePlots.py  -dp ./Results/Figure1.pkl -ofp ./Figures/Figure1.png
python .\generatePlots.py  -dp ./Results/Figure2.pkl -ofp ./Figures/Figure2.png
python .\generatePlots.py  -dp ./Results/Figure3.pkl -ofp ./Figures/Figure3.png
python .\generatePlots.py  -dp ./Results/Figure4.pkl -ofp ./Figures/Figure4.png
python .\generatePlots.py  -dp ./Results/Figure5.pkl -ofp ./Figures/Figure5.png
python .\generatePlots.py  -dp ./Results/Figure6.pkl -ofp ./Figures/Figure6.png
python .\generatePlots.py  -dp ./Results/Figure7a.pkl -ofp ./Figures/Figure7a.png
python .\generatePlots.py  -dp ./Results/Figure7b.pkl -ofp ./Figures/Figure7b.png
python .\generatePlots.py  -dp ./Results/Figure7c.pkl -ofp ./Figures/Figure7c.png
python .\generatePlots.py  -dp ./Results/Figure7d.pkl -ofp ./Figures/Figure7d.png
python .\generatePlots.py  -dp ./Results/Figure7e.pkl -ofp ./Figures/Figure7e.png