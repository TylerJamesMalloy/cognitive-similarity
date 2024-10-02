:: python .\generateSims.py -s human,human -ofp ./Results/Figure0.pkl
:: python .\generateSims.py -s human,semantic -ofp ./Results/Figure1.pkl
:: python .\generateSims.py -s human,cosine -ofp ./Results/Figure2.pkl
:: python .\generateSims.py -s human,weighted -ofp ./Results/Figure3.pkl
:: python .\generateSims.py -s human,pruned  -ofp ./Results/Figure4.pkl
:: python .\generateSims.py -s human,ensemble -ofp ./Results/Figure5.pkl
:: python .\generateSims.py -s human,ibis -ofp ./Results/Figure6.pkl
python .\generateSims.py -s human,pruned -ofp ./Results/Figure7a.pkl -i
python .\generateSims.py -s human,ensemble -ofp ./Results/Figure7b.pkl -i 
python .\generateSims.py -s human,ibis -ofp ./Results/Figure7c.pkl -i

python .\generatePlots.py  -dp ./Results/Figure0.pkl -ofp ./Figures/Figure0.png -ll center-right
python .\generatePlots.py  -dp ./Results/Figure1.pkl -ofp ./Figures/Figure1.png -ll center-right
python .\generatePlots.py  -dp ./Results/Figure2.pkl -ofp ./Figures/Figure2.png -ll center-right
python .\generatePlots.py  -dp ./Results/Figure3.pkl -ofp ./Figures/Figure3.png
python .\generatePlots.py  -dp ./Results/Figure4.pkl -ofp ./Figures/Figure4.png
python .\generatePlots.py  -dp ./Results/Figure5.pkl -ofp ./Figures/Figure5.png
python .\generatePlots.py  -dp ./Results/Figure6.pkl -ofp ./Figures/Figure6.png

python .\generatePlots.py  -dp ./Results/Figure7a.pkl -ofp ./Figures/Figure7a.png 