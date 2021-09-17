#!/bin/bash

PS=map.ps
J=H6i
R=g

gmt gmtset MAP_FRAME_TYPE plain
gmt pscoast -J$J -R$R -Bxa30f30g60 -Bya30f30g60 -Dc -W1/0 -W2/0 -A5000 -K -Y2i > $PS

gmt makecpt -Cjet -A50 -T-2/4 > Icpt.cpt
#gawk '{print $3,$2,log($1)/log(10)-6}' PS.txt | gmt psxy -R -J -Sd8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $3,$2,log($1)/log(10)-6}' P.txt | gmt psxy -R -J -Sc8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $3,$2,log($1)/log(10)-6}' S.txt | gmt psxy -R -J -St8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
gawk '{print $5,$4,log($2)/log(10)-6}' S_rg_tao.txt | gmt psxy -R -J -St8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $5,$4,log($2)/log(10)-6}' P_rg_tao.txt | gmt psxy -R -J -Sc8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
gawk '{print $6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17}' S_rg_tao_1000.txt | gmt psmeca -R -J -Sm0.4c -Gblack@50 -L0.4p -K -O >> $PS
gmt psscale -Dx-4i/-0.7i+w4i/0.2i+h -Ba1f0.5 -CIcpt.cpt -X4i -O >> $PS
gmt psconvert -Tf $PS
