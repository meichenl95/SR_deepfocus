#!/bin/bash

PS=map.ps
J=M4i
R=90/300/-40/45

gmt gmtset MAP_FRAME_TYPE plain
gmt pscoast -J$J -R$R -Bxa30f30g60 -Bya30f30g60 -Dc -W1/0 -W2/0 -A5000 -K > $PS

gmt makecpt -Ccool -T4/9 -A50 > Icpt.cpt
gawk '{print $4,$3,$2}' fc_logtao_total_P_rg.txt | gmt psxy -R -J -Sc3p -K -O -CIcpt.cpt -N >> $PS
gmt psscale -CIcpt.cpt -Baf -Dx0/0+w2i/0.1i+v -X4.5i -O >> $PS
