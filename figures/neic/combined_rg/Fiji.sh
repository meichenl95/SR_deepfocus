#!/bin/bash

PS=Fiji.ps
J=M2i
R=178/183/-25/-15

gmt gmtset MAP_FRAME_TYPE fancy
gmt gmtset MAP_FRAME_WIDTH 0.1
gmt pscoast -J$J -R$R -Bxa2f1 -Bya2f1 -Dc -W1/0 -W2/0 -A5000 -K -Y2i > $PS

GRD=GMRTv3_6_Fijitopo.grd
gmt grdgradient $GRD -Nt -A0 -Ggrid_grad.nc
gmt makecpt -Cglobe -A50 > Itopo.cpt
gmt grdimage -R$R -J$J $GRD -Igrid_grad.nc -CItopo.cpt -K -O >>$PS

gmt makecpt -Cjet -T0/780 > slab.cpt
gawk '!/?/{print $1,$2,(-1)*$3}' ker_slab2_dep_02.24.18_contours.in | gmt psxy -J$J -R$R -Cslab.cpt -Sc1p -K -O >> $PS


gmt makecpt -Cjet -A20 -T-2/4 > Icpt.cpt
#gawk '{print $3,$2,log($1)/log(10)-6}' PS.txt | gmt psxy -R -J -Sd8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $3,$2,log($1)/log(10)-6}' P.txt | gmt psxy -R -J -Sc8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $5,$4,log($2)/log(10)-5}' P_rg_tao.txt | gmt psxy -R -J -Sc8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
#gawk '{print $3,$2,log($1)/log(10)-6}' S.txt | gmt psxy -R -J -St8p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
gawk '{print $5,$4,log($2)/log(10)-6}' S_rg_tao.txt | gmt psxy -R -J -St12p -W0.5p,black -K -O -CIcpt.cpt -N >> $PS
gawk '{print $6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17}' S_rg_tao.txt | gmt psmeca -R -J -Sm0.4c -L0.4p -K -O >> $PS
gmt psscale -Dx3.5i/0i+w4i/0.2i+v -Ba1f0.5 -CIcpt.cpt -O -K >> $PS
gmt psscale -Dx3.5i/0i+w4i/0.2i+v+macl -Ba100f50 -Cslab.cpt -O >> $PS
gmt psconvert -Tf $PS
