#!/bin/bash

PS=map.ps

##---- Global constants ----##
gmt gmtset MAP_FRAME_TYPE plain
gmt gmtset MAP_FRAME_PEN 0.5p,black
gmt gmtset MAP_TICK_LENGTH 0.1
gmt gmtset FONT_LABEL 4p
gmt gmtset FONT_ANNOT_PRIMARY 6p

##---- Global stations ----##
gmt pscoast -JW4.5i -Rg -Dc -B+t"" -Ba60 -A40000 -Ggrey -K > $PS
cat station_loc_dd.txt | gmt psxy -R -J -St0.08c -Wfaint,black@50 -Gblack@50 -K -O -N >> $PS

##---- Global master events ----##
#gmt makecpt -Ccool -T400/700 -Iz > mycpt.cpt
gawk 'NR>1{print $4,$3,$5}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.3c -W0.05p,black -Gred -K -O -N >> $PS
gawk 'NR>1{print $4,$3,$5}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.3c -W0.05p,black -Gred -K -O -N >> $PS

##---- Text ----##
gmt pstext -J -R -F+f -O -N >> $PS <<EOF
210 -35 8 Fiji\ Tonga
160 20 8 Northwest\ Pacific
260 0 8 South\ America
EOF

##---- Square of Fiji region ----##
#gmt psxy -R -J -W0.4p,black,.. -K -O -N >> $PS <<EOF
#172 -32
#172 -13
#187 -13
#187 -32
#172 -32
#EOF

##---- Colorbar ----##
#gmt psscale -Bxa100 -Dx2.3i/0.5i+w0.7i/0.05i+h -Cmycpt.cpt -O >> $PS

gmt psconvert -Tf $PS
