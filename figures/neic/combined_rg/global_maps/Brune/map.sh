#!/bin/bash

main_path=/home/meichen/Research/SR_Attn/pair_events/figures/neic/S_rg/map

PS=map.ps
J=H6i
R=g



##---- Global constants ----##
gmt gmtset MAP_FRAME_TYPE plain
gmt gmtset MAP_FRAME_PEN 0.1p,black
gmt gmtset MAP_DEGREE_SYMBOL degree
gmt gmtset MAP_TICK_LENGTH 0.1
gmt gmtset FONT_LABEL 6p
gmt gmtset FONT_ANNOT_PRIMARY 10p


##---- Global stations and master events ----##
# Japan islands
gmt pscoast -JG141/37/1.5i -Rg -B -Df -A5000 -W0.05p,pink -Gpink -Sthistle -K   > $PS
gawk -v FS=',' 'NR>1{if($5>=125&&$5<=158&&$4>=13&&$4<=60)print "cat "$1"_*_30*.txt"}' pairsfile_rgs_select.csv | sh | gmt psxy -R -J -St0.05c -Wwhite -Gwhite -K -O -N  >> $PS
gawk 'NR>1{if($4>=125&&$4<=158&&$3>=14&&$3<=60)print $4,$3}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
gawk 'NR>1{if($4>=125&&$4<=158&&$3>=14&&$3<=60)print $4,$3}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
# Philippine Sea region
gmt pscoast -JG124/0/1.5i -Rg -B -Df -A5000 -W0.05p,pink -Gpink -Sthistle -K -O  -X1.6i >> $PS
gawk -v FS=',' 'NR>1{if($5>=120&&$5<=128&&$4>=-10&&$4<=10)print "cat "$1"_*_30*.txt"}' pairsfile_rgs_select.csv | sh | gmt psxy -R -J -St0.05c -Wwhite -Gwhite -K -O -N  >> $PS
gawk 'NR>1{if($4>=120&&$4<=128&&$3>=-10&&$3<=10)print $4,$3}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
gawk 'NR>1{if($4>=120&&$4<=128&&$3>=-10&&$3<=10)print $4,$3}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
# Fiji Tonga region
gmt pscoast -JG180/-20/1.5i -Rg -B -Df -A5000 -W0.05p,pink -Gpink -Sthistle -K -O  -X1.6i >> $PS
gawk -v FS=',' 'NR>1{if($5>=174||$5<=-173&&$4>=-25&&$4<=-15)print "cat "$1"_*_30*.txt"}' pairsfile_rgs_select.csv | sh | gmt psxy -R -J -St0.05c -Wwhite -Gwhite -K -O -N  >> $PS
gawk 'NR>1{if($4>=174||$4<=-173&&$3>=-25&&$3<=-15)print $4,$3}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
gawk 'NR>1{if($4>=174||$4<=-173&&$3>=-25&&$3<=-15)print $4,$3}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
# South America
gmt pscoast -JG-66/-18/1.5i -Rg -B -Df -A5000 -W0.05p,pink -Gpink -Sthistle -K -O  -X1.6i >> $PS
gawk -v FS=',' 'NR>1{if($5>=-76&&$5<=-57&&$4>=-32&&$4<=-4)print "cat "$1"_*_30*.txt"}' pairsfile_rgs_select.csv | sh | gmt psxy -R -J -St0.05c -Wwhite -Gwhite -K -O -N  >> $PS
gawk 'NR>1{if($4>=-76&&$4<=-57&&$3>=-32&&$3<=-4)print $4,$3}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS
gawk 'NR>1{if($4>=-76&&$4<=-57&&$3>=-32&&$3<=-4)print $4,$3}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N  >> $PS

##---- Zoom-in subplots ----##
gmt gmtset FONT_ANNOT_PRIMARY 6p
gmt gmtset COLOR_NAN grey@100
gmt gmtset FONT_TITLE 10p
gmt gmtset MAP_TITLE_OFFSET 7p
gmt gmtset MAP_ANNOT_OFFSET 3p

gmt makecpt -Chot -T400/700 -Iz >slab.cpt
# Japan islands
GRD_izu=izu_slab2_dep_02.24.18.grd
GRD_kur=kur_slab2_dep_02.24.18.grd
gmt grdgradient $GRD_izu -Nt -A0 -Gizu.nc
gmt grdgradient $GRD_kur -Nt -A0 -Gkur.nc
#gmt grdimage -R125/150/14/47 -JM1i $GRD_kur -Ikur.nc -Cslab.cpt -K -O  -Y3.3i >> $PS
#gmt grdimage -R -J $GRD_izu -Iizu.nc -Cslab.cpt -K -O  >> $PS
gmt pscoast -JM0.9i -R125/158/14/60 -BWSne -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -Y2.3i -X-4.5i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O  -t20 >> $PS
gmt pscoast -JM0.9i -R125/158/14/60 -BWnse+t"(a)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -Ya1.8i  >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O  -t20 -Ya1.8i >> $PS

# Philippine
GRD_sum=sum_slab2_dep_02.23.18.grd
GRD_sul=sul_slab2_dep_02.23.18.grd
GRD_phi=phi_slab2_dep_02.26.18.grd
gmt grdgradient $GRD_sum -Nt -A0 -Gsum.nc
gmt grdgradient $GRD_sul -Nt -A0 -Gsul.nc
gmt grdgradient $GRD_phi -Nt -A0 -Gphi.nc
#gmt grdimage -R120/125/-10/9 -J $GRD_sum -Isum.nc -Cslab.cpt -K -O  -X1.65i >> $PS
#gmt grdimage -R -J $GRD_sul -Isul.nc -Cslab.cpt -K -O  >> $PS
#gmt grdimage -R -J $GRD_phi -Iphi.nc -Cslab.cpt -K -O  >> $PS
gmt pscoast -JM1.2i -R122/125/2/6.2 -BWSne -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -X1.38i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O  -t20 >> $PS
gmt pscoast -JM1.2i -R122/125/2/6.2 -BWnse+t"(b)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -Ya1.8i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O  -t20 -Ya1.8i >> $PS

# Fiji Tonga
GRD_ker=ker_slab2_dep_02.24.18.grd
gmt grdgradient $GRD_ker -NT -A0 -Gker.nc
#gmt grdimage -R178/181/-25/-15 -J $GRD_ker -Iker.nc -Cslab.cpt -K -O  -X1.65i >> $PS
gmt pscoast -JM1.2i -R177/184/-25/-16 -BWSne -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -X1.72i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O  -t20 >> $PS
gmt pscoast -JM1.2i -R177/184/-25/-16 -BWnse+t"(c)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -Ya1.8i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O  -t20 -Ya1.8i >> $PS

# South America
GRD_sam=sam_slab2_dep_02.23.18.grd
gmt grdgradient $GRD_sam -Nt -A0 -Gsam.nc
#gmt grdimage -R-76/-57/-32/-4 -J $GRD_sam -Isam.nc -Cslab.cpt -K -O  -X1.65i >> $PS
gmt pscoast -JM1.05i -R-76/-57/-32/-4 -BWSne -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -X1.7i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O  -t20 >> $PS
##---- Plot labels ----##
gmt psxy -R -J -Sdp -W0.5p,black -K -O -N  >> $PS <<EOF
-51 -8 8
-51 -14 12
-51 -20 16
-51 -26 20
EOF
gmt pstext -R -J -F+f8p -K -O -N  >> $PS <<EOF
-44 -8 0.01
-44 -14 1
-44 -20 100
-44 -26 10000
EOF
gmt pstext -R -J -F+f9p -K -O -N  >> $PS <<EOF
-51 -3 S
-44 -3 @~\104\163@~ (MPa)
EOF
gmt pscoast -JM1.05i -R-76/-57/-32/-4 -BWsne+t"(d)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O  -Ya1.8i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O  -t20 -Y1.8i >> $PS
gmt psxy -R -J -Scp -W0.5p,black -K -O -N >> $PS <<EOF
-51 -8 8
-51 -14 12
-51 -20 16
-51 -26 20
EOF
gmt pstext -R -J -F+f8p -K -O -N  >> $PS <<EOF
-44 -8 0.01
-44 -14 1
-44 -20 100
-44 -26 10000
EOF
gmt pstext -R -J -F+f9p -K -O -N  >> $PS <<EOF
-51 -3 P
-44 -3 @~\104\163@~ (MPa)
EOF


gmt gmtset FONT_ANNOT_PRIMARY 10p
gmt psscale -Bxa100 -By+Lkm -Dx-4.5i/-2.2i+w5.7i/0.15i+h -Cslab.cpt -O >> $PS

gmt psconvert -Tf $PS
