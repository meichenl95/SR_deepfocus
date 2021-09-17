#!/bin/bash

main_path=/home/meichen/Research/SR_Attn/pair_events/figures/neic/S_rg/map

PS=map.ps
J=H6i
R=g



##---- Global constants ----##
gmt gmtset MAP_FRAME_TYPE fancy
gmt gmtset MAP_FRAME_WIDTH 0.055
gmt gmtset MAP_DEGREE_SYMBOL degree
gmt gmtset MAP_TICK_LENGTH 0.1
gmt gmtset FONT_LABEL 6p
gmt gmtset FONT_ANNOT_PRIMARY 10p

gmt pscoast -J$J -R$R -Bya45 -Df -A10000 -W0.05p,black -Ggray -K -P -Y4i > $PS

##---- Global stations and master events ----##
cat *_30.txt *_30_85.txt | gmt psxy -R -J -St0.2c -W0.01p,black -Gwhite -K -O -N -P >> $PS
gawk '{print $4,$3}' fc_logtao_rgs.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N -P >> $PS
gawk '{print $4,$3}' fc_logtao_rgp.txt | gmt psxy -R -J -Sa0.2c -W0.03p,magenta -Gmagenta -K -O -N -P >> $PS

##---- Squares of four regions ----##
# Japan islands
gmt psxy -R -J -W0.7p,magenta,-- -K -O -N -P >> $PS <<EOF
125 14
150 14
150 50
125 50
125 14
EOF
# Philippine Sea region
gmt psxy -R -J -W0.7p,magenta,-- -K -O -N -P >> $PS <<EOF
115 12
128 12
128 -12
115 -12
115 12
EOF
# Fiji Tonga region
gmt psxy -R -J -W0.7p,magenta,-- -K -O -N -P >> $PS <<EOF
174 -15
-173 -15
-173 -33
174 -33
174 -15
EOF
# South America
gmt psxy -R -J -W0.7p,magenta,-- -K -O -N -P >> $PS <<EOF
-76 -4
-57 -4
-57 -32
-76 -32
-76 -4
EOF

##---- Plot labels ----##
gmt pstext -R -J -F+f10p,magenta -K -O -N -P >> $PS <<EOF
160 30 (a)
135 5 (b)
188 -5 (c)
-85 -15 (d)
EOF

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
#gmt grdimage -R125/150/14/47 -JM1i $GRD_kur -Ikur.nc -Cslab.cpt -K -O -P -Y3.3i >> $PS
#gmt grdimage -R -J $GRD_izu -Iizu.nc -Cslab.cpt -K -O -P >> $PS
gmt pscoast -JM0.9i -R125/150/14/50 -BWNse+t"(a)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O -P -Y3.7i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS

# Philippine
GRD_sum=sum_slab2_dep_02.23.18.grd
GRD_sul=sul_slab2_dep_02.23.18.grd
GRD_phi=phi_slab2_dep_02.26.18.grd
gmt grdgradient $GRD_sum -Nt -A0 -Gsum.nc
gmt grdgradient $GRD_sul -Nt -A0 -Gsul.nc
gmt grdgradient $GRD_phi -Nt -A0 -Gphi.nc
#gmt grdimage -R120/125/-10/9 -J $GRD_sum -Isum.nc -Cslab.cpt -K -O -P -X1.65i >> $PS
#gmt grdimage -R -J $GRD_sul -Isul.nc -Cslab.cpt -K -O -P >> $PS
#gmt grdimage -R -J $GRD_phi -Iphi.nc -Cslab.cpt -K -O -P >> $PS
gmt pscoast -JM0.9i -R115/126/-10/9.5 -BWNse+t"(b)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O -P -X1.33i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS

# Fiji Tonga
GRD_ker=ker_slab2_dep_02.24.18.grd
gmt grdgradient $GRD_ker -NT -A0 -Gker.nc
#gmt grdimage -R178/181/-25/-15 -J $GRD_ker -Iker.nc -Cslab.cpt -K -O -P -X1.65i >> $PS
gmt pscoast -JM0.62i -R176/184/-33/-14 -BWNse+t"(c)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O -P -X1.4i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS

# South America
GRD_sam=sam_slab2_dep_02.23.18.grd
gmt grdgradient $GRD_sam -Nt -A0 -Gsam.nc
#gmt grdimage -R-76/-57/-32/-4 -J $GRD_sam -Isam.nc -Cslab.cpt -K -O -P -X1.65i >> $PS
gmt pscoast -JM1i -R-76/-57/-32/-4 -BWNse+t"(d)" -Ba -Df -W0.01p,black -Ggray -A1000 -K -O -P -X1.1i >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgs.txt | gmt psxy -R -J -Scp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS
gawk '{print $4,$3,$5,$2*2}' fc_logtao_rgp.txt | gmt psxy -R -J -Sdp -Cslab.cpt -W0.5p,black -K -O -P -t20 >> $PS

##---- Plot labels ----##
gmt psxy -R -J -Scp -W0.5p,black -K -O -N -P >> $PS <<EOF
-51 -8 8
-51 -14 12
-51 -20 16
-51 -26 20
EOF
gmt psxy -R -J -Sdp -W0.5p,black -K -O -N -P >> $PS <<EOF
-44 -8 8
-44 -14 12
-44 -20 16
-44 -26 20
EOF
gmt pstext -R -J -F+f8p -K -O -N -P >> $PS <<EOF
-37 -8 0.01
-37 -14 1
-37 -20 100
-37 -26 10000
EOF
gmt pstext -R -J -F+f9p -K -O -N -P >> $PS <<EOF
-51 -3 S
-44 -3 P
-37 -3 @~\104\163@~ (MPa)
EOF

gmt gmtset FONT_ANNOT_PRIMARY 10p
gmt psscale -Bxa100f50 -By+Lkm -Dx-3.7i/-0.4i+w5i/0.15i+h -Cslab.cpt -O >> $PS

gmt psconvert -Tf $PS
