#!/bin/bash

main_path=/home/meichen/work1/SR_Attn/all_events

# for filename in $(ls -d */);
# do
#     cd $main_path/$filename/
#     if [[ -d stations ]]
#     then
#         rm -r stations
#     fi
#     cd $main_path/$filename/waveforms/
#     rm *.mseed
#     rm *.dataless
#     cd $main_path/
# done

for event in $(gawk '{print $1}' uni_id.txt)
do
    eventid=` echo $event | gawk -F, '{print $1}'`
    evla=` echo $event | gawk -F, '{print $3}'`
    evlo=` echo $event | gawk -F, '{print $4}'`
    evdp=` echo $event | gawk -F, '{print $5}'`
    cd $main_path/event_$eventid/waveforms

#    if [[ -d BHE ]]
#    then
#        rm -r BHE BHN BHZ
#    fi
#    mkdir BHE BHN BHZ
#    echo "cp *.BHE.*.SAC BHE/" | sh  
#    echo "cp *.BHN.*.SAC BHN/" | sh
#    echo "cp *.BHZ.*.SAC BHZ/" | sh
#    cd $main_path/event_$eventid/waveforms/BHE/
#    bash $HOME/bin/snr_S.sh
#    cd $main_path/event_$eventid/waveforms/BHN/
#    bash $HOME/bin/snr_S.sh
#    cd $main_path/event_$eventid/waveforms/BHZ/
#    bash $HOME/bin/snr_P.sh

    cd $main_path/event_$eventid/waveforms/BHN/
#    mkdir gcarc_30 gcarc_30_85
#    saclst gcarc f *.SAC | gawk '$2<30 {print "mv",$1,"./gcarc_30/"}' | sh
#    saclst gcarc f *.SAC | gawk '$2>30&&$2<85 {print "mv",$1,"./gcarc_30_85/"}' | sh
    rm -r gcarc*
#sac<<EOF
#r *[0-9].[A-F]*.BHN.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[G-N]*.BHN.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[O-Z]*.BHN.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#q
#EOF
#bash mark_sac.sh
    cd $main_path/event_$eventid/waveforms/BHZ/
#    mkdir gcarc_30 gcarc_30_85
#    saclst gcarc f *.SAC | gawk '$2<30 {print "mv",$1,"./gcarc_30"}' | sh
#    saclst gcarc f *.SAC | gawk '$2>30&&$2<85 {print "mv",$1,"./gcarc_30_85"}' | sh
    rm -r gcarc*
#sac<<EOF
#r *[0-9].[A-F]*.BHZ.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[G-N]*.BHZ.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[O-Z]*.BHZ.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#q
#EOF
#bash mark_sac.sh
    cd $main_path/event_$eventid/waveforms/BHE/
#    mkdir gcarc_30 gcarc_30_85
#    saclst gcarc f *.SAC | gawk '$2<30 {print "mv",$1,"./gcarc_30"}' | sh
#    saclst gcarc f *.SAC | gawk '$2>30&&$2<85 {print "mv",$1,"./gcarc_30_85"}' | sh
    rm -r gcarc*

#sac<<EOF
#r *[0-9].[A-F]*.BHE.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[G-N]*.BHE.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#r *[0-9].[O-Z]*.BHE.?.SAC
#ch evla $evla
#ch evlo $evlo
#ch evdp $evdp
#wh
#q
#EOF
#bash mark_sac.sh
done 
