prefix=/data/staff/tomograms/vviknik/laminography_data/
# python read_data.py $prefix
# python sort.py $prefix 1870 168 7
# for k in {0..167..4}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_full.py $prefix $(($k+$j)) 1870 1 1024 7 &                
#    done
#    wait
# done
# python find_shifts_full.py $prefix 1870 168 7

# for k in {0..167..4}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $prefix $(($k+$j)) 1000 1 512 7 &                
#    done
#    wait
# done

# python find_shifts_crop.py $prefix 1000 168 7    
#python prealign_sift.py $prefix 1000 168 7    
#CUDA_VISIBLE_DEVICES=0 python rec_lam.py $prefix 1000 128 7 1 &
#CUDA_VISIBLE_DEVICES=1 python rec_lam_matlab.py $prefix 1000 128 7 1 &
python admm.py $prefix 1000 0