prefix=/data/staff/tomograms/vviknik/laminography_data/
#python read_data.py $prefix
#python sort.py $prefix 1870 168 7
# for k in {8..167..12}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_full.py $prefix $(($k+$j)) 1870 1 256 7 &                
#    done
#    wait
# done
# python find_shifts_full.py $prefix 1870 168 7

# for k in {8..167..12}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $prefix $(($k+$j)) 1000 1 1024 7 &                
#    done
#    wait
# done

# python prealign_sift.py $prefix 1000 168 7    
# for k in {8..167..12}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop2.py $prefix $(($k+$j)) 800 1 1024 7 &                
#    done
#    wait
# done
# python prealign_cm.py $prefix 800 168 7    
# for k in {0..167..12}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop3.py $prefix $(($k+$j)) 800 1 1024 7 &                
#    done
#    wait
# done


#align solveptycho initptycho recover prb
CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 512 1 0 1 0 &
CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 512 1 1 0 0 &
CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 512 1 1 1 0 &
CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 512 0 0 1 0 &
#CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 800 1 &
# CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 1000 0 &
# CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 1000 1 &
# CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 900 0 &
# CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 900 1 &
# CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 175 1 &
# CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 80 1 &
