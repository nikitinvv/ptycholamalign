prefix=/data/staff/tomograms/vviknik/laminography_data/
#prefix=/local/data/lamino_data/
# python read_data.py $prefix
# python sort.py $prefix 1870 168 7
# for k in {0..167..4}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_full.py $prefix $(($k+$j)) 1870 1 64 7 &            
#    done
#    wait
# done

# python prealign_cm.py $prefix 1870 168 7 1280    
# for k in {0..167..4}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $prefix $(($k+$j)) 1300 1 1024 7 &                 
#    done
#    wait
# done
# python prealign_cm.py $prefix 1300 168 7 768    
# for k in {0..167..4}; 
# do
#    for j in {0..3}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_crop2.py $prefix $(($k+$j)) 800 1 512 7 &            
#    done
#    wait
# done

#align solveptycho initptycho recover prb
#CUDA_VISIBLE_DEVICES=0 python admm2.py $prefix 512 1 0 1 0 &
#CUDA_VISIBLE_DEVICES=1 python admm2.py $prefix 512 0 0 1 0 &
CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 256 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 128 1 1 0 1 &
CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 384 1 1 0 1 &
CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 64 1 1 0 1 &

# CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 512 1 1 0 0 &
#CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 512 1 1 1 0 &
# CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 512 1 1 1 0 &
# CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 512 0 0 1 0 &
#CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 800 1 &
# CUDA_VISIBLE_DEVICES=0 python admm.py $prefix 1000 0 &
# CUDA_VISIBLE_DEVICES=1 python admm.py $prefix 1000 1 &
# CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 900 0 &
# CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 900 1 &
# CUDA_VISIBLE_DEVICES=2 python admm.py $prefix 175 1 &
# CUDA_VISIBLE_DEVICES=3 python admm.py $prefix 80 1 &
