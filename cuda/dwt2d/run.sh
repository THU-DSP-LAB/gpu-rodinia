./dwt2d ../../data/dwt2d/192.bmp 192.bmp.dwt -d 192x192 -c 1 -f -5 -l 3 --verify-cpu --verify-reference ../../data/dwt2d/cuda_192_192x192_c1_53_l3.reference
./dwt2d ../../data/dwt2d/rgb.bmp rgb.bmp.dwt -d 1024x1024 -f -5 -l 3 --verify-reference ../../data/dwt2d/cuda_rgb_1024x1024_c3_53_l3.reference
