### Fast computing density maps for ShanghaiTech

You can choose how to compute sigma : distance to 3 nearest neighbors, distance to nearest neighbor, or fixed value. Optionally clip sigma if it is lower than threshold. 
I've used pre-computed gaussian kernels and fast neighbors search to speed up density maps creation.
