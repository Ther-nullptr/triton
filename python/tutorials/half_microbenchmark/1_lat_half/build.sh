nvcc --resource-usage -Xptxas -dlcm=cv -Xptxas -dscm=wt \
     -gencode=arch=compute_80,code=\"sm_80,compute_80\" \
     lat_half.cu -o lat_half -I./ -L -lcudart