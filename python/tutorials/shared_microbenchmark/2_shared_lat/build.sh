nvcc --resource-usage -Xptxas -dlcm=cv -Xptxas -dscm=wt  \
    -gencode=arch=compute_80,code=\"sm_80,compute_80\" \
     shared_lat.cu -o shared_lat -I./ -L -lcudart