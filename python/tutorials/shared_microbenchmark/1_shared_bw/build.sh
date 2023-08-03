nvcc --resource-usage -Xptxas -dlcm=cv -Xptxas -dscm=wt  \
     -gencode=arch=compute_80,code=\"sm_80,compute_80\" \
      shared_bw.cu -o shared_bw -I./ -L -lcudart